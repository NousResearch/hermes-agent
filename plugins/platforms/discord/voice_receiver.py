"""Bounded Discord RTP capture, decryption, and per-speaker PCM accounting."""
from __future__ import annotations

import logging
import struct
import subprocess
import threading
from collections import defaultdict
from typing import Any, Dict, Optional

logger = logging.getLogger((__package__ + ".adapter") if __package__ else "adapter")

def _adapter_clock():
    # Keep the adapter's injectable clock shared with the listener's silence threshold.
    if __package__:
        from . import adapter as adapter_module
    else:
        import adapter as adapter_module
    return adapter_module.time

class VoiceReceiver:
    """Captures and decodes voice audio from a Discord voice channel.

    Attaches to a VoiceClient's socket listener, decrypts RTP packets
    (NaCl transport + DAVE E2EE), decodes Opus to PCM, and buffers
    per-user audio.  A polling loop detects silence and delivers
    completed utterances via a callback.
    """

    SILENCE_THRESHOLD = 1.5    # seconds of silence → end of utterance
    MIN_SPEECH_DURATION = 0.5  # minimum seconds to process (skip noise)
    SAMPLE_RATE = 48000        # Discord native rate
    CHANNELS = 2               # Discord sends stereo
    REKEY_FAILURE_STREAK = 25  # consecutive NaCl failures → re-resolve creds
    PCM_BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * 2
    MAX_UTTERANCE_SECONDS = 60.0
    MAX_UTTERANCE_BYTES = int(PCM_BYTES_PER_SECOND * MAX_UTTERANCE_SECONDS)
    MAX_STREAM_QUEUE_SECONDS = 5.0
    MAX_STREAM_QUEUE_BYTES = int(PCM_BYTES_PER_SECOND * MAX_STREAM_QUEUE_SECONDS)
    MAX_ACTIVE_SSRC_BUFFERS = 4

    def __init__(self, voice_client, allowed_user_ids: set | None = None):
        self._vc = voice_client
        self._allowed_user_ids = allowed_user_ids or set()
        self._running = False

        # Decryption state, kept as ONE tuple (secret_key, dave_session,
        # dave_protocol_version, dave_downgraded) so the receive thread
        # reads a consistent generation in a single reference load while
        # refreshes happen on other threads — two separate assignments
        # could pair a new key with an old session for a packet.
        self._creds: tuple = (b"", None, 0, False)
        self._bot_ssrc: int = 0
        # Monotonic deadline while DAVE plaintext passthrough is allowed
        # (mirrors discord.py's set_passthrough_mode windows, see the
        # voice-ws hook).  Replace-semantics: an upgrade's 10s grace
        # SHORTENS a residual downgrade window, never extends it.
        self._dave_passthrough_until: float = 0.0

        # SSRC -> user_id mapping (populated from SPEAKING events)
        self._ssrc_to_user: Dict[int, int] = {}
        self._ssrc_generation: Dict[int, int] = defaultdict(int)
        self._capture_generation = 0
        # Re-entrant because packet-time safe user inference updates the SSRC
        # map through the same lock used by silence extraction.
        self._lock = threading.RLock()

        # Per-user audio buffers
        self._buffers: Dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: Dict[int, float] = {}
        # Decoded chunks awaiting async Realtime STT dispatch. Chunks are
        # queued only after an SSRC has a safely resolved user id.
        self._stream_chunks: list[tuple[int, int, int, int, bytes]] = []
        self._stream_chunk_bytes = 0

        # Opus decoder per SSRC (each user needs own decoder state)
        self._decoders: Dict[int, Any] = {}
        self._hook_connection = None
        self._original_connection_hook = None
        self._original_ws_hook = None
        self._installed_speaking_hook = None

        # Pause flag: don't capture while bot is playing TTS
        self._paused = False
        self._channel_move_pending = False
        self.require_explicit_mapping = False

        # Debug logging counter (instance-level to avoid cross-instance races)
        self._packet_debug_count = 0

        # Decode-health counters (logged at teardown) and the NaCl failure
        # streak used to detect stale credentials after a re-key.
        self._decode_ok = 0
        self._decode_failed = 0
        self._dave_unmapped_dropped = 0
        self._nacl_fail_streak = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start listening for voice packets."""
        conn = self._vc._connection
        self._resolve_credentials(conn)

        self._install_speaking_hook(conn)
        conn.add_socket_listener(self._on_packet)
        with self._lock:
            self._capture_generation += 1
            self._running = True
        logger.info("VoiceReceiver started (bot_ssrc=%d)", self._bot_ssrc)

    def _resolve_credentials(self, conn) -> None:
        """Read the current decryption state from the live connection.

        Discord rotates the transport ``secret_key`` on every voice
        (re)connect (op 4 SESSION_DESCRIPTION), and ``reinit_dave_session``
        REPLACES ``conn.dave_session`` with a new object when none existed
        yet — e.g. when DAVE finishes negotiating after this receiver
        started.  Credentials must therefore be re-resolvable at runtime;
        a one-time snapshot decrypts nothing after either event, silently.

        ``dave_protocol_version`` rides along because a non-null session is
        NOT proof frames are encrypted: after a downgrade transition the
        session object survives with the protocol at 0 and senders emit
        plaintext.
        """
        self._creds = (
            bytes(conn.secret_key),
            conn.dave_session,
            int(getattr(conn, "dave_protocol_version", 0) or 0),
            bool(getattr(conn, "dave_downgraded", False)),
        )
        self._bot_ssrc = conn.ssrc
        # A receiver can start (or refresh) while a downgrade transition is
        # already pending — upstream has passthrough enabled for up to 120s
        # but we never saw the op 21.  Seed the window from the connection's
        # physical pending-transition state; the entry is popped on execute,
        # so this cannot re-grant after the transition completes.
        pending = getattr(conn, "dave_pending_transitions", None) or {}
        if 0 in pending.values() and _adapter_clock().monotonic() >= self._dave_passthrough_until:
            self.note_dave_passthrough_window(120.0)

    @property
    def _secret_key(self) -> bytes:
        return self._creds[0]

    @property
    def _dave_session(self):
        return self._creds[1]

    @property
    def _dave_protocol_version(self) -> int:
        return self._creds[2]

    @property
    def _dave_downgraded(self) -> bool:
        return self._creds[3]

    def note_dave_passthrough_window(self, seconds: float) -> None:
        """Open a plaintext-passthrough grace window for unmapped SSRCs.

        Mirrors discord.py's own ``set_passthrough_mode`` calls: a pending
        downgrade transition allows plaintext for up to 120s, and a
        confirmed upgrade-after-downgrade allows a 10s grace while senders
        catch up.  Replace-semantics, matching upstream: each grant resets
        the deadline, so an upgrade's short grace supersedes a residual
        downgrade window instead of being swallowed by it.
        """
        self._dave_passthrough_until = _adapter_clock().monotonic() + seconds

    def refresh_credentials(self, reason: str) -> None:
        """Re-resolve decryption state from the live connection.

        Cheap and idempotent (attribute reads plus one small copy), so
        callers invoke it on every trigger — session description, DAVE
        epoch prepare, a membership change in the channel, or a decrypt
        failure streak — without needing to coalesce.
        """
        if not self._running:
            return
        try:
            conn = self._vc._connection
            self._resolve_credentials(conn)
            self._nacl_fail_streak = 0
            logger.info(
                "VoiceReceiver credentials refreshed (%s; bot_ssrc=%d, dave=%s)",
                reason,
                self._bot_ssrc,
                "on" if self._dave_session else "off",
            )
        except Exception as e:
            logger.warning(
                "VoiceReceiver credential refresh failed (%s): %s", reason, e
            )

    def stop(self):
        """Stop listening and clean up."""
        with self._lock:
            self._capture_generation += 1
            self._running = False
        try:
            self._vc._connection.remove_socket_listener(self._on_packet)
        except Exception:
            pass
        self._restore_speaking_hook()
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()
            self._stream_chunks.clear()
            self._stream_chunk_bytes = 0
            self._decoders.clear()
            self._ssrc_to_user.clear()
            self._ssrc_generation.clear()
        logger.info(
            "VoiceReceiver stopped (frames ok=%d, decrypt_failed=%d, "
            "dave_unmapped_dropped=%d)",
            self._decode_ok,
            self._decode_failed,
            self._dave_unmapped_dropped,
        )

    def pause(self):
        with self._lock:
            self._paused = True
            self.discard_pending()

    def resume(self):
        with self._lock:
            self._capture_generation += 1
            self._paused = False

    def discard_pending(self) -> None:
        """Discard local utterance state without stopping the receiver."""
        with self._lock:
            self._capture_generation += 1
            self._buffers.clear()
            self._last_packet_time.clear()
            self._stream_chunks.clear()
            self._stream_chunk_bytes = 0
            # Capture invalidation also abandons unmapped in-flight decoders.
            for ssrc in list(self._decoders):
                if not self._ssrc_to_user.get(ssrc):
                    self._decoders.pop(ssrc, None)

    def invalidate_channel(self) -> tuple:
        """Fence a move, retaining only the old channel's RTP identity for rollback."""
        with self._lock:
            prior = (dict(self._ssrc_to_user), dict(self._ssrc_generation),
                     dict(self._decoders), self.require_explicit_mapping)
            self._channel_move_pending = True
            self.discard_pending()
            self._ssrc_to_user.clear()
            self._ssrc_generation.clear()
            self._decoders.clear()
            self.require_explicit_mapping = True
            return prior

    def restore_channel_move(self, prior: tuple) -> None:
        """Resume the old channel without reviving samples from either side of the move."""
        with self._lock:
            self.discard_pending()
            users, generations, decoders, explicit_mapping = prior
            self._ssrc_to_user = users
            self._ssrc_generation = defaultdict(int, generations)
            self._decoders = decoders
            self.require_explicit_mapping = explicit_mapping
            self._channel_move_pending = False

    def complete_channel_move(self) -> None:
        """Resume packet and SPEAKING ingress only after Discord finishes moving."""
        with self._lock:
            self._channel_move_pending = False

    # ------------------------------------------------------------------
    # SSRC -> user_id mapping via SPEAKING opcode hook
    # ------------------------------------------------------------------

    def map_ssrc(self, ssrc: int, user_id: int):
        with self._lock:
            if self._channel_move_pending:
                return
            previous_user = self._ssrc_to_user.get(ssrc, 0)
            if previous_user == user_id:
                return

            self._ssrc_generation[ssrc] += 1
            generation = self._ssrc_generation[ssrc]
            self._stream_chunks = [
                chunk for chunk in self._stream_chunks if chunk[1] != ssrc
            ]
            self._stream_chunk_bytes = sum(
                len(chunk[4]) for chunk in self._stream_chunks
            )

            if previous_user:
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
                self._decoders.pop(ssrc, None)

            self._ssrc_to_user[ssrc] = user_id
            if not previous_user:
                preroll = bytes(self._buffers.get(ssrc, b""))
                if (
                    preroll
                    and self._stream_chunk_bytes + len(preroll)
                    <= self.MAX_STREAM_QUEUE_BYTES
                ):
                    self._stream_chunks.append(
                        (
                            self._capture_generation,
                            ssrc,
                            generation,
                            user_id,
                            preroll,
                        )
                    )
                    self._stream_chunk_bytes += len(preroll)

    def unmap_user(self, user_id: int) -> None:
        with self._lock:
            stale_ssrcs = [
                ssrc
                for ssrc, mapped_user in self._ssrc_to_user.items()
                if mapped_user == user_id
            ]
            if not stale_ssrcs:
                return
            stale = set(stale_ssrcs)
            for ssrc in stale_ssrcs:
                self._ssrc_generation[ssrc] += 1
                self._ssrc_to_user.pop(ssrc, None)
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
                self._decoders.pop(ssrc, None)
            self._stream_chunks = [
                chunk for chunk in self._stream_chunks if chunk[1] not in stale
            ]
            self._stream_chunk_bytes = sum(
                len(chunk[4]) for chunk in self._stream_chunks
            )

    def _install_speaking_hook(self, conn):
        """Wrap the voice websocket hook to capture SPEAKING events (op 5).

        VoiceConnectionState stores the hook as ``conn.hook`` (public attr).
        It is passed to DiscordVoiceWebSocket on each (re)connect, so we
        must wrap it on the VoiceConnectionState level AND on the current
        live websocket instance.
        """
        original_hook = conn.hook
        receiver_self = self

        async def wrapped_hook(ws, msg):
            if isinstance(msg, dict) and msg.get("op") == 5:
                data = msg.get("d", {})
                ssrc = data.get("ssrc")
                user_id = data.get("user_id")
                with receiver_self._lock:
                    receiver_active = receiver_self._running
                if ssrc and user_id and receiver_active:
                    logger.info("SPEAKING event: ssrc=%d -> user=%s", ssrc, user_id)
                    receiver_self.map_ssrc(int(ssrc), int(user_id))
            if original_hook:
                await original_hook(ws, msg)
            # Re-resolve decryption state after the events that change it.
            # DiscordVoiceWebSocket.received_message dispatches the op (which
            # is what updates conn.secret_key / conn.dave_session /
            # conn.dave_protocol_version) BEFORE calling this hook, so the
            # refresh reads the new values:
            #   op 4  SESSION_DESCRIPTION — new transport key, and
            #         reinit_dave_session() may replace conn.dave_session
            #   op 24 DAVE_PREPARE_EPOCH  — epoch 1 recreates the MLS group
            #   op 21 DAVE_PREPARE_TRANSITION with protocol_version 0 —
            #         discord.py opens a 120s plaintext passthrough window
            #         (set_passthrough_mode(True, 120)); mirror it so the
            #         unmapped-SSRC gate doesn't drop legitimate plaintext
            #   op 22 DAVE_EXECUTE_TRANSITION — the protocol version just
            #         changed; refresh, plus a 10s grace mirroring the
            #         upgrade path's set_passthrough_mode(True, 10)
            if isinstance(msg, dict):
                op = msg.get("op")
                if op in (4, 24):
                    receiver_self.refresh_credentials(f"voice ws op {op}")
                elif op == 21:
                    if (msg.get("d") or {}).get("protocol_version") == 0:
                        # Pending downgrade: upstream enables plaintext
                        # passthrough for up to 120s (set_passthrough_mode).
                        receiver_self.note_dave_passthrough_window(120.0)
                elif op == 22:
                    # A transition may have executed; refresh and detect the
                    # upgrade-after-downgrade EDGE (dave_downgraded flipping
                    # True -> False) — the only case upstream grants the 10s
                    # passthrough grace.  Same-version transitions and
                    # unknown/duplicate ids change no state upstream, so no
                    # edge fires and no window opens.
                    was_downgraded = receiver_self._dave_downgraded
                    receiver_self.refresh_credentials("voice ws op 22")
                    if (
                        was_downgraded
                        and not receiver_self._dave_downgraded
                        and receiver_self._dave_protocol_version > 0
                    ):
                        receiver_self.note_dave_passthrough_window(10.0)

        # Set on connection state (for future reconnects)
        self._hook_connection = conn
        self._original_connection_hook = original_hook
        self._installed_speaking_hook = wrapped_hook
        conn.hook = wrapped_hook
        # Set on the current live websocket (for immediate effect)
        try:
            from discord.utils import MISSING
            if hasattr(conn, 'ws') and conn.ws is not MISSING:
                self._original_ws_hook = getattr(conn.ws, "_hook", None)
                conn.ws._hook = wrapped_hook
                logger.info("Speaking hook installed on live websocket")
        except Exception as e:
            logger.warning("Could not install hook on live ws: %s", e)

    def _restore_speaking_hook(self) -> None:
        conn = self._hook_connection
        installed = self._installed_speaking_hook
        if conn is None or installed is None:
            return
        try:
            if getattr(conn, "hook", None) is installed:
                conn.hook = self._original_connection_hook
            ws = getattr(conn, "ws", None)
            if ws is not None and getattr(ws, "_hook", None) is installed:
                ws._hook = self._original_ws_hook
        except Exception:  # health: allow BLE001 -- discord.py connection/websocket hooks may be plugin-owned; teardown must continue
            logger.debug("Could not restore speaking hook", exc_info=True)
        finally:
            self._hook_connection = None
            self._installed_speaking_hook = None
            self._original_connection_hook = None
            self._original_ws_hook = None

    # ------------------------------------------------------------------
    # Packet handler (called from SocketReader thread)
    # ------------------------------------------------------------------

    def _snapshot_packet_generation(
        self,
        ssrc: int,
    ) -> tuple[int, int, int]:
        with self._lock:
            return (
                self._capture_generation,
                self._ssrc_generation.get(ssrc, 0),
                int(self._ssrc_to_user.get(ssrc, 0)),
            )

    def _packet_generation_is_current(
        self,
        ssrc: int,
        capture_generation: int,
        ssrc_generation: int,
        user_id: int,
    ) -> bool:
        return (
            self._running
            and not self._paused
            and not self._channel_move_pending
            and self._capture_generation == capture_generation
            and self._ssrc_generation.get(ssrc, 0) == ssrc_generation
            and int(self._ssrc_to_user.get(ssrc, 0)) == user_id
        )

    def _strip_rtp_padding(self, payload: bytes, ssrc: int) -> bytes:
        if not payload:
            if self._packet_debug_count <= 10:
                logger.warning("RTP padding bit set but no payload (ssrc=%d)", ssrc)
            return b""
        pad_len = payload[-1]
        if pad_len == 0 or pad_len > len(payload):
            if self._packet_debug_count <= 10:
                logger.warning(
                    "Invalid RTP padding length %d for payload size %d (ssrc=%d)",
                    pad_len, len(payload), ssrc,
                )
            return b""
        return payload[:-pad_len]

    def _decoder_for_ssrc_locked(self, ssrc: int):
        decoder = self._decoders.get(ssrc)
        if decoder is not None:
            return decoder
        if len(self._decoders) >= self.MAX_ACTIVE_SSRC_BUFFERS:
            idle = next((candidate for candidate in self._decoders if candidate not in self._buffers), None)
            if idle is None:
                return None
            self._decoders.pop(idle, None)
        if __package__:
            from . import adapter as adapter_module
        else:
            import adapter as adapter_module
        decoder = adapter_module.discord.opus.Decoder()
        self._decoders[ssrc] = decoder
        return decoder

    def _on_packet(self, data: bytes):
        if not self._running or self._paused or self._channel_move_pending:
            return

        # One consistent credential generation for this packet — a refresh
        # on another thread swaps the whole tuple, never half of it.
        secret_key, dave_session, dave_pver, _dave_downgraded = self._creds

        # Log first few raw packets for debugging
        self._packet_debug_count += 1
        if self._packet_debug_count <= 5:
            logger.debug(
                "Raw UDP packet: len=%d, first_bytes=%s",
                len(data), data[:4].hex() if len(data) >= 4 else "short",
            )

        if len(data) < 16:
            return

        # RTP version check: top 2 bits must be 10 (version 2).
        # Lower bits may vary (padding, extension, CSRC count).
        # Payload type (byte 1 lower 7 bits) = 0x78 (120) for voice.
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            if self._packet_debug_count <= 5:
                logger.debug("Skipped non-RTP: byte0=0x%02x byte1=0x%02x", data[0], data[1])
            return

        first_byte = data[0]
        _, _, seq, timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)

        # Skip bot's own audio
        if ssrc == self._bot_ssrc:
            return
        packet_generation = self._snapshot_packet_generation(ssrc)

        # Calculate dynamic RTP header size (RFC 9335 / rtpsize mode)
        cc = first_byte & 0x0F  # CSRC count
        has_extension = bool(first_byte & 0x10)  # extension bit
        has_padding = bool(first_byte & 0x20)  # padding bit (RFC 3550 §5.1)
        header_size = 12 + (4 * cc) + (4 if has_extension else 0)

        if len(data) < header_size + 4:  # need at least header + nonce
            return

        # Read extension length from preamble (for skipping after decrypt)
        ext_data_len = 0
        if has_extension:
            ext_preamble_offset = 12 + (4 * cc)
            ext_words = struct.unpack_from(">H", data, ext_preamble_offset + 2)[0]
            ext_data_len = ext_words * 4

        if self._packet_debug_count <= 10:
            with self._lock:
                known_user = self._ssrc_to_user.get(ssrc, "unknown")
            logger.debug(
                "RTP packet: ssrc=%d, seq=%d, user=%s, hdr=%d, ext_data=%d",
                ssrc, seq, known_user, header_size, ext_data_len,
            )

        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]

        # --- NaCl transport decrypt (aead_xchacha20_poly1305_rtpsize) ---
        if len(payload_with_nonce) < 4:
            return
        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])

        try:
            import nacl.secret
            box = nacl.secret.Aead(secret_key)
            decrypted = box.decrypt(encrypted, header, bytes(nonce))
            self._nacl_fail_streak = 0
        except Exception as e:
            self._decode_failed += 1
            self._nacl_fail_streak += 1
            # Never go fully dark: after the first 10 warnings, keep emitting
            # one every 250 failures so a deaf session stays diagnosable.
            if self._packet_debug_count <= 10 or self._nacl_fail_streak % 250 == 0:
                logger.warning(
                    "NaCl decrypt failed: %s (hdr=%d, enc=%d, streak=%d)",
                    e, header_size, len(encrypted), self._nacl_fail_streak,
                )
            # A sustained failure streak means the transport key rotated
            # under us (voice reconnect / re-key) — re-read it from the live
            # connection instead of staying deaf on a stale copy.  The
            # refresh resets the streak, so this retries every
            # REKEY_FAILURE_STREAK packets while the failure persists.
            if self._nacl_fail_streak >= self.REKEY_FAILURE_STREAK:
                self.refresh_credentials("decrypt-failure streak")
            return

        # Skip encrypted extension data to get the actual opus payload
        if ext_data_len and len(decrypted) > ext_data_len:
            decrypted = decrypted[ext_data_len:]

        # --- Strip RTP padding (RFC 3550 §5.1) ---
        # When the P bit is set, the last payload byte holds the count of
        # trailing padding bytes (including itself) that must be removed
        # before further processing. Skipping this passes padding-contaminated
        # bytes into DAVE/Opus and corrupts inbound audio.
        if has_padding:
            decrypted = self._strip_rtp_padding(decrypted, ssrc)
            if not decrypted:
                return

        # --- DAVE E2EE decrypt ---
        if dave_session:
            user_id = packet_generation[2]
            if not user_id and not self.require_explicit_mapping:
                # A sole-member rejoin can lack SPEAKING; optional modes demand
                # explicit SSRC identity and never infer an owner.
                user_id = self._infer_user_for_ssrc(ssrc)
                if user_id:
                    packet_generation = self._snapshot_packet_generation(ssrc)
            if user_id:
                try:
                    import davey
                    decrypted = dave_session.decrypt(
                        user_id, davey.MediaType.audio, decrypted
                    )
                except Exception as e:
                    # Unencrypted passthrough — use NaCl-decrypted data as-is
                    if "Unencrypted" not in str(e):
                        if self._packet_debug_count <= 10:
                            logger.warning("DAVE decrypt failed for ssrc=%d: %s", ssrc, e)
                        return
            elif (
                dave_pver > 0
                and _adapter_clock().monotonic() >= self._dave_passthrough_until
            ):
                # E2EE is actively on (protocol > 0, no passthrough window),
                # so an unmapped SSRC's payload is still ciphertext.  Opus
                # will happily "decode" it (producing shredded audio and
                # poisoning decoder state), so drop the frame until a
                # SPEAKING event maps the SSRC — bounded loss beats corrupt
                # audio.  A non-null session alone is NOT this predicate: the
                # session object survives protocol downgrades to 0 and
                # passthrough transitions, where plaintext is legitimate and
                # must fall through to opus below.
                self._dave_unmapped_dropped += 1
                if (
                    self._packet_debug_count <= 10
                    or self._dave_unmapped_dropped % 250 == 1
                ):
                    logger.debug(
                        "Dropping DAVE frame for unmapped ssrc=%d (dropped=%d)",
                        ssrc, self._dave_unmapped_dropped,
                    )
                return

        # --- Opus decode -> PCM ---
        try:
            with self._lock:
                if not self._packet_generation_is_current(
                    ssrc,
                    *packet_generation,
                ):
                    return
                decoder = self._decoder_for_ssrc_locked(ssrc)
                if decoder is None:
                    return
            pcm = decoder.decode(decrypted)
            self._buffer_decoded_pcm(ssrc, pcm, *packet_generation)
            self._decode_ok += 1
        except Exception as e:
            self._decode_failed += 1
            with self._lock:
                self._decoders.pop(ssrc, None)
            logger.debug(
                "Opus decode error for SSRC %s; reset decoder: %s",
                ssrc,
                e,
            )
            return

    def _buffer_decoded_pcm(
        self,
        ssrc: int,
        pcm: bytes,
        capture_generation: Optional[int] = None,
        ssrc_generation: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Buffer PCM locally and queue only explicitly mapped frames."""
        if not pcm:
            return
        with self._lock:
            if capture_generation is None:
                capture_generation = self._capture_generation
            if ssrc_generation is None:
                ssrc_generation = self._ssrc_generation.get(ssrc, 0)
            if user_id is None:
                user_id = int(self._ssrc_to_user.get(ssrc, 0))
            if not self._packet_generation_is_current(
                ssrc,
                capture_generation,
                ssrc_generation,
                user_id,
            ):
                return
            if (
                ssrc not in self._buffers
                and len(self._buffers) >= self.MAX_ACTIVE_SSRC_BUFFERS
            ):
                return
            buf = self._buffers[ssrc]
            remaining = self.MAX_UTTERANCE_BYTES - len(buf)
            accepted = bytes(pcm[:remaining]) if remaining > 0 else b""
            if accepted:
                buf.extend(accepted)
            self._last_packet_time[ssrc] = _adapter_clock().monotonic()
            if (
                user_id
                and accepted
                and self._stream_chunk_bytes + len(accepted)
                <= self.MAX_STREAM_QUEUE_BYTES
            ):
                self._stream_chunks.append(
                    (
                        capture_generation,
                        ssrc,
                        ssrc_generation,
                        int(user_id),
                        accepted,
                    )
                )
                self._stream_chunk_bytes += len(accepted)

    def snapshot_sources(self) -> dict:
        """Bind utterances to current capture and SSRC generations before extraction."""
        with self._lock:
            sources = defaultdict(list)
            for ssrc, user_id in self._ssrc_to_user.items():
                if user_id:
                    sources[user_id].append((self._capture_generation, ssrc,
                                             self._ssrc_generation.get(ssrc, 0), user_id))
            return {user_id: tuple(tokens) for user_id, tokens in sources.items()}

    def drain_stream_chunks(self) -> list[tuple[int, int, int, int, bytes]]:
        """Return current-generation explicitly mapped chunks exactly once."""
        with self._lock:
            queued = self._stream_chunks
            self._stream_chunks = []
            self._stream_chunk_bytes = 0
            chunks = [
                (capture, ssrc, generation, user_id, pcm)
                for capture, ssrc, generation, user_id, pcm in queued
                if self._capture_generation == capture
                and self._ssrc_to_user.get(ssrc) == user_id
                and self._ssrc_generation.get(ssrc) == generation
            ]
        return chunks

    def stream_chunk_is_current(
        self,
        capture_generation: int,
        ssrc: int,
        generation: int,
        user_id: int,
    ) -> bool:
        with self._lock:
            return (
                self._running
                and not self._paused
                and not self._channel_move_pending
                and self._capture_generation == capture_generation
                and self._ssrc_to_user.get(ssrc) == user_id
                and self._ssrc_generation.get(ssrc) == generation
            )

    # ------------------------------------------------------------------
    # Silence detection
    # ------------------------------------------------------------------

    def _infer_user_for_ssrc(self, ssrc: int) -> int:
        """Infer user_id for an unmapped SSRC: after a bot rejoin Discord may not resend
        SPEAKING, so if exactly one allowed user is in the channel, map the SSRC to them."""
        try:
            channel = self._vc.channel
            if not channel:
                return 0
            bot_id = self._vc.user.id if self._vc.user else 0
            allowed = self._allowed_user_ids
            candidates = [
                m.id for m in channel.members
                if m.id != bot_id and (not allowed or str(m.id) in allowed)
            ]
            if len(candidates) == 1:
                uid = candidates[0]
                self.map_ssrc(ssrc, uid)
                logger.info("Auto-mapped ssrc=%d -> user=%d (sole allowed member)", ssrc, uid)
                return uid
        except Exception:
            pass
        return 0

    def check_silence(self) -> list:
        """Return list of (user_id, pcm_bytes) for completed utterances."""
        now = _adapter_clock().monotonic()
        completed = []

        with self._lock:
            ssrc_user_map = dict(self._ssrc_to_user)
            ssrc_list = list(self._buffers.keys())

            for ssrc in ssrc_list:
                last_time = self._last_packet_time.get(ssrc, now)
                silence_duration = now - last_time
                buf = self._buffers[ssrc]
                # 48kHz, 16-bit, stereo = 192000 bytes/sec
                buf_duration = len(buf) / (self.SAMPLE_RATE * self.CHANNELS * 2)

                turn_complete = (
                    silence_duration >= self.SILENCE_THRESHOLD
                    or len(buf) >= self.MAX_UTTERANCE_BYTES
                )
                if turn_complete and buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id and not self.require_explicit_mapping:
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)
                elif silence_duration >= self.SILENCE_THRESHOLD * 2:
                    # Stale buffer with no valid user — discard
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)
                if ssrc not in self._buffers and not self._ssrc_to_user.get(ssrc):
                    self._decoders.pop(ssrc, None)

        return completed

    def flush_pending(self) -> list:
        """Return buffered utterances that have not yet reached silence."""
        completed = []

        with self._lock:
            ssrc_user_map = dict(self._ssrc_to_user)
            for ssrc, buf in list(self._buffers.items()):
                # 48kHz, 16-bit, stereo = 192000 bytes/sec
                buf_duration = len(buf) / (self.SAMPLE_RATE * self.CHANNELS * 2)
                if buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id and not self.require_explicit_mapping:
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
                if not self._ssrc_to_user.get(ssrc):
                    self._decoders.pop(ssrc, None)

        return completed

    # ------------------------------------------------------------------
    # PCM -> WAV conversion (for Whisper STT)
    # ------------------------------------------------------------------

    @staticmethod
    def pcm_to_wav(pcm_data: bytes, output_path: str, src_rate: int = 48000, src_channels: int = 2):
        """Convert raw PCM to 16kHz mono WAV via ffmpeg into *output_path* (not stdout: ffmpeg
        can't seek a pipe, so piped WAV carries placeholder RIFF sizes strict readers misreport)."""
        from hermes_cli._subprocess_compat import windows_hide_flags
        if __package__:
            from . import adapter as adapter_module
        else:
            import adapter as adapter_module
        subprocess.run(
            [
                adapter_module.resolve_ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "s16le",
                "-ar", str(src_rate), "-ac", str(src_channels), "-i", "pipe:0", "-ar", "16000",
                "-ac", "1", output_path,
            ],
            input=pcm_data,
            check=True,
            timeout=10,
            # Capture stderr so a failure's CalledProcessError carries ffmpeg's real message.
            stderr=subprocess.PIPE,
            creationflags=windows_hide_flags(),
        )
