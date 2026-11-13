from __future__ import annotations

"""Discord voice capture and decoding."""

import logging
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path as _Path

try:
    import discord
except ImportError:
    discord = None

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

try:
    from .ffmpeg_utils import resolve_ffmpeg_executable
except ImportError:
    from ffmpeg_utils import resolve_ffmpeg_executable

logger = logging.getLogger(__name__)


class VoiceReceiver:
    """Captures voice audio from a Discord voice channel: hooks the VoiceClient socket, decrypts
    RTP (NaCl + DAVE E2EE), decodes Opus per user; a polling loop delivers utterances on silence."""

    SILENCE_THRESHOLD = 1.5    # seconds of silence → end of utterance
    MIN_SPEECH_DURATION = 0.5  # minimum seconds to process (skip noise)
    SAMPLE_RATE = 48000        # Discord native rate
    CHANNELS = 2               # Discord sends stereo
    REKEY_FAILURE_STREAK = 25  # consecutive NaCl failures → re-resolve creds

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
        self._ssrc_to_user: dict[int, int] = {}
        self._lock = threading.Lock()
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: dict[int, float] = {}
        # Opus decoder per SSRC (each user needs own decoder state)
        self._decoders: dict[int, object] = {}
        # Pause flag: don't capture while bot is playing TTS
        self._paused = False
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
        if 0 in pending.values() and time.monotonic() >= self._dave_passthrough_until:
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
        self._dave_passthrough_until = time.monotonic() + seconds

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
        self._running = False
        try:
            self._vc._connection.remove_socket_listener(self._on_packet)
        except Exception:
            pass
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()
            self._decoders.clear()
            self._ssrc_to_user.clear()
        logger.info(
            "VoiceReceiver stopped (frames ok=%d, decrypt_failed=%d, "
            "dave_unmapped_dropped=%d)",
            self._decode_ok,
            self._decode_failed,
            self._dave_unmapped_dropped,
        )
    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # --- SSRC -> user_id mapping via SPEAKING opcode hook ---

    def map_ssrc(self, ssrc: int, user_id: int):
        with self._lock:
            self._ssrc_to_user[ssrc] = user_id

    def _install_speaking_hook(self, conn):
        """Wrap the voice websocket hook to capture SPEAKING events (op 5); ``conn.hook`` is
        re-passed on each (re)connect, so wrap it on the state AND the live websocket."""
        original_hook = conn.hook
        receiver_self = self

        async def wrapped_hook(ws, msg):
            if isinstance(msg, dict) and msg.get("op") == 5:
                data = msg.get("d", {})
                ssrc = data.get("ssrc")
                user_id = data.get("user_id")
                if ssrc and user_id:
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
        conn.hook = wrapped_hook
        try:
            from discord.utils import MISSING
            if hasattr(conn, 'ws') and conn.ws is not MISSING:
                conn.ws._hook = wrapped_hook
                logger.info("Speaking hook installed on live websocket")
        except Exception as e:
            logger.warning("Could not install hook on live ws: %s", e)

    # --- Packet handler (called from SocketReader thread) ---

    def _on_packet(self, data: bytes):
        if not self._running or self._paused:
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
        # RTP v2: top 2 bits 10 (rest varies); voice payload type (byte 1 & 0x7F) is 0x78.
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            if self._packet_debug_count <= 5:
                logger.debug("Skipped non-RTP: byte0=0x%02x byte1=0x%02x", data[0], data[1])
            return
        first_byte = data[0]
        _, _, seq, _timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)
        if ssrc == self._bot_ssrc:
            return
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
        # Strip RTP padding (RFC 3550 §5.1): last payload byte is the count; leaving it corrupts DAVE/Opus.
        if has_padding:
            if not decrypted:
                if self._packet_debug_count <= 10:
                    logger.warning("RTP padding bit set but no payload (ssrc=%d)", ssrc)
                return
            pad_len = decrypted[-1]
            if pad_len == 0 or pad_len > len(decrypted):
                if self._packet_debug_count <= 10:
                    logger.warning(
                        "Invalid RTP padding length %d for payload size %d (ssrc=%d)",
                        pad_len, len(decrypted), ssrc,
                    )
                return
            decrypted = decrypted[:-pad_len]
            if not decrypted:
                return
        # --- DAVE E2EE decrypt ---
        if dave_session:
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0)
                if not user_id:
                    # Rejoin race: SPEAKING may never be resent for a user who
                    # was already talking — try the sole-member inference
                    # before giving up on this frame.
                    user_id = self._infer_user_for_ssrc(ssrc)
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
                and time.monotonic() >= self._dave_passthrough_until
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
            if ssrc not in self._decoders:
                self._decoders[ssrc] = discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            self._decode_ok += 1
            with self._lock:
                self._buffers[ssrc].extend(pcm)
                self._last_packet_time[ssrc] = time.monotonic()
        except Exception as e:
            with self._lock:
                self._decoders.pop(ssrc, None)
            logger.debug("Opus decode error for SSRC %s; reset decoder: %s", ssrc, e)
            return

    # --- Silence detection ---

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
                self._ssrc_to_user[ssrc] = uid
                logger.info("Auto-mapped ssrc=%d -> user=%d (sole allowed member)", ssrc, uid)
                return uid
        except Exception:
            pass
        return 0

    def check_silence(self) -> list:
        """Return list of (user_id, pcm_bytes) for completed utterances."""
        now = time.monotonic()
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
                if silence_duration >= self.SILENCE_THRESHOLD and buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id:
                        # SSRC unmapped (SPEAKING missing after rejoin) — infer from channel.
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                    self._buffers[ssrc] = bytearray()
                    self._last_packet_time.pop(ssrc, None)
                elif silence_duration >= self.SILENCE_THRESHOLD * 2:
                    # Stale buffer with no valid user — discard
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)
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
                    if not user_id:
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
        return completed

    def discard_pending(self) -> None:
        """Drop buffered PCM that no poll or flush has emitted yet."""
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()

    # --- PCM -> WAV conversion (for Whisper STT) ---

    @staticmethod
    def pcm_to_wav(pcm_data: bytes, output_path: str, src_rate: int = 48000, src_channels: int = 2):
        """Convert raw PCM to 16kHz mono WAV via ffmpeg into *output_path* (not stdout: ffmpeg
        can't seek a pipe, so piped WAV carries placeholder RIFF sizes strict readers misreport)."""
        from hermes_cli._subprocess_compat import windows_hide_flags
        subprocess.run(
            [
                resolve_ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "s16le",
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
