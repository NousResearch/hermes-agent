"""Discord voice methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordVoiceMixin:
    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Play auto-TTS audio: in the guild's VC if joined, else as a file attachment."""
        from . import adapter as _adapter

        for gid, text_ch_id in self._voice_text_channels.items():
            if str(text_ch_id) == str(chat_id) and self.is_in_voice_channel(gid):
                _adapter.logger.info("[%s] Playing TTS in voice channel (guild=%d)", self.name, gid)
                success = await self.play_in_voice_channel(gid, audio_path)
                return _adapter.SendResult(success=success)
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send audio as a Discord file attachment."""
        from . import adapter as _adapter

        try:
            import io
            channel = await self._resolve_channel(chat_id)
            if not channel:
                return _adapter.SendResult(success=False, error=f"Channel {chat_id} not found")
            if not _adapter.os.path.exists(audio_path):
                return _adapter.SendResult(success=False, error=f"Audio file not found: {audio_path}")
            filename = _adapter.os.path.basename(audio_path)
            reference = self._reply_reference_for_send(reply_to, channel)
            with open(audio_path, "rb") as f:
                file_data = f.read()
            # Forum channels reject POST /messages (native voice path too); create a thread post instead.
            if self._is_forum_parent(channel):
                forum_file = _adapter.discord.File(io.BytesIO(file_data), filename=filename)
                return await self._forum_post_file(
                    channel, content=(caption or "").strip(), file=forum_file,
                )
            # Try sending as a native voice message via raw API (flags=8192).
            try:
                import base64
                try:
                    from mutagen.oggopus import OggOpus
                    duration_secs = OggOpus(audio_path).info.length
                except Exception:
                    duration_secs = max(1.0, len(file_data) / 2000.0)
                payload_data = {
                    "flags": 8192,
                    "attachments": [{
                        "id": "0", "filename": "voice-message.ogg", "duration_secs": round(duration_secs, 2),
                        "waveform": base64.b64encode(bytes([128] * 256)).decode(),
                    }],
                }
                if reference is not None:
                    payload_data["message_reference"] = {"message_id": str(reply_to), "fail_if_not_exists": False}
                form = [
                    {"name": "payload_json", "value": _adapter.json.dumps(payload_data)},
                    {
                        "name": "files[0]", "value": file_data, "filename": "voice-message.ogg",
                        "content_type": "audio/ogg",
                    },
                ]
                msg_data = await self._client.http.request(
                    _adapter.discord.http.Route("POST", "/channels/{channel_id}/messages", channel_id=channel.id),
                    form=form,
                )
                return _adapter.SendResult(success=True, message_id=str(msg_data["id"]))
            except Exception as voice_err:
                _adapter.logger.debug("Voice message flag failed, falling back to file: %s", voice_err)
                file = _adapter.discord.File(io.BytesIO(file_data), filename=filename)
                try:
                    msg = await channel.send(file=file, reference=reference)
                except Exception as send_err:
                    if reference is not None and self._is_reply_reference_rejected(send_err):
                        msg = await channel.send(file=file, reference=None)
                    else:
                        raise
                return _adapter.SendResult(success=True, message_id=str(msg.id))
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to send audio, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)

    def _load_voice_fx_config(self) -> Dict[str, Any]:
        """Read ``discord.voice_fx`` from config.yaml (not .env; off by default) with safe defaults."""
        from . import adapter as _adapter

        defaults: _adapter.Dict[str, _adapter.Any] = {
            "enabled": False,        # master switch for the mixer subsystem
            "ambient_enabled": True, # idle "thinking" bed while tools run
            "ambient_path": "",      # optional custom loop file; "" = synthesised
            "ambient_gain": 0.18,    # idle bed loudness (0..1)
            "duck_gain": 0.06,       # ambient loudness while speech plays
            "speech_gain": 1.0,      # TTS / ack loudness
            "lead_silence_ms": 200,  # silence prepended to each clip so the
                                     # voice socket's warm-up doesn't clip the first word
            "ack_enabled": True,     # speak a short phrase before tool calls
            "ack_phrases": [
                "Let me look into that.", "One moment.", "Checking on that now.", "Give me a sec.",
                "On it.",
            ],
        }
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
            fx = ((cfg.get("discord") or {}).get("voice_fx") or {})
            if isinstance(fx, dict):
                for k, v in fx.items():
                    if k in defaults and v is not None:
                        defaults[k] = v
        except Exception as e:
            _adapter.logger.debug("Could not load discord.voice_fx config: %s", e)
        return defaults

    def _load_discord_int_config(self, key: str, default: int, *, minimum: int = 0) -> int:
        """Read a non-secret integer from the top-level ``discord`` config."""
        from . import adapter as _adapter

        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
            raw = (cfg.get("discord") or {}).get(key, default)
            value = int(raw)
            return max(minimum, value)
        except Exception as e:
            _adapter.logger.debug("Could not load discord.%s config: %s", key, e)
            return default

    def _load_voice_timeout(self) -> int:
        """Return voice-channel inactivity timeout seconds; 0 disables it."""
        return self._load_discord_int_config(
            "voice_channel_inactivity_timeout_seconds", self.VOICE_TIMEOUT, minimum=0,
        )

    def _load_playback_timeout(self) -> int:
        """Return minimum playback wait seconds for Discord VC audio."""
        return self._load_discord_int_config(
            "voice_playback_timeout_seconds", self.PLAYBACK_TIMEOUT, minimum=1,
        )

    def _voice_timeout_limit(self) -> int:
        from . import adapter as _adapter

        return int(getattr(self, "_voice_timeout_seconds", self.VOICE_TIMEOUT))

    def _playback_timeout_limit(self) -> int:
        from . import adapter as _adapter

        return int(getattr(self, "_playback_timeout_seconds", self.PLAYBACK_TIMEOUT))

    def _probe_audio_duration_seconds(self, audio_path: str) -> Optional[float]:
        """Best-effort audio duration probe used to size playback timeouts."""
        from . import adapter as _adapter

        try:
            import importlib
            mutagen = importlib.import_module("mutagen")
            audio = mutagen.File(audio_path)
            length = getattr(getattr(audio, "info", None), "length", None)
            if length:
                return float(length)
        except Exception:
            pass
        try:
            proc = _adapter.subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                stdin=_adapter.subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                raw = (proc.stdout or "").strip()
                if raw:
                    return float(raw)
        except Exception:
            pass
        return None

    async def _playback_timeout_for_audio(self, audio_path: str) -> float:
        """Return timeout for this clip: configured floor or duration+padding."""
        from . import adapter as _adapter

        floor = float(self._playback_timeout_limit())
        duration = await _adapter.asyncio.to_thread(self._probe_audio_duration_seconds, audio_path)
        if not duration or duration <= 0:
            return floor
        return max(floor, duration + float(self.PLAYBACK_TIMEOUT_PADDING))

    def _get_ambient_pcm(self) -> Optional[bytes]:
        """Return cached 48k/stereo/s16le PCM for the ambient bed: custom ``ambient_path`` if decodable, else synthesised."""
        from . import adapter as _adapter

        if self._ambient_pcm_cache is not None:
            return self._ambient_pcm_cache
        if not self._voice_fx_cfg.get("ambient_enabled"):
            return None
        vm = _adapter._voice_mixer_module()
        decode_to_pcm, synth_ambient_pcm = vm.decode_to_pcm, vm.synth_ambient_pcm
        pcm: _adapter.Optional[bytes] = None
        path = (self._voice_fx_cfg.get("ambient_path") or "").strip()
        if path and _adapter.os.path.isfile(path):
            pcm = decode_to_pcm(path)
            if not pcm:
                _adapter.logger.warning("Ambient file %s failed to decode; using synth bed", path)
        if not pcm:
            pcm = synth_ambient_pcm()
        self._ambient_pcm_cache = pcm
        return pcm

    async def _install_voice_mixer(self, guild_id: int, vc) -> None:
        """Install a VoiceMixer on the VC; one ``vc.play(mixer)`` runs for the whole connection."""
        from . import adapter as _adapter

        VoiceMixer = _adapter._voice_mixer_module().VoiceMixer
        mixer = VoiceMixer(
            ambient_gain=float(self._voice_fx_cfg.get("ambient_gain", 0.18)),
            duck_gain=float(self._voice_fx_cfg.get("duck_gain", 0.06)),
            speech_gain=float(self._voice_fx_cfg.get("speech_gain", 1.0)),
        )
        ambient = await _adapter.asyncio.to_thread(self._get_ambient_pcm)
        if ambient:
            mixer.set_ambient(ambient)

        def _after(error):
            if error:
                _adapter.logger.error("Voice mixer stream error (guild=%d): %s", guild_id, error)
        if vc.is_playing():
            vc.stop()
        vc.play(mixer, after=_after)
        self._voice_mixers[guild_id] = mixer
        _adapter.logger.info("Voice mixer installed (guild=%d, ambient=%s)", guild_id, bool(ambient))

    def _lead_silence_bytes(self) -> bytes:
        """Silence prepended to speech clips: Discord's voice socket warm-up otherwise clips
        the first ~100-200ms. Returns b"" when ``lead_silence_ms`` <= 0 (opt-out)."""
        from . import adapter as _adapter

        cfg = getattr(self, "_voice_fx_cfg", None) or {}
        try:
            lead_ms = int(cfg.get("lead_silence_ms", 0) or 0)
        except (TypeError, ValueError):
            return b""
        if lead_ms <= 0:
            return b""
        return b"\x00" * (_adapter._voice_mixer_module().BYTES_PER_MS * lead_ms)

    async def play_ack_in_voice(self, guild_id: int, phrase: Optional[str] = None) -> bool:
        """Speak a short ack over the ambient bed (first tool call of a turn); no-op without mixer/acks."""
        from . import adapter as _adapter

        if not self._voice_fx_cfg.get("ack_enabled"):
            return False
        mixer = self._voice_mixers.get(guild_id)
        if mixer is None:
            return False
        if phrase is None:
            import random
            phrases = self._voice_fx_cfg.get("ack_phrases") or ["One moment."]
            phrase = random.choice(phrases)
        import uuid as _uuid
        audio_path = _adapter.os.path.join(
            _adapter.tempfile.gettempdir(), "hermes_voice", f"ack_{_uuid.uuid4().hex[:12]}.mp3",
        )
        _adapter.os.makedirs(_adapter.os.path.dirname(audio_path), exist_ok=True)
        try:
            from tools.tts_tool import text_to_speech_tool
            result_json = await _adapter.asyncio.to_thread(
                text_to_speech_tool, text=phrase, output_path=audio_path
            )
            result = _adapter.json.loads(result_json)
            actual = result.get("file_path", audio_path)
            if not result.get("success") or not _adapter.os.path.isfile(actual):
                return False
            decode_to_pcm = _adapter._voice_mixer_module().decode_to_pcm
            pcm = await _adapter.asyncio.to_thread(decode_to_pcm, actual)
            if not pcm:
                return False
            mixer.play_speech(
                self._lead_silence_bytes() + pcm,
                gain=float(self._voice_fx_cfg.get("speech_gain", 1.0)),
            )
            self._reset_voice_timeout(guild_id)
            return True
        except Exception as e:
            _adapter.logger.debug("play_ack_in_voice failed: %s", e)
            return False
        finally:
            for p in {audio_path, locals().get("actual")}:
                if p and _adapter.os.path.isfile(p):
                    try:
                        _adapter.os.unlink(p)
                    except OSError:
                        pass

    def voice_mixer_active(self, guild_id: int) -> bool:
        """True when a continuous mixer is installed for this guild."""
        mixers = getattr(self, "_voice_mixers", None)
        return bool(mixers) and mixers.get(guild_id) is not None

    async def join_voice_channel(self, channel, *, text_channel_id: int = None, source: dict = None) -> bool:
        """Join a voice channel; returns True on success. ``text_channel_id`` stores the
        transcription-routing binding so programmatic joins work without ``/voice join``."""
        from . import adapter as _adapter

        if not self._client or not _adapter.DISCORD_AVAILABLE:
            return False
        guild_id = channel.guild.id
        async with self._voice_locks.setdefault(guild_id, _adapter.asyncio.Lock()):
            existing = self._voice_clients.get(guild_id)
            if existing and existing.is_connected():
                if existing.channel.id == channel.id:
                    self._reset_voice_timeout(guild_id)
                    return True
                await existing.move_to(channel)
                self._reset_voice_timeout(guild_id)
                return True
            vc = await channel.connect()
            self._voice_clients[guild_id] = vc
            self._reset_voice_timeout(guild_id)
            if text_channel_id is not None:
                self._voice_text_channels[guild_id] = text_channel_id
            if source is not None:
                self._voice_sources[guild_id] = source
            try:
                receiver = _adapter.VoiceReceiver(vc, allowed_user_ids=self._allowed_user_ids)
                receiver.start()
                self._voice_receivers[guild_id] = receiver
                self._voice_listen_tasks[guild_id] = _adapter.asyncio.ensure_future(
                    self._voice_listen_loop(guild_id)
                )
            except Exception as e:
                _adapter.logger.warning("Voice receiver failed to start: %s", e)
            # Mixer is best-effort; failure falls back to one-shot FFmpegPCMAudio playback.
            if getattr(self, "_voice_fx_cfg", {}).get("enabled"):
                try:
                    await self._install_voice_mixer(guild_id, vc)
                except Exception as e:
                    _adapter.logger.warning("Voice mixer failed to start: %s", e)
            return True

    async def leave_voice_channel(self, guild_id: int) -> None:
        """Disconnect from the voice channel in a guild."""
        from . import adapter as _adapter

        async with self._voice_locks.setdefault(guild_id, _adapter.asyncio.Lock()):
            receiver = self._voice_receivers.pop(guild_id, None)
            pending_inputs = []
            if receiver:
                pending_inputs = receiver.flush_pending()
                receiver.stop()
            listen_task = self._voice_listen_tasks.pop(guild_id, None)
            if listen_task:
                listen_task.cancel()
            guild = self._client.get_guild(guild_id) if self._client is not None else None
            for user_id, pcm_data in pending_inputs:
                if self._is_allowed_user(str(user_id), guild=guild, is_dm=False):
                    await self._process_voice_input(guild_id, user_id, pcm_data)
            # Tear down the mixer (stops the continuous outgoing stream).
            if getattr(self, "_voice_mixers", None) is not None:
                self._voice_mixers.pop(guild_id, None)
            vc = self._voice_clients.pop(guild_id, None)
            if vc and vc.is_connected():
                try:
                    if vc.is_playing():
                        vc.stop()
                except Exception:
                    pass
                await vc.disconnect()
            task = self._voice_timeout_tasks.pop(guild_id, None)
            if task:
                task.cancel()
            self._voice_text_channels.pop(guild_id, None)
            self._voice_sources.pop(guild_id, None)

    async def play_in_voice_channel(self, guild_id: int, audio_path: str) -> bool:
        """Play audio in the VC: via the mixer (layered over the ambient bed, ducking it)
        when installed, else the legacy one-shot FFmpegPCMAudio path."""
        from . import adapter as _adapter

        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return False
        # Playback counts as activity: suspend the inactivity timer, re-arm in finally.
        self._cancel_voice_timeout(guild_id)
        try:
            playback_timeout = await self._playback_timeout_for_audio(audio_path)
            # ── Mixer path (overlap + ducking) ──────────────────────────────
            mixer = getattr(self, "_voice_mixers", {}).get(guild_id) if getattr(self, "_voice_mixers", None) else None
            if mixer is not None:
                decode_to_pcm = _adapter._voice_mixer_module().decode_to_pcm
                pcm = await _adapter.asyncio.to_thread(decode_to_pcm, audio_path)
                if pcm:
                    speech_gain = float(self._voice_fx_cfg.get("speech_gain", 1.0))
                    mixer.play_speech(self._lead_silence_bytes() + pcm, gain=speech_gain)
                    # Block until speech drains so callers serialise replies; ambient keeps playing.
                    wait_start = _adapter.time.monotonic()
                    while mixer.speech_active:
                        if _adapter.time.monotonic() - wait_start > playback_timeout:
                            _adapter.logger.warning("Mixer speech playback timed out after %.1fs", playback_timeout)
                            mixer.stop_speech()
                            break
                        await _adapter.asyncio.sleep(0.05)
                    return True
                _adapter.logger.warning("Mixer decode failed for %s; falling back to legacy playback", audio_path)
            # Legacy one-shot path: pause receiver while playing (echo prevention).
            receiver = self._voice_receivers.get(guild_id)
            if receiver:
                receiver.pause()
            try:
                wait_start = _adapter.time.monotonic()
                while vc.is_playing():
                    if _adapter.time.monotonic() - wait_start > playback_timeout:
                        _adapter.logger.warning("Timed out waiting for previous playback to finish")
                        vc.stop()
                        break
                    await _adapter.asyncio.sleep(0.1)
                done = _adapter.asyncio.Event()
                loop = _adapter.asyncio.get_running_loop()

                def _after(error):
                    if error:
                        _adapter.logger.error("Voice playback error: %s", error)
                    loop.call_soon_threadsafe(done.set)
                # Lead silence so socket warm-up doesn't clip the first word (mirrors mixer path).
                ffmpeg_opts: _adapter.Dict[str, _adapter.Any] = {}
                _fx_cfg = getattr(self, "_voice_fx_cfg", None) or {}
                try:
                    lead_ms = int(_fx_cfg.get("lead_silence_ms", 0) or 0)
                except (TypeError, ValueError):
                    lead_ms = 0
                if lead_ms > 0:
                    ffmpeg_opts["options"] = f"-af adelay={lead_ms}:all=1"
                source = _adapter.discord.FFmpegPCMAudio(
                    audio_path, executable=_adapter.resolve_ffmpeg_executable(), **ffmpeg_opts,
                )
                source = _adapter.discord.PCMVolumeTransformer(source, volume=1.0)
                vc.play(source, after=_after)
                try:
                    await _adapter.asyncio.wait_for(done.wait(), timeout=playback_timeout)
                except _adapter.asyncio.TimeoutError:
                    _adapter.logger.warning("Voice playback timed out after %.1fs", playback_timeout)
                    vc.stop()
                return True
            finally:
                if receiver:
                    receiver.resume()
        finally:
            self._reset_voice_timeout(guild_id)

    async def get_user_voice_channel(self, guild_id: int, user_id: str):
        """Return the voice channel the user is currently in, or None."""
        from . import adapter as _adapter

        if not self._client:
            return None
        guild = self._client.get_guild(guild_id)
        if not guild:
            return None
        member = guild.get_member(int(user_id))
        if not member or not member.voice:
            return None
        return member.voice.channel

    def _cancel_voice_timeout(self, guild_id: int) -> None:
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()

    def _reset_voice_timeout(self, guild_id: int) -> None:
        """Reset the auto-disconnect inactivity timer."""
        from . import adapter as _adapter

        self._cancel_voice_timeout(guild_id)
        timeout = self._voice_timeout_limit()
        if timeout <= 0:
            _adapter.logger.debug("Voice inactivity timeout disabled (guild=%d)", guild_id)
            return
        self._voice_timeout_tasks[guild_id] = _adapter.asyncio.ensure_future(
            self._voice_timeout_handler(guild_id, timeout)
        )

    async def _voice_timeout_handler(self, guild_id: int, timeout: Optional[int] = None) -> None:
        """Auto-disconnect after the configured inactivity timeout."""
        from . import adapter as _adapter

        timeout = self._voice_timeout_limit() if timeout is None else int(timeout)
        if timeout <= 0:
            return
        try:
            await _adapter.asyncio.sleep(timeout)
        except _adapter.asyncio.CancelledError:
            return
        text_ch_id = self._voice_text_channels.get(guild_id)
        # ``/voice off`` keeps the bot in the channel; only the bot's own audio counts as
        # activity, so the timer would fire every VOICE_TIMEOUT and spam "Left voice channel".
        _mode_getter = getattr(self, "_voice_mode_getter", None)
        if text_ch_id is not None and _mode_getter is not None:
            try:
                if _mode_getter(str(text_ch_id)) == "off":
                    return
            except Exception:
                pass
        await self.leave_voice_channel(guild_id)
        # Notify the runner so it can clean up voice_mode state
        if self._on_voice_disconnect and text_ch_id:
            try:
                self._on_voice_disconnect(str(text_ch_id))
            except Exception:
                pass
        if text_ch_id and self._client:
            ch = self._client.get_channel(text_ch_id)
            if ch:
                try:
                    await ch.send("Left voice channel (inactivity timeout).")
                except Exception:
                    pass

    def is_in_voice_channel(self, guild_id: int) -> bool:
        """Check if the bot is connected to a voice channel in this guild."""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()

    def get_voice_channel_info(self, guild_id: int) -> Optional[Dict[str, Any]]:
        """Return voice channel info (name, members, count, speaking user IDs) or None if not connected."""
        from . import adapter as _adapter

        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return None
        channel = vc.channel
        if not channel:
            return None
        members_info = []
        bot_user = self._client.user if self._client else None
        for m in channel.members:
            if bot_user and m.id == bot_user.id:
                continue  # skip the bot itself
            members_info.append({"user_id": m.id, "display_name": m.display_name, "is_bot": m.bot})
        speaking_user_ids: set = set()
        receiver = self._voice_receivers.get(guild_id)
        if receiver:
            now = _adapter.time.monotonic()
            with receiver._lock:
                for ssrc, last_t in receiver._last_packet_time.items():
                    if now - last_t < 2.0:
                        uid = receiver._ssrc_to_user.get(ssrc)
                        if uid:
                            speaking_user_ids.add(uid)
        for info in members_info:
            info["is_speaking"] = info["user_id"] in speaking_user_ids
        return {
            "channel_name": channel.name, "member_count": len(members_info),
            "members": members_info, "speaking_count": len(speaking_user_ids),
        }

    def get_voice_channel_context(self, guild_id: int) -> str:
        """Return a human-readable voice channel context string for prompt injection."""
        info = self.get_voice_channel_info(guild_id)
        if not info:
            return ""
        parts = [f"[Voice channel: #{info['channel_name']} — {info['member_count']} participant(s)]"]
        for m in info["members"]:
            status = " (speaking)" if m["is_speaking"] else ""
            parts.append(f"  - {m['display_name']}{status}")
        return "\n".join(parts)

    async def _voice_listen_loop(self, guild_id: int):
        """Periodically check for completed utterances and process them."""
        from . import adapter as _adapter

        receiver = self._voice_receivers.get(guild_id)
        if not receiver:
            return
        last_keepalive = _adapter.time.monotonic()
        try:
            while receiver._running:
                await _adapter.asyncio.sleep(0.2)
                now = _adapter.time.monotonic()
                if now - last_keepalive >= self._KEEPALIVE_INTERVAL:
                    last_keepalive = now
                    try:
                        vc = self._voice_clients.get(guild_id)
                        if vc and vc.is_connected():
                            vc._connection.send_packet(b'\xf8\xff\xfe')
                    except Exception:
                        pass
                completed = receiver.check_silence()
                # Pass guild so role checks stay guild-scoped.
                _vc_guild = self._client.get_guild(guild_id) if self._client is not None else None
                for user_id, pcm_data in completed:
                    if not self._is_allowed_user(str(user_id), guild=_vc_guild, is_dm=False):
                        continue
                    # User speech is activity too; keeps active listeners connected.
                    self._reset_voice_timeout(guild_id)
                    await self._process_voice_input(guild_id, user_id, pcm_data)
        except _adapter.asyncio.CancelledError:
            pass
        except Exception as e:
            _adapter.logger.error("Voice listen loop error: %s", e, exc_info=True)

    async def _process_voice_input(self, guild_id: int, user_id: int, pcm_data: bytes):
        """Convert PCM -> WAV -> STT -> callback."""
        from . import adapter as _adapter

        from tools.voice_mode import is_whisper_hallucination
        tmp_f = _adapter.tempfile.NamedTemporaryFile(suffix=".wav", prefix="vc_listen_", delete=False)
        wav_path = tmp_f.name
        tmp_f.close()
        try:
            await _adapter.asyncio.to_thread(_adapter.VoiceReceiver.pcm_to_wav, pcm_data, wav_path)
            from tools.transcription_tools import transcribe_audio
            result = await _adapter.asyncio.to_thread(transcribe_audio, wav_path)
            if not result.get("success"):
                return
            transcript = result.get("transcript", "").strip()
            if not transcript or is_whisper_hallucination(transcript):
                return
            _adapter.logger.info("Voice input from user %d: %s", user_id, transcript[:100])
            if self._voice_input_callback:
                await self._voice_input_callback(
                    guild_id=guild_id, user_id=user_id, transcript=transcript,
                )
        except Exception as e:
            # Surface ffmpeg's captured stderr from CalledProcessError, else log just says "exit status N".
            _ff_err = getattr(e, "stderr", None)
            if _ff_err:
                if isinstance(_ff_err, bytes):
                    _ff_err = _ff_err.decode("utf-8", "replace")
                _adapter.logger.warning(
                    "Voice input processing failed: %s (ffmpeg: %s)",
                    e, _ff_err.strip(), exc_info=True,
                )
            else:
                _adapter.logger.warning("Voice input processing failed: %s", e, exc_info=True)
        finally:
            try:
                _adapter.os.unlink(wav_path)
            except OSError:
                pass
