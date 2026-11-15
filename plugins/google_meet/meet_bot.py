"""Headless Google Meet bot — Playwright + live-caption scraping.

Standalone subprocess spawned by ``process_manager.py``; configured via ``HERMES_MEET_*`` env,
status + transcript written under ``$HERMES_MEET_OUT_DIR`` (filesystem is the only IPC).
No WebRTC audio parsing for *output*, but remote audio IS captured (see ``_RTC_HOOK_JS``):
an ``RTCPeerConnection`` wrapper intercepts ``ontrack`` remote streams and MediaRecorder
streams 10 s opus fragments out via ``page.expose_function``; each fragment is decoded
with PyAV and transcribed locally with faster-whisper (``HERMES_MEET_RTC_AUDIO=0`` to
disable, ``HERMES_MEET_WHISPER_MODEL`` to pick the model).
Debug: ``HERMES_MEET_URL=... HERMES_MEET_OUT_DIR=./meet-out HERMES_MEET_HEADED=1 \\
    python -m plugins.google_meet.meet_bot``
"""

from __future__ import annotations

import base64
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from hermes_cli.browser_runtime import chromium_executable
from utils import atomic_json_write

# Short three-segment code, a lookup URL, /new, or a Microsoft Teams meeting URL.
MEET_URL_RE = re.compile(
    r"^https://(?:meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,}|lookup/[^/?#]+|new)|teams\.(?:microsoft|live)\.com/(?:meet|l/meetup-join)/[^/?#\s]+)"
    r"(?:[/?#].*)?$")

_FFMPEG_MISSING = "ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS"


def _is_safe_meet_url(url: str) -> bool:
    """True if *url* is a Google Meet or Microsoft Teams meeting URL we're willing to navigate to."""
    return isinstance(url, str) and bool(MEET_URL_RE.match(url.strip()))


def _meeting_id_from_url(url: str) -> str:
    """3-segment meeting code, teams-<id>, or a timestamped id for ``/lookup/...`` and ``/new``."""
    m = re.search(r"meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,})", url or "")
    if m:
        return m.group(1)
    mt = re.search(r"teams\.(?:microsoft|live)\.com/meet/([A-Za-z0-9_-]+)", url or "")
    if mt:
        return f"teams-{mt.group(1)[:16]}"
    return f"meet-{int(time.time())}"


def _quiet(fn, *args, **kwargs):
    """Call *fn*, swallowing any exception (best-effort teardown steps)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


# status.json keys in file order → _BotState attribute + initial value.
_STATUS_FIELDS = (
    ("meetingId", "meeting_id", None), ("url", "url", None), ("inCall", "in_call", False),
    ("captioning", "captioning", False), ("captionsEnabledAttempted", "captions_enabled_attempted", False),
    ("lobbyWaiting", "lobby_waiting", False), ("joinAttemptedAt", "join_attempted_at", None),
    ("joinedAt", "joined_at", None), ("lastCaptionAt", "last_caption_at", None),
    ("transcriptLines", "transcript_lines", 0), ("transcriptPath", "transcript_path", None),
    ("error", "error", None), ("exited", "exited", False), ("pid", None, None),
    # realtime telemetry
    ("realtime", "realtime", False), ("realtimeReady", "realtime_ready", False),
    ("realtimeDevice", "realtime_device", None), ("audioBytesOut", "audio_bytes_out", 0),
    ("lastAudioOutAt", "last_audio_out_at", None), ("lastBargeInAt", "last_barge_in_at", None),
    ("leaveReason", "leave_reason", None), ("micState", "mic_state", None),
    # rtc capture telemetry
    ("rtcCapturing", "rtc_capturing", False), ("audioFragments", "audio_fragments", 0),
    ("whisperModel", "whisper_model", None))


class _BotState:
    """Single-process mutable state, flushed to ``status.json`` on each change."""

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self.__dict__.update({attr: default for _, attr, default in _STATUS_FIELDS if attr})
        self.__dict__.update(out_dir=out_dir, meeting_id=meeting_id, url=url, _seen=set(),  # seen "speaker|text"
                             speaker_timeline=[],  # list of (epoch_ts, speaker_name) from DOM active-speaker/captions
                             transcript_path=out_dir / "transcript.txt", status_path=out_dir / "status.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._flush()

    def record_dom_speaker(self, speaker: str, epoch_ts: Optional[float] = None) -> None:
        """Record an active human speaker observed in the Meet/Teams DOM at *epoch_ts*."""
        spk = (speaker or "").strip()
        if not spk or spk.lower() in {"unknown", "you", "du", "mig", "me"}:
            return
        if re.search(r"(notetaker|tl\s*;?\s*dv|tldv|mötesanteckningar|motesanteckningar|techzola\s+assistant)", spk, re.IGNORECASE):
            return
        ts = float(epoch_ts) if epoch_ts is not None else time.time()
        self.speaker_timeline.append((ts, spk))
        if len(self.speaker_timeline) > 5000:
            self.speaker_timeline = self.speaker_timeline[-2500:]

    def resolve_speaker_at(self, epoch_ts: float, fallback: str, window_s: float = 6.0) -> str:
        """Return the closest DOM-observed active speaker within *window_s* of *epoch_ts*, or *fallback*."""
        if not self.speaker_timeline:
            return fallback
        best_dt, best_spk = window_s + 1.0, fallback
        for ts, spk in self.speaker_timeline:
            dt = abs(float(epoch_ts) - float(ts))
            if dt <= window_s and dt < best_dt:
                best_dt, best_spk = dt, spk
        return best_spk

    def record_caption(self, speaker: str, text: str, offset_s: Optional[float] = None) -> None:
        """Append a caption line unless this exact (speaker, text) was already seen."""
        speaker, text = (speaker or "").strip() or "Unknown", (text or "").strip()
        key = f"{speaker}|{text}"
        if not text or key in self._seen:
            return
        self._seen.add(key)
        self.transcript_lines += 1
        now_ts = time.time()
        self.last_caption_at = now_ts
        anchor = self.joined_at or self.join_attempted_at
        cap_epoch = (float(anchor) + float(offset_s)) if (offset_s is not None and anchor) else now_ts
        ts = time.strftime("%H:%M:%S", time.localtime(cap_epoch))
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {speaker}: {text}\n")
        self._flush()

    def _flush(self) -> None:
        data = {key: getattr(self, attr) if attr else None for key, attr, _ in _STATUS_FIELDS}
        data.update(transcriptPath=str(self.transcript_path), pid=os.getpid())  # keeps table key order
        atomic_json_write(self.status_path, data)

    def set(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self._flush()


# JS injected before page load: wraps RTCPeerConnection so every remote audio stream
# Meet sets up is recorded by its own MediaRecorder; 10 s opus fragments are shipped to
# Python via the ``__hermesMeetAudioChunk`` binding (base64 through readAsDataURL —
# page.evaluate corrupts bytes > 127). One stream per remote participant (Meet's unified
# plan), so per-stream transcription doubles as speaker separation.
_RTC_HOOK_JS = r"""
(() => {
  if (navigator.mediaDevices && !window.__hermesSynthMedia) {
    window.__hermesSynthMedia = true;
    const makeAudioTrack = () => {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.0;
      const dst = ctx.createMediaStreamDestination();
      osc.connect(gain);
      gain.connect(dst);
      osc.start();
      return dst.stream.getAudioTracks()[0];
    };
    const makeVideoTrack = () => {
      const c = document.createElement("canvas");
      c.width = 640; c.height = 360;
      const g = c.getContext("2d");
      g.fillStyle = "#111"; g.fillRect(0, 0, 640, 360);
      return c.captureStream(1).getVideoTracks()[0];
    };
    navigator.mediaDevices.enumerateDevices = async () => [
      { deviceId: "default", kind: "audioinput", label: "Virtual Mic", groupId: "g1", toJSON() { return this; } },
      { deviceId: "default", kind: "videoinput", label: "Virtual Cam", groupId: "g2", toJSON() { return this; } },
      { deviceId: "default", kind: "audiooutput", label: "Virtual Speaker", groupId: "g3", toJSON() { return this; } }
    ];
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const tracks = [];
      if (constraints && constraints.audio) tracks.push(makeAudioTrack());
      if (constraints && constraints.video) tracks.push(makeVideoTrack());
      return new MediaStream(tracks);
    };
  }
  window.__hermesRtcStreams = new Map();
  window.__hermesPeerConns = new Set();
  const CHUNK_MS = window.__hermesMeetChunkMs || 10000;
  function hookTrack(track) {
    try {
      if (!track || track.kind !== 'audio' || track.__hermesRtc) return;
      track.__hermesRtc = true;
      const audioStream = new MediaStream([track]);
      try {
        if (!window.__hermesPullCtx) {
          window.__hermesPullCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (window.__hermesPullCtx.state === 'suspended') {
          window.__hermesPullCtx.resume();
        }
        const src = window.__hermesPullCtx.createMediaStreamSource(audioStream);
        const silentGain = window.__hermesPullCtx.createGain();
        silentGain.gain.value = 0.0001;
        src.connect(silentGain);
        silentGain.connect(window.__hermesPullCtx.destination);
        window.__hermesPullNodes = window.__hermesPullNodes || [];
        window.__hermesPullNodes.push({src, silentGain, audioStream, track});
      } catch (e) {}
      track.addEventListener('unmute', () => {
        try {
          if (window.__hermesPullCtx && window.__hermesPullCtx.state === 'suspended') {
            window.__hermesPullCtx.resume();
          }
        } catch (e) {}
      });
      const id = window.__hermesRtcStreams.size + 1;
      const mr = new MediaRecorder(audioStream, {mimeType: 'audio/webm;codecs=opus'});
      mr.ondataavailable = e => {
        if (!e.data || e.data.size === 0) return;
        const r = new FileReader();
        r.onload = () => {
          try { window.__hermesMeetAudioChunk(id, r.result.split(',')[1]); } catch (err) {}
        };
        r.readAsDataURL(e.data);
      };
      mr.start(CHUNK_MS);
      window.__hermesRtcStreams.set(id, mr);
    } catch (e) { window.__hermesRtcErr = String(e); }
  }
  function hookStream(stream) {
    try {
      if (!stream || !stream.getAudioTracks) return;
      for (const t of stream.getAudioTracks()) hookTrack(t);
    } catch (e) {}
  }
  const OrigPC = window.RTCPeerConnection;
  function PatchedPC(...args) {
    const pc = new OrigPC(...args);
    try {
      window.__hermesPeerConns.add(pc);
      pc.addEventListener('track', ev => {
        try {
          if (ev.track) hookTrack(ev.track);
          if (ev.streams) for (const s of ev.streams) hookStream(s);
        } catch (e) {}
      });
    } catch (e) {}
    return pc;
  }
  PatchedPC.prototype = OrigPC.prototype;
  window.RTCPeerConnection = PatchedPC;
  window.webkitRTCPeerConnection = PatchedPC;
  const desc = Object.getOwnPropertyDescriptor(OrigPC.prototype, 'ontrack');
  if (desc && desc.set) {
    Object.defineProperty(OrigPC.prototype, 'ontrack', {
      get: desc.get,
      set(fn) {
        desc.set.call(this, function(ev) {
          try {
            if (ev.track) hookTrack(ev.track);
            if (ev.streams) for (const s of ev.streams) hookStream(s);
          } catch (e) {}
          if (fn) return fn.call(this, ev);
        });
      }
    });
  }
  try {
    const srcDesc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
    if (srcDesc && srcDesc.set) {
      Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
        get: srcDesc.get,
        set(val) {
          try { if (val) hookStream(val); } catch (e) {}
          return srcDesc.set.call(this, val);
        }
      });
    }
    const origPlay = HTMLMediaElement.prototype.play;
    if (origPlay) {
      HTMLMediaElement.prototype.play = function(...args) {
        try { if (this.srcObject) hookStream(this.srcObject); } catch (e) {}
        return origPlay.apply(this, args);
      };
    }
  } catch (e) {}
  setInterval(() => {
    try {
      if (window.__hermesPullCtx && window.__hermesPullCtx.state === 'suspended') {
        window.__hermesPullCtx.resume();
      }
      for (const pc of window.__hermesPeerConns) {
        if (pc.connectionState === 'closed') continue;
        for (const r of (pc.getReceivers ? pc.getReceivers() : [])) {
          if (r && r.track && r.track.kind === 'audio' && r.track.readyState === 'live') {
            hookTrack(r.track);
          }
        }
      }
      for (const el of document.querySelectorAll('audio,video')) {
        if (el.srcObject && el.srcObject.getAudioTracks) {
          hookStream(el.srcObject);
        }
      }
    } catch (e) {}
  }, 2000);
  window.__hermesMeetRtcStop = () => {
    for (const mr of window.__hermesRtcStreams.values()) {
      try { if (mr.state !== 'inactive') mr.stop(); } catch (e) {}
    }
  };
  window.__hermesMeetRtcPoll = () => ({streams: window.__hermesRtcStreams.size,
                                       err: window.__hermesRtcErr || null});
})();
"""


def _patch_silero_vad_batch_limit(max_batch: int = 1500) -> None:
    """Cap SileroVADModel encoder_batch_size from 10000 to max_batch to avoid a 2.4 GB ONNX arena."""
    try:
        from faster_whisper import vad as fw_vad
        import numpy as np
        if getattr(fw_vad.SileroVADModel, "_hermes_batch_patched", False):
            return
        def _bounded_call(self, audio, num_samples: int = 512, context_size_samples: int = 64):
            h = np.zeros((1, 1, 128), dtype="float32")
            c = np.zeros((1, 1, 128), dtype="float32")
            batched_audio = audio.reshape(-1, num_samples)
            context = batched_audio[..., -context_size_samples:].copy()
            context[-1] = 0
            context = np.roll(context, 1, 0)
            batched_audio = np.concatenate([context, batched_audio], 1).reshape(
                -1, num_samples + context_size_samples
            )
            outputs = []
            for i in range(0, batched_audio.shape[0], max_batch):
                output, h, c = self.session.run(
                    None, {"input": batched_audio[i : i + max_batch], "h": h, "c": c}
                )
                outputs.append(output)
            return np.concatenate(outputs, axis=0)
        fw_vad.SileroVADModel.__call__ = _bounded_call
        fw_vad.SileroVADModel._hermes_batch_patched = True
    except Exception:
        pass


def _is_ct2_float16(model_bin: Path) -> bool:
    if not model_bin.is_file():
        return False
    import struct
    dtypes = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
    try:
        with model_bin.open("rb") as fin:
            def read_str():
                n = struct.unpack("H", fin.read(2))[0]
                s = fin.read(n - 1).decode("utf-8")
                fin.read(1)
                return s
            _ver = struct.unpack("I", fin.read(4))[0]
            _spec = read_str()
            _rev = struct.unpack("I", fin.read(4))[0]
            nvars = struct.unpack("I", fin.read(4))[0]
            for _ in range(min(nvars, 20)):
                vname = read_str()
                rank = struct.unpack("B", fin.read(1))[0]
                _dims = [struct.unpack("I", fin.read(4))[0] for _ in range(rank)]
                tid = struct.unpack("B", fin.read(1))[0]
                nbytes = struct.unpack("I", fin.read(4))[0]
                if vname.endswith("/weight"):
                    return dtypes[tid] == "float16"
                fin.seek(nbytes, 1)
    except Exception:
        return False
    return False


def _ensure_ct2_int8_on_disk(model_path_or_id: str) -> bool:
    """If model_path_or_id points to a float16 CTranslate2 model.bin, atomically pre-quantize it to int8 on disk.
    Eliminates the 6.1 GB peak / 2.9 GB MALLOC_LARGE runtime float16->int8 quantization spike per process."""
    p = Path(str(model_path_or_id))
    model_bin = p / "model.bin" if p.is_dir() else (p if p.name == "model.bin" else None)
    if model_bin is None or not _is_ct2_float16(model_bin):
        return False
    import struct
    import numpy as np
    target_bin = model_bin.resolve()
    dtypes = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
    np_dtypes = {"float32": np.float32, "int8": np.int8, "int16": np.int16, "int32": np.int32, "float16": np.float16}
    entries = []
    with target_bin.open("rb") as fin:
        def read_str():
            n = struct.unpack("H", fin.read(2))[0]
            s = fin.read(n - 1).decode("utf-8")
            fin.read(1)
            return s
        ver = struct.unpack("I", fin.read(4))[0]
        spec_name = read_str()
        rev = struct.unpack("I", fin.read(4))[0]
        nvars = struct.unpack("I", fin.read(4))[0]
        for _ in range(nvars):
            vname = read_str()
            rank = struct.unpack("B", fin.read(1))[0]
            dims = [struct.unpack("I", fin.read(4))[0] for _ in range(rank)]
            tid = struct.unpack("B", fin.read(1))[0]
            nbytes = struct.unpack("I", fin.read(4))[0]
            offset = fin.tell()
            fin.seek(nbytes, 1)
            dt_str = dtypes[tid]
            is_q = vname.endswith("/weight") and len(dims) == 2 and dt_str in ("float16", "float32") and not vname.endswith(("/conv1/weight", "/conv2/weight", "/embeddings/weight"))
            entries.append((vname, dims, tid, nbytes, offset, is_q))
        naliases = struct.unpack("I", fin.read(4))[0]
        aliases = [(read_str(), read_str()) for _ in range(naliases)]

    plan = []
    for vname, dims, tid, nbytes, offset, is_q in entries:
        if is_q:
            plan.append((vname, "quant_weight", dims, tid, nbytes, offset))
            plan.append((f"{vname}_scale", "quant_scale", dims, tid, nbytes, offset))
        elif dtypes[tid] == "float16":
            plan.append((vname, "to_f32", dims, tid, nbytes, offset))
        else:
            plan.append((vname, "raw", dims, tid, nbytes, offset))
    plan.sort(key=lambda x: x[0])

    out_aliases = []
    for al, target in aliases:
        out_aliases.append((al, target))
        if target.endswith("/weight") and not target.endswith(("/conv1/weight", "/conv2/weight", "/embeddings/weight")):
            out_aliases.append((f"{al}_scale", f"{target}_scale"))

    tmp_bin = target_bin.parent / f"{target_bin.name}.int8.{os.getpid()}.tmp"
    cache_last = {}
    with target_bin.open("rb") as fin, tmp_bin.open("wb") as fout:
        def write_str(s: str):
            b = s.encode("utf-8")
            fout.write(struct.pack("H", len(b) + 1))
            fout.write(b)
            fout.write(struct.pack("B", 0))
        fout.write(struct.pack("I", ver))
        write_str(spec_name)
        fout.write(struct.pack("I", rev))
        fout.write(struct.pack("I", len(plan)))

        for out_name, kind, dims, tid, nbytes, offset in plan:
            if kind in ("quant_weight", "quant_scale"):
                base_name = out_name[:-6] if kind == "quant_scale" else out_name
                if base_name not in cache_last:
                    cache_last.clear()
                    fin.seek(offset)
                    raw = fin.read(nbytes)
                    arr = np.frombuffer(raw, dtype=np_dtypes[dtypes[tid]]).reshape(dims).astype(np.float32)
                    old_shape = None
                    if len(arr.shape) == 3:
                        old_shape = arr.shape
                        arr = arr.reshape(arr.shape[0], -1)
                    amax = np.amax(np.absolute(arr), axis=1)
                    amax[amax == 0] = 127.0
                    scale = (127.0 / amax).astype(np.float32)
                    q_arr = np.rint(arr * np.expand_dims(scale, 1)).astype(np.int8)
                    if old_shape:
                        q_arr = q_arr.reshape(old_shape)
                    cache_last[base_name] = (
                        (list(q_arr.shape), 1, q_arr.tobytes()),
                        (list(scale.shape), 0, scale.tobytes()),
                    )
                    del arr, q_arr, scale, raw
                w_item, s_item = cache_last[base_name]
                out_dims, out_tid, out_bytes = w_item if kind == "quant_weight" else s_item
            elif kind == "to_f32":
                fin.seek(offset)
                raw = fin.read(nbytes)
                arr = np.frombuffer(raw, dtype=np.float16).reshape(dims).astype(np.float32)
                out_dims, out_tid, out_bytes = dims, 0, arr.tobytes()
            else:
                fin.seek(offset)
                out_dims, out_tid, out_bytes = dims, tid, fin.read(nbytes)

            write_str(out_name)
            fout.write(struct.pack("B", len(out_dims)))
            for d in out_dims:
                fout.write(struct.pack("I", d))
            fout.write(struct.pack("B", out_tid))
            fout.write(struct.pack("I", len(out_bytes)))
            fout.write(out_bytes)

        fout.write(struct.pack("I", len(out_aliases)))
        for al, target in out_aliases:
            write_str(al)
            write_str(target)

    os.replace(tmp_bin, target_bin)
    return True


class _RtcTranscriber:
    """Decodes recorded WebRTC audio and transcribes it with faster-whisper.

    MediaRecorder timeslice blobs are NOT standalone files: only the first blob per
    stream carries the EBML/Tracks header; continuation blobs are raw slices of the
    same byte stream (sometimes with a fresh EBML header when Chrome restarts a
    segment). So fragments are appended, in order, to one growing ``stream_NN.webm``
    per stream — re-headers stripped down to the first Cluster element — and the
    merged file is re-decoded per batch, transcribing only the PCM beyond what was
    already transcribed. One worker thread per stream keeps fragments in order;
    silence is skipped by a peak gate and VAD. Whisper runs lazily so the bot still
    joins/records when av/faster-whisper are missing (merged audio stays on disk
    under ``audio/`` for post-processing).
    """

    _EBML_MAGIC = b"\x1a\x45\xdf\xa3"
    _CLUSTER_ID = b"\x1f\x43\xb6\x75"

    def __init__(self, cfg: _BotConfig, state: _BotState, stop_flag: dict):
        self.cfg, self.state, self.stop = cfg, state, stop_flag
        self._queue = deque()          # (stream_id, nbytes)
        self._lock = threading.Lock()
        self._whisper_lock = threading.Lock()
        self.threads = []
        self._last_text = {}
        self._transcribed = {}         # stream_id → samples already transcribed
        self._decoded_bytes = {}       # stream_id → byte offset in merged WebM already decoded
        self._webm_headers = {}        # stream_id → EBML+Tracks header bytes before first Cluster
        self._detected_language = None
        self._model = None
        self._whisper_failed = False

    def _whisper(self):
        if self._model is not None:
            return self._model
        with self._whisper_lock:
            if self._model is None:
                _patch_silero_vad_batch_limit(1500)
                _ensure_ct2_int8_on_disk(self.cfg.whisper_model)
                from faster_whisper import WhisperModel
                cpu_threads = int(os.environ.get("HERMES_MEET_CPU_THREADS", "2") or 2)
                self._model = WhisperModel(
                    self.cfg.whisper_model, device="cpu", compute_type="int8",
                    cpu_threads=cpu_threads,
                    download_root=str(Path.home() / ".open-webui/cache/whisper/models"))
        return self._model

    def _decode_bytes(self, buf: bytes):
        import io
        import av
        import numpy as np
        c = av.open(io.BytesIO(buf))
        astream = next((s for s in c.streams if s.type == "audio"), None)
        if astream is None:
            c.close()
            return None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm = bytearray()
        for frame in c.decode(astream):
            rf = resampler.resample(frame)
            for f in (rf if isinstance(rf, list) else [rf]):
                pcm.extend(f.to_ndarray().tobytes())
        c.close()
        return np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0

    def _decode(self, path: Path):
        import av
        import numpy as np
        c = av.open(str(path))
        astream = next((s for s in c.streams if s.type == "audio"), None)
        if astream is None:
            c.close()
            return None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm = bytearray()
        for frame in c.decode(astream):
            rf = resampler.resample(frame)
            for f in (rf if isinstance(rf, list) else [rf]):
                pcm.extend(f.to_ndarray().tobytes())
        c.close()
        return np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0

    def _merge_fragment_immediately(self, stream_id: int, path: Path) -> int:
        merged = self.cfg.out_dir / "audio" / f"stream_{stream_id:02d}.webm"
        merged.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = path.read_bytes()
            if data.startswith(self._EBML_MAGIC):
                cluster = data.find(self._CLUSTER_ID)
                if cluster > 0 and stream_id not in self._webm_headers:
                    self._webm_headers[stream_id] = data[:cluster]
                if merged.exists() and merged.stat().st_size > 0 and cluster > 0:
                    data = data[cluster:]
            with merged.open("ab") as f:
                f.write(data)
            self.state.set(audio_fragments=self.state.audio_fragments + 1)
            _quiet(path.unlink)
            return len(data)
        except Exception as e:
            sys.stderr.write(f"meet rtc: merge {path.name} failed: {e}\n")
            return 0

    def submit(self, stream_id: int, path: Path) -> None:
        with self._lock:
            nbytes = self._merge_fragment_immediately(stream_id, path)
            self._queue.append((stream_id, nbytes))
        alive = any(t.name == f"meet-rtc-{stream_id}" and t.is_alive() for t in self.threads)
        if not alive:
            t = threading.Thread(target=self._worker, args=(stream_id,),
                                 name=f"meet-rtc-{stream_id}", daemon=True)
            self.threads.append(t)
            t.start()

    def _pop(self, stream_id):
        with self._lock:
            for item in self._queue:
                if item[0] == stream_id:
                    self._queue.remove(item)
                    return item
        return None

    def _drain_stream_queue(self, stream_id: int) -> int:
        """Drain all queued items for stream_id whose bytes are already merged on disk."""
        drained = 0
        with self._lock:
            keep = deque()
            for item in self._queue:
                if item[0] == stream_id:
                    drained += int(item[1] or 0)
                else:
                    keep.append(item)
            self._queue = keep
        return drained

    def _worker(self, stream_id: int) -> None:
        label = f"Voice {stream_id}"
        merged = self.cfg.out_dir / "audio" / f"stream_{stream_id:02d}.webm"
        merged.parent.mkdir(parents=True, exist_ok=True)  # standalone/offline use
        fresh_bytes, last_flush = 0, time.time()
        flush_sec = float(os.environ.get("HERMES_MEET_FLUSH_SEC", "15") or 15)
        idle_flush_sec = max(flush_sec + 1.0, 25.0 if flush_sec >= 15.0 else flush_sec * 1.5)
        while True:
            item = self._pop(stream_id)
            if item is None:
                if self.stop.get("stop"):
                    self._drain_stream_queue(stream_id)
                    self._flush_stream(stream_id, merged, label)
                    return
                # idle: flush whatever accumulated so the transcript stays near-live
                if fresh_bytes > 0 and time.time() - last_flush > idle_flush_sec:
                    self._drain_stream_queue(stream_id)
                    self._flush_stream(stream_id, merged, label)
                    fresh_bytes, last_flush = 0, time.time()
                time.sleep(0.5)
                continue
            _, nbytes = item
            fresh_bytes += int(nbytes or 0)
            if fresh_bytes > 400_000 or time.time() - last_flush > flush_sec:
                self._drain_stream_queue(stream_id)
                self._flush_stream(stream_id, merged, label)
                fresh_bytes, last_flush = 0, time.time()

    def _decode_incremental_tail(self, stream_id: int, merged: Path):
        """Decode only the newly appended WebM clusters in O(1) time; fall back to full _decode if needed."""
        done = self._transcribed.get(stream_id, 0)
        min_tail_samples = int(16000 * (1.2 if os.environ.get("HERMES_MEET_REALTIME_STANDUP") == "1" else 5.0))
        if "_decode" in self.__dict__:
            audio = self._decode(merged)
            if audio is None or len(audio) <= done + min_tail_samples:
                return None, done
            tail = audio[done:]
            self._transcribed[stream_id] = len(audio)
            return tail, done

        file_size = merged.stat().st_size
        prev_off = self._decoded_bytes.get(stream_id, 0)
        if prev_off > 0 and file_size <= prev_off:
            return None, done

        try:
            with merged.open("rb") as f:
                if stream_id not in self._webm_headers:
                    head_probe = f.read(4096)
                    if head_probe.startswith(self._EBML_MAGIC):
                        first_cl = head_probe.find(self._CLUSTER_ID)
                        if first_cl > 0:
                            self._webm_headers[stream_id] = head_probe[:first_cl]
                header = self._webm_headers.get(stream_id)
                if header is not None:
                    start_off = max(prev_off, len(header))
                    if file_size <= start_off:
                        return None, done
                    f.seek(start_off)
                    new_clusters = f.read(file_size - start_off)
                    if new_clusters.startswith(self._CLUSTER_ID):
                        tail = self._decode_bytes(header + new_clusters)
                        if tail is not None:
                            if len(tail) <= min_tail_samples:
                                return None, done
                            self._decoded_bytes[stream_id] = file_size
                            self._transcribed[stream_id] = done + len(tail)
                            return tail, done
        except Exception:
            pass

        audio = self._decode(merged)
        if audio is None or len(audio) <= done + min_tail_samples:
            return None, done
        tail = audio[done:]
        self._decoded_bytes[stream_id] = file_size
        self._transcribed[stream_id] = len(audio)
        return tail, done

    def _flush_stream(self, stream_id: int, merged: Path, label: str) -> None:
        """Decode the merged stream file incrementally and transcribe only the PCM not yet done."""
        if not merged.exists() or merged.stat().st_size == 0:
            return
        try:
            tail, done = self._decode_incremental_tail(stream_id, merged)
        except Exception as e:
            sys.stderr.write(f"meet rtc: decode {merged.name} failed: {e}\n")
            return
        if tail is None:
            return
        try:
            peak = float(abs(tail).max())
            sys.stderr.write(f"meet rtc: flush stream {stream_id}: +{len(tail)/16000:.1f}s new audio, peak {peak:.4f}\n")
            if peak < 0.01:  # digital silence / DTX noise floor
                return
        except Exception:
            return
        speech_dur_s = 0.0
        try:
            _patch_silero_vad_batch_limit(1500)
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            speeches = get_speech_timestamps(
                tail, VadOptions(threshold=0.5, min_speech_duration_ms=350, speech_pad_ms=200)
            )
            if not speeches:
                return
            speech_dur_s = sum(sp["end"] - sp["start"] for sp in speeches) / 16000.0
        except Exception:
            pass
        try:
            model = self._whisper()
        except Exception as e:
            if not self._whisper_failed:
                self._whisper_failed = True
                self.state.set(error=f"whisper unavailable ({e}) — merged audio kept on disk")
            return
        try:
            lang_env = os.environ.get("HERMES_MEET_LANGUAGE", "").strip() or None
            multilang_set = {
                x.strip().lower()
                for x in os.environ.get("HERMES_MEET_MULTILANG", "").split(",")
                if x.strip()
            }
            if lang_env == "auto" or lang_env is None:
                lang_env = None if multilang_set else self._detected_language
            prompt_env = os.environ.get("HERMES_MEET_INITIAL_PROMPT", "").strip() or self._last_text.get(stream_id) or None
            with self._whisper_lock:
                segs_iter, _info = model.transcribe(
                    tail,
                    language=lang_env,
                    vad_filter=True,
                    vad_parameters=dict(threshold=0.5, min_speech_duration_ms=350, speech_pad_ms=200),
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=prompt_env,
                )
                segs = list(segs_iter)
                det_lang = getattr(_info, "language", None)
                if multilang_set and det_lang not in multilang_set and det_lang not in ("no", "da"):
                    fallback_lang = getattr(self, "_stream_lang", {}).get(stream_id, "en")
                    segs_iter, _info = model.transcribe(
                        tail,
                        language=fallback_lang,
                        vad_filter=True,
                        vad_parameters=dict(threshold=0.5, min_speech_duration_ms=350, speech_pad_ms=200),
                        beam_size=1,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        initial_prompt=prompt_env,
                    )
                    segs = list(segs_iter)
                elif multilang_set and det_lang in multilang_set:
                    if not hasattr(self, "_stream_lang"):
                        self._stream_lang = {}
                    self._stream_lang[stream_id] = det_lang
                if (
                    not multilang_set
                    and self._detected_language is None
                    and speech_dur_s >= 3.0
                    and det_lang
                    and float(getattr(_info, "language_probability", 0.0) or 0.0) >= 0.80
                ):
                    self._detected_language = det_lang
            base_offset_s = float(done) / 16000.0
            _hallucination_pat = re.compile(
                r"^(?:mikael wiehe:?|tack till elever och personal.*|undertextning:?.*|textning\.nu.*)$",
                re.IGNORECASE,
            )
            for s in segs:
                text = (s.text or "").strip()
                if re.search(r"\b(\w+)(?:[\s,]+\1){4,}", text, re.IGNORECASE):
                    continue
                if text and not _hallucination_pat.match(text):
                    seg_offset_s = base_offset_s + float(getattr(s, "start", 0.0) or 0.0)
                    anchor = self.state.joined_at or self.state.join_attempted_at or time.time()
                    seg_epoch = float(anchor) + seg_offset_s
                    resolved_speaker = self.state.resolve_speaker_at(seg_epoch, fallback=label)
                    self.state.record_caption(speaker=resolved_speaker, text=text, offset_s=seg_offset_s)
                    self._last_text[stream_id] = text
        except Exception as e:
            sys.stderr.write(f"meet rtc: transcribe {merged.name} failed: {e}\n")


# JS injected into the Meet tab: MutationObserver on the caption container
# collects {speaker, text}; ``window.__hermesMeetDrain()`` pulls new entries.
_CAPTION_OBSERVER_JS = r"""
(() => {
  if (window.__hermesMeetInstalled) return;
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = [];

  const captionSelector = '[role="region"][aria-label*="aption" i], ' +
                          'div[jsname="YSxPC"], ' +  // legacy
                          'div[jsname="tgaKEf"]';    // current (Apr 2026)

  function pushEntry(speaker, text) {
    if (!text || !text.trim()) return;
    window.__hermesMeetQueue.push({
      ts: Date.now(),
      speaker: (speaker || '').trim(),
      text: text.trim(),
    });
  }

  function scan(root) {
    // Meet captions render as rows of speaker label + text block. Selectors
    // vary across Meet rewrites; try a few shapes and fall back to raw text.
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        const spkEl = row.querySelector('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb');
        pushEntry(spkEl ? spkEl.innerText : '', txtEl ? txtEl.innerText : row.innerText);
      });
      return;
    }
    // Fallback: treat the whole region's innerText as one anonymous line.
    pushEntry('', (root.innerText || '').split('\n').filter(Boolean).pop());
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (!el) return false;
    new MutationObserver(() => scan(el)).observe(el, { childList: true, subtree: true, characterData: true });
    scan(el);
    return true;
  }

  // Retry on interval — the caption region only appears after captions are
  // enabled and someone speaks.
  if (!attach()) {
    const iv = setInterval(() => { if (attach()) clearInterval(iv); }, 1500);
  }

  window.__hermesMeetDrain = () => {
    const out = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return out;
  };

  window.__hermesSpeakerQueue = [];
  window.__hermesActiveSpeaker = () => {
    const botRe = /(notetaker|tl\s*;?\s*dv|tldv|mötesanteckningar|motesanteckningar|techzola\s+assistant|fireflies|otter\.ai)/i;
    const selfRe = /^(\s*(du|you|mig|me)\s*)$/i;
    // 1. Check active-speaker indicators on Meet & Teams participant tiles
    const tiles = Array.from(document.querySelectorAll('[data-participant-id], [data-cid="calling-participant-stream"], [data-tid="roster-participant"]'));
    for (const t of tiles) {
      const isSpeaking = !!(
        t.querySelector('[aria-label*="speaking" i], [aria-label*="pratar" i], [aria-label*="talar" i], [data-tid="voice-level-stream-outline"], .kssMZb, .S7urwe') ||
        t.getAttribute('data-is-speaking') === 'true' ||
        t.classList.contains('active-speaker')
      );
      if (!isSpeaking) continue;
      const selfName = (t.getAttribute('data-self-name') || '').trim();
      const nameEl = t.querySelector('.zWGUib, .KcIKyf, .zs7s8d, [data-tid="participant-name"]');
      const spk = (selfName || (nameEl ? nameEl.innerText : '') || '').trim();
      if (spk && !botRe.test(spk) && !selfRe.test(spk)) return spk;
    }
    // 2. Fallback: latest speaker header in live caption region
    const spkEls = document.querySelectorAll('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"]');
    if (spkEls.length) {
      const lastSpk = (spkEls[spkEls.length - 1].innerText || '').trim();
      if (lastSpk && !botRe.test(lastSpk) && !selfRe.test(lastSpk)) return lastSpk;
    }
    return null;
  };
  setInterval(() => {
    const spk = window.__hermesActiveSpeaker && window.__hermesActiveSpeaker();
    if (spk) {
      window.__hermesSpeakerQueue.push({ ts: Date.now() / 1000.0, speaker: spk });
      if (window.__hermesSpeakerQueue.length > 500) window.__hermesSpeakerQueue.shift();
    }
  }, 500);
  window.__hermesDrainSpeakers = () => {
    const out = window.__hermesSpeakerQueue.slice();
    window.__hermesSpeakerQueue = [];
    return out;
  };
})();
"""

# Best-effort caption toggle: Meet binds it to the ``c`` key; click targeting is too brittle.
_ENABLE_CAPTIONS_JS = (
    "(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', "
    "{ key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true })); return true; })();")

_LEAVE_CALL_JS = (
    "() => { const b = document.querySelector('button[aria-label*=\"eave call\" i], button[aria-label*=\"ämna samtal\" i], button[aria-label*=\"Avsluta samtal\" i], button[id=\"hangup-button\"], button[data-tid=\"hangup-main-btn\"], button[data-tid*=\"hangup\" i]');"
    " if (b) b.click(); }")

# True once past the lobby: leave button, caption region (once our observer is installed)
# or participant list visible (supports both English and Swedish Meet UI + Microsoft Teams web UI).
_ADMISSION_PROBE_JS = r"""
    (() => {
      const bodyTxt = document.body ? (document.body.innerText || '') : '';
      if (/Vi har meddelat personerna i mötet att du väntar|Someone in the meeting should let you in soon|Vi släpper in dig snart|Väntar i lobbyn|Waiting in lobby|Redo att gå med\?|Ready to join\?/i.test(bodyTxt)) return false;
      const openDlg = document.querySelector('div[role="dialog"], div[ role="alertdialog"]');
      if (openDlg && /Gå med nu|Join now|spelas in|being recorded|transkrib|Anteckningar förs|taking notes/i.test(openDlg.innerText || '')) return false;
      if (document.querySelector('button[aria-label*="eave call" i], button[aria-label*="ämna samtal" i], button[aria-label*="Avsluta samtal" i], button[id="hangup-button"], button[data-tid="hangup-main-btn"], button[data-tid*="hangup" i]')) return true;
      if (window.__hermesMeetInstalled && document.querySelector(
          '[role="region"][aria-label*="aption" i], [role="region"][aria-label*="textning" i], div[jsname="YSxPC"], div[jsname="tgaKEf"]')) return true;
      return !!document.querySelector('[aria-label*="articipants" i]');
    })();
    """

# Count of HUMAN participant tiles in call (returns 1 when bot is alone or only with other notetaker bots like tl;dv).
_PARTICIPANT_COUNT_JS = r"""
    (() => {
      const bodyTxt = document.body ? (document.body.innerText || '') : '';
      if (/Det finns ingen annan här|You're the only one here|Ingen annan är här|Väntar på att andra ska ansluta|Waiting for others to join/i.test(bodyTxt)) return 1;
      const botRe = /(notetaker|tl\s*;?\s*dv|tldv|mötesanteckningar|motesanteckningar|techzola\s+assistant|fireflies|otter\.ai|fathom|read\.ai|gemini\s+notes|meeting\s+bot|recording\s+bot)/i;
      const selfRe = /^(\s*(du|you|mig|me)\s*)$/i;
      const tiles = Array.from(document.querySelectorAll('[data-participant-id], [data-cid="calling-participant-stream"], [data-tid="roster-participant"]'));
      if (tiles.length > 0) {
        const seenIds = new Set();
        let humanOtherCount = 0;
        for (const t of tiles) {
          const pid = t.getAttribute('data-participant-id') || t.getAttribute('data-tid') || t.id || '';
          if (pid && seenIds.has(pid) && pid !== 'roster-participant') continue;
          if (pid) seenIds.add(pid);
          const selfName = (t.getAttribute('data-self-name') || '').trim();
          const rawTxt = (t.innerText || t.textContent || '').trim();
          const aria = (t.getAttribute('aria-label') || '').trim();
          const combined = `${selfName} ${rawTxt} ${aria}`.trim();
          if (!combined) continue;
          if (botRe.test(combined)) continue;
          const lines = rawTxt.split(/\n+/).map(s => s.trim()).filter(Boolean);
          if (lines.length > 0 && lines.every(l => selfRe.test(l) || botRe.test(l))) continue;
          if (selfRe.test(selfName) || selfRe.test(rawTxt)) continue;
          humanOtherCount += 1;
        }
        return 1 + humanOtherCount;
      }
      return 2;
    })();
    """
_DENIED_PROBE_JS = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      return /You can't join this video call|You were removed from the meeting|No one responded to your request to join|Du kan inte gå med i det här videosamtalet|Du har tagits bort från mötet|Ingen svarade på din förfrågan|Samtalet har avslutats|Call ended|Mötet har avslutats|The meeting has ended|Vi kunde inte släppa in dig|Sorry, we couldn't connect you|Ingen släppte in dig|No one let you in/i.test(text);
    })();
    """


def _probe(page, js: str) -> bool:
    """Evaluate a boolean JS probe; conservative — False on any error."""
    return bool(_quiet(page.evaluate, js))


def _visible(locator):
    """``locator.first`` if it exists and is visible, else None (swallows Playwright errors)."""
    return _quiet(lambda: locator.first if locator.first.count() and locator.first.is_visible() else None)


def _pcm_tail_loop(proc, pcm_path: Path, stop_flag: dict, poll_interval: float = 0.05) -> None:
    """Follow ``speaker.pcm`` as it grows and forward every appended chunk to the pump's stdin.
    The pump itself would hit EOF on the (empty) file at start-up and exit; this thread keeps
    feeding it until the stop flag is set or the pump dies."""
    try:
        with open(pcm_path, "rb") as f:
            while not stop_flag.get("stop") and proc.poll() is None:
                chunk = f.read(65536)
                if not chunk:
                    time.sleep(poll_interval)
                    continue
                proc.stdin.write(chunk)
                proc.stdin.flush()
    except (OSError, ValueError):
        pass  # pump exited / pipe closed (BrokenPipeError, write on closed stdin): nothing left to stream to
    finally:
        _quiet(proc.stdin.close)


def _start_pcm_pump(rt: dict, bridge_info: dict, pcm_path: Path, state: _BotState,
                    stop_flag: dict) -> None:
    """Stream the growing ``speaker.pcm`` (24kHz s16le mono) into the device Chrome's fake mic reads.
    The pump reads raw PCM from stdin (``-``) so audio appended after start-up is still played —
    pointed at the file it would read the empty sink to EOF and exit before Realtime spoke."""
    bridge_info = bridge_info or {}
    platform_tag = bridge_info.get("platform")
    target = bridge_info.get("write_target")
    if platform_tag == "linux":
        cmd = ["paplay", "--raw", "--rate=24000", "--format=s16le", "--channels=1",
               f"--device={target or 'hermes_meet_sink'}", "-"]
        missing = "paplay not found — install pulseaudio-utils for realtime on Linux"
    elif platform_tag == "darwin":
        # User must have BlackHole as default input; ffmpeg targets it by audiotoolbox index.
        if not shutil.which("ffmpeg"):
            state.set(error=_FFMPEG_MISSING)
            return
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "-", "-f", "audiotoolbox",
               "-audio_device_index", _mac_audio_device_index(target or "BlackHole 2ch"), "-"]
        missing = _FFMPEG_MISSING
    else:
        return
    try:
        rt["pcm_pump"] = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        state.set(error=missing)
        return
    except Exception as e:
        if platform_tag != "darwin":
            raise
        state.set(error=f"macOS pcm pump failed to start: {e}")
        return
    rt["pcm_tail_thread"] = threading.Thread(
        target=_pcm_tail_loop, args=(rt["pcm_pump"], pcm_path, stop_flag), name="meet-pcm-tail", daemon=True)
    rt["pcm_tail_thread"].start()


def _start_realtime_speaker(rt: dict, cfg: _BotConfig, stop_flag: dict, state: _BotState) -> None:
    """Wire up the OpenAI Realtime session, the say-queue speaker thread and the PCM pump."""
    pcm_path, queue_path = cfg.out_dir / "speaker.pcm", cfg.out_dir / "say_queue.jsonl"
    pcm_path.write_bytes(b"")  # clean sink file per session
    queue_path.touch()  # so the speaker poller doesn't error on first iteration
    phase = "import"
    try:
        from plugins.google_meet.realtime.openai_client import RealtimeSession, RealtimeSpeaker
        phase = "connect"
        session = RealtimeSession(
            api_key=cfg.realtime_api_key, model=cfg.realtime_model, voice=cfg.realtime_voice,
            instructions=cfg.realtime_instructions, audio_sink_path=pcm_path, sample_rate=24000)
        session.connect()
    except Exception as e:
        state.set(error=f"realtime {phase} failed: {e}")
        return
    rt["session"] = session
    speaker = RealtimeSpeaker(session=session, queue_path=queue_path,
                              processed_path=cfg.out_dir / "say_processed.jsonl")

    def _speaker_loop():
        try:
            speaker.run_until_stopped(lambda: stop_flag.get("stop", False))
        except Exception as e:
            state.set(error=f"realtime speaker crashed: {e}")

    rt["speaker_thread"] = threading.Thread(target=_speaker_loop, name="meet-speaker", daemon=True)
    rt["speaker_thread"].start()
    _start_pcm_pump(rt, rt["bridge_info"], pcm_path, state, stop_flag)
    state.set(realtime_ready=True)


def _mac_audio_device_index(device_name: str) -> str:
    """ffmpeg ``-audio_device_index`` for *device_name* (case-insensitive; ``"0"`` if not found).
    ffmpeg prints the avfoundation device table on stderr as ``[N] Name``."""
    out = _quiet(subprocess.run, ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                 capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
    needle = device_name.strip().lower()
    for line in (out.stderr if out else "").splitlines():
        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if m and m.group(2).strip().lower() == needle:
            return m.group(1)
    return "0"


def _setup_realtime(rt: dict, api_key: str, state: _BotState) -> None:
    """Provision the virtual audio bridge; on any failure fall back to transcribe mode."""
    if not api_key:
        state.set(error="realtime mode requested but no API key in HERMES_MEET_REALTIME_KEY/OPENAI_API_KEY — falling back to transcribe")
        rt["enabled"] = False
        return
    try:
        from plugins.google_meet.audio_bridge import AudioBridge
        rt["bridge"] = AudioBridge()
        rt["bridge_info"] = rt["bridge"].setup()
        state.set(realtime=True, realtime_device=rt["bridge_info"].get("device_name"))
    except Exception as e:
        state.set(error=f"audio bridge setup failed: {e} — falling back to transcribe")
        rt["enabled"] = False


def _teardown_realtime(rt: dict) -> None:
    if rt.get("pcm_pump"):
        _quiet(rt["pcm_pump"].terminate)
        _quiet(rt["pcm_pump"].wait, timeout=3)
    for key, method, kw in (("pcm_tail_thread", "join", {"timeout": 1.0}),
                            ("speaker_thread", "join", {"timeout": 5.0}), ("session", "close", {}),
                            ("bridge", "teardown", {})):
        if rt.get(key) is not None:
            _quiet(getattr(rt[key], method), **kw)


_BotConfig = SimpleNamespace  # everything the bot reads from ``HERMES_MEET_*`` env vars


def _config_from_env() -> _BotConfig:
    env = os.environ.get
    out_raw = env("HERMES_MEET_OUT_DIR", "").strip()
    return _BotConfig(
        url=env("HERMES_MEET_URL", "").strip(),
        out_dir=Path(out_raw) if out_raw else None,
        headed=env("HERMES_MEET_HEADED", "").lower() in {"1", "true", "yes"},
        auth_state=env("HERMES_MEET_AUTH_STATE", "").strip(),
        guest_name=env("HERMES_MEET_GUEST_NAME", "Hermes Agent"),
        duration_s=_parse_duration(env("HERMES_MEET_DURATION", "")),
        realtime=env("HERMES_MEET_MODE", "transcribe").strip().lower() == "realtime",
        # HERMES_MEET_REALTIME_KEY is resolved by process_manager.start() via the parent's
        # profile secret scope; OPENAI_API_KEY only serves standalone `python -m` runs.
        realtime_api_key=env("HERMES_MEET_REALTIME_KEY") or env("OPENAI_API_KEY", ""),
        realtime_model=env("HERMES_MEET_REALTIME_MODEL", "gpt-realtime"),
        realtime_voice=env("HERMES_MEET_REALTIME_VOICE", "alloy"),
        # RTC audio capture + local transcription (the reliable path; captions are dead)
        rtc_audio=env("HERMES_MEET_RTC_AUDIO", "1").strip().lower() not in {"0", "false", "no"},
        whisper_model=env("HERMES_MEET_WHISPER_MODEL", "small"),
        realtime_instructions=env("HERMES_MEET_REALTIME_INSTRUCTIONS", ""),
        lobby_timeout=float(env("HERMES_MEET_LOBBY_TIMEOUT", "300")))


def _click_nowait(el, timeout: int = 2_000) -> bool:
    try:
        el.click(timeout=timeout, no_wait_after=True)
        return True
    except TypeError:
        el.click(timeout=timeout)
        return True


def _join(page, cfg: _BotConfig, state: _BotState, timeout: float = 60.0) -> None:
    """Fill the guest-name field and click 'Join now' / 'Ask to join' (the latter → lobby_waiting).

    Meet renders the pre-join buttons asynchronously after ``domcontentloaded``, so poll for up to
    *timeout* seconds instead of checking once — a single miss leaves the bot silently in the lobby.
    Blocking modals are dismissed first ("Switch the call here" when this account is already in
    the call from another device; misc one-time notices) — they intercept all clicks — and the
    fake camera is turned off (visually disruptive to participants). Late-rendered dialogs are
    caught by later poll iterations.
    """
    deadline = time.time() + timeout
    while True:
        for dismiss_sel in (
            'div[role="dialog"][aria-label*="Byt samtal" i] button',
            'div[role="dialog"][aria-label*="Switch" i] button',
            'button:has-text("Got it")',
            'button:has-text("Uppfattat")',
            'button:has-text("Jag förstår")',
        ):
            dlg = _visible(page.locator(dismiss_sel))
            if dlg is not None:
                _quiet(_click_nowait, dlg, 2_000)
                _quiet(page.wait_for_timeout, 1_500)
        # Microsoft Teams web launcher & pre-join screen handling
        page_url = str(getattr(page, "url", "") or "")
        cfg_url = str(getattr(cfg, "url", "") or "")
        if "teams.microsoft.com" in page_url or "teams.live.com" in page_url or "teams." in cfg_url:
            for web_sel in (
                'button[data-tid="joinOnWeb"]',
                'button:has-text("Continue on this browser")',
                'button:has-text("Fortsätt i den här webbläsaren")',
                'button:has-text("Join on the web instead")',
                'button:has-text("Anslut på webben i stället")',
            ):
                wbtn = _visible(page.locator(web_sel))
                if wbtn is not None:
                    _quiet(_click_nowait, wbtn, 2_000)
                    _quiet(page.wait_for_timeout, 2_000)
                    break
            t_name = _visible(page.locator('input[data-tid="prejoin-display-name-input"]'))
            if t_name is not None:
                teams_guest = re.sub(r"&", " och ", cfg.guest_name or "Ur och Penn Mötesanteckningar")
                teams_guest = re.sub(r"[^\w\s\-'._@]", "", teams_guest, flags=re.UNICODE)
                teams_guest = re.sub(r"\s+", " ", teams_guest).strip() or "Ur och Penn Motesanteckningar"
                _quiet(t_name.click, timeout=2_000)
                _quiet(t_name.fill, teams_guest, timeout=2_000)
                _quiet(page.wait_for_timeout, 400)
            for t_toggle in ('input[data-tid="toggle-video"]', 'input[data-tid="toggle-mute"]'):
                tog = _visible(page.locator(t_toggle))
                if tog is not None and _quiet(tog.is_checked):
                    _quiet(_click_nowait, tog, 2_000)
            t_join = _visible(page.locator('button[data-tid="prejoin-join-button"]'))
            if t_join is not None and _quiet(t_join.is_enabled) and _quiet(_click_nowait, t_join, 3_000):
                state.set(lobby_waiting=True)
                return
            if time.time() >= deadline:
                return
            time.sleep(0.5)
            continue

        # Camera and microphone off on pre-join screen (EN + SV).
        for cam_label in ("Turn off camera", "Slå av kameran", "Stäng av kameran", "Slå av kamera"):
            cam_off = _visible(page.get_by_role("button", name=cam_label, exact=False))
            if cam_off is not None:
                _quiet(_click_nowait, cam_off, 2_000)
                break
        if getattr(cfg, "rtc_audio", False) and not getattr(cfg, "realtime", False):
            for mic_label in ("Turn off microphone", "Slå av mikrofonen", "Stäng av mikrofonen", "Slå av mikrofon", "Stäng av mikrofon"):
                mic_off = _visible(page.get_by_role("button", name=mic_label, exact=False))
                if mic_off is not None:
                    _quiet(_click_nowait, mic_off, 2_000)
                    break
        name_box = _visible(page.locator('input[aria-label*="name" i], input[aria-label*="namn" i]'))
        if name_box is not None:
            _quiet(name_box.fill, cfg.guest_name, timeout=2_000)
        for label in ("Join now", "Gå med nu", "Ask to join", "Be om att få gå med", "Join here too", "Gå med här också"):
            btn = _visible(page.get_by_role("button", name=label, exact=False))
            if btn is None:
                btn = _visible(page.locator(f'button:has-text("{label}")'))
            if btn is None and label in ("Join here too", "Gå med här också"):
                for acc_txt in ("Other ways to join", "Andra sätt att gå med"):
                    accordion = _visible(page.locator(f'text={acc_txt}'))
                    if accordion is not None:
                        _quiet(_click_nowait, accordion, 2_000)
                        time.sleep(1.5)
                        btn = _visible(page.locator(f'button:has-text("{label}")'))
                        break
            if btn is not None and _quiet(_click_nowait, btn, 3_000):
                if label in ("Ask to join", "Be om att få gå med"):
                    state.set(lobby_waiting=True)
                _quiet(getattr(page, "wait_for_timeout", None), 2_000)
                for consent_sel in (
                    'div[role="dialog"] button:has-text("Gå med nu")',
                    'div[role="dialog"] button:has-text("Join now")',
                    'div[role="dialog"] button:has-text("Jag förstår")',
                    'div[role="dialog"] button:has-text("Uppfattat")',
                    'div[role="dialog"] button:has-text("Got it")',
                ):
                    cbtn = _visible(page.locator(consent_sel))
                    if cbtn is not None:
                        _quiet(_click_nowait, cbtn, 2_000)
                        _quiet(getattr(page, "wait_for_timeout", None), 1_000)
                return
        if time.time() >= deadline:
            return
        time.sleep(0.5)


def _ensure_mic_on(page) -> str:
    """Unmute the bot once admitted — Meet may seat an authenticated bot muted, and a muted mic makes
    realtime speech inaudible. The in-call toggle's aria-label is "Turn on microphone" while muted /
    "Turn off microphone" while live; returns ``unmuted_clicked`` / ``unmuted`` / ``unknown``
    (toggle not found: Meet variant changed or the label is localized)."""
    muted = _visible(page.locator('button[aria-label*="Turn on microphone" i], button[aria-label*="Slå på mikrofon" i], button[aria-label*="Aktivera mikrofon" i]'))
    if muted is not None and _quiet(lambda: (muted.click(timeout=3_000), True)):
        return "unmuted_clicked"
    if _visible(page.locator('button[aria-label*="Turn off microphone" i], button[aria-label*="Slå av mikrofon" i], button[aria-label*="Inaktivera mikrofon" i], button[aria-label*="Stäng av mikrofon" i]')) is not None:
        return "unmuted"
    return "unknown"


def _ensure_mic_muted(page) -> str:
    """RTC-capture mirror of ``_ensure_mic_on``: a transcribing bot must NEVER broadcast
    room audio from a real mic — mute on admission. Returns ``muted_clicked``/``muted``/``unknown``."""
    live = _visible(page.locator('button[aria-label*="Turn off microphone" i], button[aria-label*="Slå av mikrofon" i], button[aria-label*="Inaktivera mikrofon" i], button[aria-label*="Stäng av mikrofon" i]'))
    if live is not None and _quiet(lambda: (live.click(timeout=3_000, no_wait_after=True), True)):
        return "muted_clicked"
    if _visible(page.locator('button[aria-label*="Turn on microphone" i], button[aria-label*="Slå på mikrofon" i], button[aria-label*="Aktivera mikrofon" i]')) is not None:
        return "muted"
    return "unknown"


def _drain_loop(page, cfg: _BotConfig, state: _BotState, rt: dict, stop_flag: dict) -> None:
    """Admission + caption drain loop until SIGTERM, duration expiry, lobby timeout/denial or page loss.
    Sets ``leave_reason`` for every exit but SIGTERM; triggers barge-in; mirrors realtime counters."""
    deadline = (time.time() + cfg.duration_s) if cfg.duration_s else None
    lobby_deadline = time.time() + cfg.lobby_timeout
    last_admission_check = 0.0
    had_other_participants = False
    alone_since = None
    while not stop_flag["stop"]:
        now = time.time()
        if deadline and now > deadline:
            joined_ts = getattr(state, "joined_at", None)
            last_cap_ts = getattr(state, "last_caption_at", None)
            is_in_call = bool(getattr(state, "in_call", False))
            in_call_elapsed = (now - float(joined_ts)) if joined_ts else 0.0
            recent_speech = bool(last_cap_ts and (now - float(last_cap_ts)) < 600.0)
            still_populated = bool(is_in_call and alone_since is None and in_call_elapsed < 14400.0)
            if is_in_call and in_call_elapsed < 21600.0 and (recent_speech or still_populated):
                deadline = now + 900.0
                print(
                    f"[meet_bot] overtime_extend new_deadline=+900s elapsed_s={in_call_elapsed:.1f} recent_speech={recent_speech}",
                    flush=True,
                )
            else:
                state.set(leave_reason="duration_expired")
                return
        if not state.in_call and (now - last_admission_check) > 3.0:
            last_admission_check = now
            if _probe(page, _ADMISSION_PROBE_JS):
                state.set(in_call=True, lobby_waiting=False, joined_at=now,
                          mic_state=(_ensure_mic_muted(page) if (getattr(cfg, "rtc_audio", False) and not rt.get("enabled"))
                                     else _ensure_mic_on(page)))
            elif _visible(page.locator('button[data-tid="prejoin-join-button"], button[data-tid="joinOnWeb"]')) is not None:
                _join(page, cfg, state, timeout=5.0)
            elif now > lobby_deadline:
                waited = int(lobby_deadline - state.join_attempted_at) if state.join_attempted_at else 0
                state.set(error=f"lobby timeout — host never admitted the bot within {waited}s",
                          leave_reason="lobby_timeout")
                return
            elif _probe(page, _DENIED_PROBE_JS):
                state.set(error="host denied admission", leave_reason="denied")
                return
        elif state.in_call and (now - last_admission_check) > 5.0:
            last_admission_check = now
            if state.mic_state == "unknown" and getattr(cfg, "rtc_audio", False) and not rt.get("enabled"):
                m_st = _ensure_mic_muted(page)
                if m_st != "unknown":
                    state.set(mic_state=m_st)
            if _probe(page, _DENIED_PROBE_JS):
                state.set(leave_reason="kicked_or_ended")
                return
            pcount = _quiet(page.evaluate, _PARTICIPANT_COUNT_JS) or 2
            if isinstance(pcount, int) and pcount >= 2:
                had_other_participants = True
                alone_since = None
            elif isinstance(pcount, int) and pcount <= 1:
                recent_audio = bool(state.last_caption_at and (now - float(state.last_caption_at)) < 45.0)
                if recent_audio:
                    alone_since = None
                elif had_other_participants:
                    if alone_since is None:
                        alone_since = now
                    elif (now - alone_since) >= 12.0:
                        print(f"[meet_bot] leave_reason=everyone_left alone_s={now - alone_since:.1f}", flush=True)
                        state.set(leave_reason="everyone_left")
                        return
                elif state.joined_at and (now - state.joined_at) >= 300.0:
                    reason = "empty_room_timeout" if state.transcript_lines == 0 else "everyone_left"
                    print(f"[meet_bot] leave_reason={reason} empty_from_start_s={now - state.joined_at:.1f}", flush=True)
                    state.set(leave_reason=reason)
                    return
        try:
            spk_events = page.evaluate("window.__hermesDrainSpeakers && window.__hermesDrainSpeakers()")
            for sev in (e for e in (spk_events if isinstance(spk_events, list) else ()) if isinstance(e, dict)):
                state.record_dom_speaker(str(sev.get("speaker", "")), sev.get("ts"))
            queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
            for entry in (e for e in (queued if isinstance(queued, list) else ()) if isinstance(e, dict)):
                speaker = str(entry.get("speaker", ""))
                state.record_dom_speaker(speaker, now)
                if not cfg.rtc_audio:
                    state.record_caption(speaker=speaker, text=str(entry.get("text", "")))
                # Barge-in: a real human spoke while we may be generating audio.
                if (rt["session"] is not None and _looks_like_human_speaker(speaker, cfg.guest_name)
                        and _quiet(rt["session"].cancel_response)):
                    state.set(last_barge_in_at=now)
        except Exception:
            if page.is_closed():  # Meet reloaded or we got booted — exit rather than spin
                state.set(leave_reason="page_closed")
                return
        if rt["session"] is not None:
            state.set(audio_bytes_out=rt["session"].audio_bytes_out,
                      last_audio_out_at=rt["session"].last_audio_out_at)
        time.sleep(1.0)


_CONTEXT_ARGS = {
    "viewport": {"width": 1280, "height": 800},
    "user_agent": os.environ.get(
        "HERMES_MEET_USER_AGENT",
        # Default keeps the historical Linux UA (verified working 2026-09-08); the RTC audio
        # path, however, needs a Mac UA — Meet serves Linux clients digital-silence-only
        # audio in some builds (A/B receipt 2026-09-21: Mac UA captured speech, Linux UA
        # peak 0.0000 across 4 live runs). process_manager.start() sets the Mac UA when
        # rtc audio is enabled.
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"),
    "permissions": ["microphone", "camera"]}


def run_bot() -> int:
    cfg = _config_from_env()
    if not _is_safe_meet_url(cfg.url):
        sys.stderr.write("google_meet bot: refusing to launch — HERMES_MEET_URL must be a "
                         f"meet.google.com URL. got: {cfg.url!r}\n")
        return 2
    if cfg.out_dir is None:
        sys.stderr.write("google_meet bot: HERMES_MEET_OUT_DIR is required\n")
        return 2
    state = _BotState(out_dir=cfg.out_dir, meeting_id=_meeting_id_from_url(cfg.url), url=cfg.url)
    # SIGTERM sets a flag (not an exception) so the Playwright teardown below still runs
    # and ``meet_leave`` gets a finalized transcript.
    stop_flag = {"stop": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _sig, _frame: stop_flag.__setitem__("stop", True))
    # Realtime resources in one dict so teardown works however we exit.
    rt = dict(enabled=cfg.realtime, bridge=None, bridge_info=None, session=None, speaker_thread=None)
    if rt["enabled"]:
        _setup_realtime(rt, cfg.realtime_api_key, state)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        state.set(error=f"playwright not installed: {e}", exited=True)
        sys.stderr.write("google_meet bot: playwright is not installed. Run "
                         "`hermes meet install`\n")
        if rt["bridge"]:
            rt["bridge"].teardown()
        return 3
    chrome_args = ["--use-fake-ui-for-media-stream", "--disable-blink-features=AutomationControlled"]
    # The fake silent mic breaks remote audio reception in some Meet builds (A/B receipt
    # 2026-09-21: spike with real-device grant captured speech, fake-device bot got digital
    # silence) — HERMES_MEET_NO_FAKE_DEVICE=1 opts into the real media stack.
    _no_fake = os.environ.get("HERMES_MEET_NO_FAKE_DEVICE", "").lower() in {"1", "true", "yes"}
    if not rt["enabled"] and not _no_fake:
        chrome_args.insert(1, "--use-fake-device-for-media-stream")  # silent fake mic
    elif rt["bridge_info"] and rt["bridge_info"].get("platform") == "linux":
        # Playwright's launch() takes no env: set PULSE_SOURCE on ourselves so Chrome inherits it.
        os.environ["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")
    context_args = dict(_CONTEXT_ARGS)
    if cfg.auth_state and Path(cfg.auth_state).is_file():
        context_args["storage_state"] = cfg.auth_state
    # RTC capture plumbing: init script must precede any page load, the binding before
    # the first MediaRecorder chunk (~10 s after join). Fragments land in audio/ and are
    # transcribed by the _RtcTranscriber workers.
    rtc = {"transcriber": None, "audio_dir": None}
    if cfg.rtc_audio:
        rtc["audio_dir"] = cfg.out_dir / "audio"
        rtc["audio_dir"].mkdir(parents=True, exist_ok=True)
        rtc["transcriber"] = _RtcTranscriber(cfg, state, stop_flag)
        state.set(whisper_model=cfg.whisper_model)
        frag_seq = itertools.count()
        transcriber_ref = rtc["transcriber"]
        audio_dir_ref = rtc["audio_dir"]

        def _on_audio_chunk(stream_id, b64):
            try:
                frag = audio_dir_ref / f"frag_{int(stream_id):02d}_{next(frag_seq):06d}.webm"
                frag.write_bytes(base64.b64decode(b64))
                if not state.rtc_capturing:
                    state.set(rtc_capturing=True)
                transcriber_ref.submit(int(stream_id), frag)
            except Exception as e:  # never kill the binding thread
                sys.stderr.write(f"meet rtc: chunk save failed: {e}\n")

    try:
        with sync_playwright() as pw:
            launch_kwargs = dict(headless=not cfg.headed, args=chrome_args)
            if cfg.headed:
                chrome_args.append("--window-position=-2000,-2000")
            if cfg.rtc_audio:
                # headless Chromium ships --mute-audio by default — remote streams record
                # as digital silence without dropping it.
                launch_kwargs["ignore_default_args"] = ["--mute-audio"]
                chrome_args.append("--autoplay-policy=no-user-gesture-required")
                launch_kwargs["args"] = chrome_args
            browser = pw.chromium.launch(
                channel="chromium", executable_path=chromium_executable(), **launch_kwargs)
            context = browser.new_context(**context_args)
            if cfg.rtc_audio:
                chunk_ms = int(os.environ.get("HERMES_MEET_CHUNK_MS", "10000") or 10000)
                if chunk_ms != 10000:
                    context.add_init_script(f"window.__hermesMeetChunkMs = {chunk_ms};")
                context.add_init_script(_RTC_HOOK_JS)
            page = context.new_page()
            if cfg.rtc_audio:
                page.expose_function("__hermesMeetAudioChunk", _on_audio_chunk)
            try:
                page.goto(cfg.url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4
            _join(page, cfg, state)
            if _quiet(page.evaluate, _ENABLE_CAPTIONS_JS):
                state.set(captions_enabled_attempted=True)
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception as e:
                state.set(error=f"caption observer install failed: {e}")
            # in_call stays False until admission is confirmed by the drain loop.
            state.set(captioning=True, join_attempted_at=time.time())
            if rt["enabled"]:
                _start_realtime_speaker(rt, cfg, stop_flag, state)
            _drain_loop(page, cfg, state, rt, stop_flag)
            # Stop the recorders so MediaRecorder flushes a final fragment, hang up and close
            # the browser IMMEDIATELY so the bot leaves the call right away, then drain workers.
            if cfg.rtc_audio:
                _quiet(page.evaluate, "window.__hermesMeetRtcStop && window.__hermesMeetRtcStop()")
                _quiet(page.wait_for_timeout, 1500)
            _quiet(page.evaluate, _LEAVE_CALL_JS)
            if cfg.auth_state:
                _quiet(context.storage_state, path=cfg.auth_state)
            _quiet(context.close)
            _quiet(browser.close)
            if cfg.rtc_audio:
                stop_flag["stop"] = True
                for t in rtc["transcriber"].threads:
                    _quiet(t.join, timeout=45.0)
            _teardown_realtime(rt)
            state.set(in_call=False, captioning=False, rtc_capturing=False, exited=True)
            return 0
    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True)
        return 1


def _looks_like_human_speaker(speaker: str, bot_guest_name: str) -> bool:
    """Whether a caption's speaker is probably a human rather than our own echo (Meet attributes
    our fake-mic audio to the bot's name; blank/unknown speakers are ambiguous — no barge-in)."""
    return bool(speaker and speaker.strip()) and (
        speaker.strip().lower() not in {"unknown", "you", bot_guest_name.strip().lower()})


_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def _parse_duration(raw: str) -> Optional[float]:
    """Parse ``30m`` / ``2h`` / ``90`` (seconds) → float seconds, or None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    mult = _DURATION_UNITS.get(raw[-1:])
    try:
        return float(raw[:-1]) * mult if mult else float(raw)
    except ValueError:
        return None


if __name__ == "__main__":  # pragma: no cover — subprocess entry point
    sys.exit(run_bot())
