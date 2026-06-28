"""Route AI Partner OS voice output through Hermes VOICEVOX / irodoriTTS."""

from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SUPPORTED_PROVIDERS = ("auto", "irodori", "voicevox", "none")


def _irodori_status() -> dict[str, Any]:
    try:
        from plugins.irodori_tts import core as irodori_core
    except Exception as exc:
        return {"ok": False, "provider": "irodori", "available": False, "error": str(exc)}
    payload = irodori_core.status_payload()
    server = payload.get("server") if isinstance(payload, dict) else {}
    server_ok = isinstance(server, dict) and server.get("ok") is True
    return {
        **payload,
        "provider": "irodori",
        "usable": bool(payload.get("available") and server_ok),
    }


def _voicevox_status() -> dict[str, Any]:
    try:
        from plugins.voicevox_tts import core as voicevox_core
    except Exception as exc:
        return {"ok": False, "provider": "voicevox", "available": False, "error": str(exc)}
    payload = voicevox_core.status_payload()
    return {
        **payload,
        "provider": "voicevox",
        "usable": bool(payload.get("available")),
    }


def select_provider(explicit: str | None = None) -> str:
    requested = (explicit or "auto").strip().lower()
    if requested in SUPPORTED_PROVIDERS and requested != "auto":
        return requested
    irodori = _irodori_status()
    if irodori.get("usable"):
        return "irodori"
    voicevox = _voicevox_status()
    if voicevox.get("usable"):
        return "voicevox"
    return "none"


def tts_status(requested: str | None = None) -> dict[str, Any]:
    irodori = _irodori_status()
    voicevox = _voicevox_status()
    selected = select_provider(requested)
    ready = (selected == "irodori" and bool(irodori.get("usable"))) or (
        selected == "voicevox" and bool(voicevox.get("usable"))
    )
    return {
        "ok": ready,
        "requested_provider": requested or "auto",
        "selected_provider": selected,
        "ready": ready,
        "providers": {"irodori": irodori, "voicevox": voicevox},
        "note": "AI Partner OS uses Hermes TTS via play_tts_on_pc(data URL), not in-app VOICEVOX.",
    }


def start_backend(provider: str | None = None, *, timeout_seconds: int = 120) -> dict[str, Any]:
    selected = select_provider(provider)
    if selected == "irodori":
        try:
            from plugins.irodori_tts import core as irodori_core
        except Exception as exc:
            return {"ok": False, "provider": "irodori", "error": str(exc)}
        before = _irodori_status()
        if before.get("usable"):
            return {"ok": True, "provider": "irodori", "already_running": True, "status": before}
        cfg = irodori_core.settings()
        ps = irodori_core.powershell_path()
        if not ps or not cfg.start_script.is_file():
            return {"ok": False, "provider": "irodori", "error": "irodori start script unavailable"}
        import subprocess

        proc = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(cfg.start_script), "-RepoDir", str(cfg.repo_dir)],
            cwd=str(cfg.repo_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        after = _irodori_status()
        return {
            "ok": proc.returncode == 0 and bool(after.get("usable")),
            "provider": "irodori",
            "returncode": proc.returncode,
            "status": after,
        }
    if selected == "voicevox":
        return {
            "ok": bool(_voicevox_status().get("usable")),
            "provider": "voicevox",
            "status": _voicevox_status(),
            "hint": "Start VOICEVOX Engine on http://127.0.0.1:50021 or configure plugins.voicevox_tts.",
        }
    return {"ok": False, "provider": selected, "error": "No Hermes TTS backend is ready."}


def synthesize(
    text: str,
    *,
    provider: str | None = None,
    voice: str | int | None = None,
    speed: float | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("text must not be empty")
    selected = select_provider(provider)
    if selected == "irodori":
        from plugins.irodori_tts import core as irodori_core

        return irodori_core.synthesize_text(
            clean,
            output_path=output_path,
            voice=str(voice) if voice is not None else None,
            speed=speed,
        )
    if selected == "voicevox":
        from plugins.voicevox_tts import core as voicevox_core

        return voicevox_core.synthesize_text(
            clean,
            output_path=output_path,
            voice=voice,
            speed=speed,
        )
    raise RuntimeError("No Hermes TTS backend available (enable irodori_tts or voicevox_tts).")


def file_to_data_url(path: str | Path) -> str:
    file_path = Path(path)
    raw = file_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "audio/wav" if file_path.suffix.lower() == ".wav" else "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def synthesize_data_url(
    text: str,
    *,
    provider: str | None = None,
    voice: str | int | None = None,
    speed: float | None = None,
) -> dict[str, Any]:
    result = synthesize(text, provider=provider, voice=voice, speed=speed)
    file_path = result.get("file_path")
    if not file_path:
        raise RuntimeError("TTS synthesis did not return file_path")
    data_url = file_to_data_url(file_path)
    return {**result, "data_url": data_url, "data_url_bytes": len(data_url)}


def play_audio_local(path: str | Path, *, blocking: bool = False) -> dict[str, Any]:
    """Play synthesized audio on this PC when Eel play_tts_on_pc is unavailable."""
    file_path = Path(path)
    if not file_path.is_file():
        return {"ok": False, "error": f"audio file not found: {file_path}"}

    if sys.platform == "win32":
        suffix = file_path.suffix.lower()
        if suffix == ".wav":
            try:
                import winsound

                flags = winsound.SND_FILENAME
                if not blocking:
                    flags |= winsound.SND_ASYNC
                winsound.PlaySound(str(file_path), flags)
                return {"ok": True, "via": "winsound", "file_path": str(file_path), "async": not blocking}
            except Exception as exc:
                return {"ok": False, "error": str(exc), "file_path": str(file_path)}

        ps = (
            f"$p = New-Object System.Media.SoundPlayer('{file_path}'); "
            + ("$p.PlaySync()" if blocking else "$p.Play()")
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120 if blocking else 15,
            )
            if proc.returncode == 0:
                return {"ok": True, "via": "powershell_soundplayer", "file_path": str(file_path)}
            return {
                "ok": False,
                "error": (proc.stderr or proc.stdout or "powershell playback failed").strip(),
                "file_path": str(file_path),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc), "file_path": str(file_path)}

    for player in ("ffplay", "mpv", "aplay", "paplay"):
        if not _which(player):
            continue
        cmd = [player, str(file_path)]
        if player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(file_path)]
        try:
            if blocking:
                subprocess.run(cmd, check=False, timeout=120, stdin=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=os.name != "nt")
            return {"ok": True, "via": player, "file_path": str(file_path), "async": not blocking}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc), "file_path": str(file_path)}

    return {"ok": False, "error": "No local audio player available", "file_path": str(file_path)}


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)
