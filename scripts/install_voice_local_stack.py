#!/usr/bin/env python3
"""Install local voice-native Hermes wiring for WhatsApp voice and calling.

The script writes the two Linux user-service pieces that are otherwise easy to
misconfigure by hand:

1. A ``voice-webrtc-sidecar.service`` user unit for the voice repo sidecar.
2. A ``hermes-gateway.service`` drop-in that points Hermes at the checkout with
   the WhatsApp voice/calling stack and exposes the sidecar stream command.

By default the script is a dry run and prints the files it would write. Pass
``--apply`` to write them, reload systemd, and optionally restart Hermes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_HERMES_SERVICE = "hermes-gateway.service"
DEFAULT_SIDECAR_SERVICE = "voice-webrtc-sidecar.service"
DEFAULT_STREAM_TIMEOUT = 180.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return os.path.abspath(os.path.expanduser(value))

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def default_voice_repo() -> str:
    configured = os.environ.get("VOICE_REPO")
    if configured:
        return configured
    return str(Path.home() / "code" / "src" / "github.com" / "rgbkrk" / "voice")


def default_webrtc_python() -> str:
    return os.environ.get("VOICE_WEBRTC_PYTHON", "python3")


def parse_sidecar_url(value: str) -> tuple[str, int]:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise SystemExit(f"sidecar URL must be http(s): {value!r}")
    if not parsed.hostname:
        raise SystemExit(f"sidecar URL must include a host: {value!r}")
    if parsed.path not in {"", "/"}:
        raise SystemExit(f"sidecar URL must not include a path: {value!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


def systemd_env_line(key: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'Environment="{key}={escaped}"'


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def default_gateway_path(
    *,
    hermes_home: Path,
    voice_bin: str,
    bridge_bin_dir: Path | None,
) -> str:
    home = Path.home()
    parts = [
        str(hermes_home / "hermes-agent" / "venv" / "bin"),
        str(bridge_bin_dir) if bridge_bin_dir else "",
    ]
    if "/" in voice_bin:
        parts.append(str(Path(voice_bin).resolve().parent))
    parts.extend(
        [
            str(home / ".local" / "bin"),
            str(home / ".cargo" / "bin"),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
    )
    return os.pathsep.join(dedupe_preserve_order(parts))


def render_sidecar_unit(
    *,
    voice_repo: Path,
    voice_bin: str,
    webrtc_python_bin: str,
    host: str,
    port: int,
    rx_pcm_path: Path,
    log_level: str,
) -> str:
    sidecar = voice_repo / "examples" / "webrtc-sidecar" / "sidecar.py"
    command = [
        webrtc_python_bin,
        str(sidecar),
        "--host",
        host,
        "--port",
        str(port),
        "--rx-pcm",
        str(rx_pcm_path),
        "--log-level",
        log_level,
    ]
    return "\n".join(
        [
            "[Unit]",
            "Description=Voice WebRTC Sidecar",
            "After=network.target voice-daemon.service",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={voice_repo}",
            systemd_env_line("VOICE_BIN", voice_bin),
            "ExecStart=" + " ".join(command),
            "Restart=on-failure",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def render_hermes_dropin(
    *,
    live_hermes_root: Path,
    gateway_path: str,
    sidecar_url: str,
    voice_bin: str,
    voice: str,
    speed: str,
    stream_timeout: float,
) -> str:
    stream_command = (
        f"{voice_bin} stream --quiet --sample-rate {{sample_rate}} "
        f"--frame-ms {{frame_ms}} --raw-output - --input-file {{input_path}} "
        f"--voice {voice} --speed {speed}"
    )
    return "\n".join(
        [
            "[Service]",
            systemd_env_line("PYTHONPATH", str(live_hermes_root)),
            systemd_env_line("PATH", gateway_path),
            systemd_env_line("WHATSAPP_CLOUD_CALLING_SIDECAR_URL", sidecar_url),
            systemd_env_line(
                "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND",
                stream_command,
            ),
            systemd_env_line(
                "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT",
                str(stream_timeout),
            ),
            "",
        ]
    )


def build_tts_provider(
    *,
    voice_bin: str,
    voice: str,
    speed: str,
    timeout: int,
    max_text_length: int,
) -> dict[str, Any]:
    return {
        "type": "command",
        "command": (
            f"{voice_bin} say --format ogg-opus --input-file {{input_path}} "
            f"--output {{output_path}} --voice {{voice}} --speed {{speed}}"
        ),
        "output_format": "ogg",
        "voice_compatible": True,
        "voice": voice,
        "speed": float(speed),
        "timeout": timeout,
        "max_text_length": max_text_length,
    }


def configure_tts_provider(
    *,
    config_path: Path,
    provider_name: str,
    provider: dict[str, Any],
) -> dict[str, Any]:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle) or {}
    else:
        data = {}

    if not isinstance(data, dict):
        raise SystemExit(f"Hermes config root must be a mapping: {config_path}")

    tts = data.setdefault("tts", {})
    if not isinstance(tts, dict):
        raise SystemExit(f"Hermes config tts section must be a mapping: {config_path}")
    providers = tts.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise SystemExit(
            f"Hermes config tts.providers section must be a mapping: {config_path}"
        )

    tts["provider"] = provider_name
    providers[provider_name] = provider
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)

    return {
        "path": str(config_path),
        "provider": provider_name,
        "output_format": provider["output_format"],
        "voice_compatible": provider["voice_compatible"],
    }


def run_systemctl(command: list[str], *, timeout: float) -> None:
    completed = subprocess.run(
        ["systemctl", "--user", *command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            "systemctl --user "
            + " ".join(command)
            + f" failed with exit code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    sidecar_host, sidecar_port = parse_sidecar_url(args.sidecar_url)
    if args.host and args.host != sidecar_host:
        raise SystemExit(
            f"--host {args.host!r} does not match --sidecar-url host {sidecar_host!r}"
        )
    if args.port and args.port != sidecar_port:
        raise SystemExit(
            f"--port {args.port!r} does not match --sidecar-url port {sidecar_port!r}"
        )

    hermes_home = Path(args.hermes_home).expanduser().resolve()
    live_hermes_root = Path(args.live_hermes_root).expanduser().resolve()
    voice_repo = Path(args.voice_repo).expanduser().resolve()
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    webrtc_python_bin = resolve_executable(
        args.webrtc_python_bin,
        label="WebRTC sidecar Python",
    )
    bridge_bin_dir = (
        Path(args.bridge_bin_dir).expanduser().resolve()
        if args.bridge_bin_dir
        else live_hermes_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    )
    gateway_path = args.gateway_path or default_gateway_path(
        hermes_home=hermes_home,
        voice_bin=voice_bin,
        bridge_bin_dir=bridge_bin_dir,
    )
    systemd_user_dir = Path(args.systemd_user_dir).expanduser().resolve()
    rx_pcm_path = (
        Path(args.rx_pcm_path).expanduser().resolve()
        if args.rx_pcm_path
        else hermes_home / "voice-webrtc-sidecar" / "inbound.s16le"
    )
    hermes_dropin_dir = systemd_user_dir / f"{args.hermes_service}.d"
    hermes_dropin_path = hermes_dropin_dir / "voice-stack.conf"
    sidecar_unit_path = systemd_user_dir / args.sidecar_service
    config_path = Path(args.config_path).expanduser().resolve()
    tts_provider = build_tts_provider(
        voice_bin=voice_bin,
        voice=args.voice,
        speed=args.speed,
        timeout=args.tts_timeout,
        max_text_length=args.max_text_length,
    )

    files: list[dict[str, str]] = []
    if not args.skip_sidecar_service:
        files.append(
            {
                "path": str(sidecar_unit_path),
                "kind": "systemd_user_service",
                "content": render_sidecar_unit(
                    voice_repo=voice_repo,
                    voice_bin=voice_bin,
                    webrtc_python_bin=webrtc_python_bin,
                    host=sidecar_host,
                    port=sidecar_port,
                    rx_pcm_path=rx_pcm_path,
                    log_level=args.log_level,
                ),
            }
        )
    if not args.skip_hermes_dropin:
        files.append(
            {
                "path": str(hermes_dropin_path),
                "kind": "systemd_user_dropin",
                "content": render_hermes_dropin(
                    live_hermes_root=live_hermes_root,
                    gateway_path=gateway_path,
                    sidecar_url=args.sidecar_url.rstrip("/"),
                    voice_bin=voice_bin,
                    voice=args.voice,
                    speed=args.speed,
                    stream_timeout=args.stream_timeout,
                ),
            }
        )

    return {
        "apply": args.apply,
        "configure_tts": args.configure_tts,
        "hermes_home": str(hermes_home),
        "live_hermes_root": str(live_hermes_root),
        "voice_repo": str(voice_repo),
        "voice_bin": voice_bin,
        "webrtc_python_bin": webrtc_python_bin,
        "sidecar_url": args.sidecar_url.rstrip("/"),
        "hermes_service": args.hermes_service,
        "sidecar_service": args.sidecar_service,
        "paths": {
            "systemd_user_dir": str(systemd_user_dir),
            "hermes_dropin": str(hermes_dropin_path),
            "sidecar_unit": str(sidecar_unit_path),
            "rx_pcm": str(rx_pcm_path),
            "config": str(config_path),
        },
        "files": files,
        "tts_provider": tts_provider,
    }


def apply_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    written: list[str] = []
    for file_plan in plan["files"]:
        path = Path(file_plan["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_plan["content"], encoding="utf-8")
        written.append(str(path))

    Path(plan["paths"]["rx_pcm"]).parent.mkdir(parents=True, exist_ok=True)
    config_result = None
    if plan["configure_tts"]:
        config_result = configure_tts_provider(
            config_path=Path(plan["paths"]["config"]),
            provider_name=args.provider_name,
            provider=plan["tts_provider"],
        )

    systemctl_actions: list[str] = []
    if not args.no_systemctl:
        run_systemctl(["daemon-reload"], timeout=args.systemctl_timeout)
        systemctl_actions.append("daemon-reload")
        if not args.skip_sidecar_service and not args.no_start:
            run_systemctl(
                ["enable", "--now", plan["sidecar_service"]],
                timeout=args.systemctl_timeout,
            )
            systemctl_actions.append(f"enable --now {plan['sidecar_service']}")
        if args.restart_hermes:
            run_systemctl(
                ["restart", plan["hermes_service"]],
                timeout=args.systemctl_timeout,
            )
            systemctl_actions.append(f"restart {plan['hermes_service']}")

    return {
        "written": written,
        "configured_tts": config_result,
        "systemctl": systemctl_actions,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write files")
    parser.add_argument(
        "--configure-tts",
        action="store_true",
        help="update config.yaml to use voice say --format ogg-opus",
    )
    parser.add_argument(
        "--restart-hermes",
        action="store_true",
        help="restart the Hermes gateway service after writing the drop-in",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="write the sidecar unit without enabling or starting it",
    )
    parser.add_argument(
        "--no-systemctl",
        action="store_true",
        help="write files without invoking systemctl --user",
    )
    parser.add_argument(
        "--skip-sidecar-service",
        action="store_true",
        help="do not write the voice WebRTC sidecar service",
    )
    parser.add_argument(
        "--skip-hermes-dropin",
        action="store_true",
        help="do not write the Hermes gateway drop-in",
    )
    parser.add_argument("--hermes-home", default=str(Path.home() / ".hermes"))
    parser.add_argument(
        "--live-hermes-root",
        default=str(repo_root()),
        help="Hermes checkout imported by the live gateway",
    )
    parser.add_argument("--voice-repo", default=default_voice_repo())
    parser.add_argument("--voice-bin", default="voice")
    parser.add_argument("--webrtc-python-bin", default=default_webrtc_python())
    parser.add_argument(
        "--sidecar-url",
        default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
        help="loopback base URL exposed by the voice WebRTC sidecar",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--rx-pcm-path")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--hermes-service", default=DEFAULT_HERMES_SERVICE)
    parser.add_argument("--sidecar-service", default=DEFAULT_SIDECAR_SERVICE)
    parser.add_argument(
        "--systemd-user-dir",
        default=str(Path.home() / ".config" / "systemd" / "user"),
    )
    parser.add_argument("--bridge-bin-dir")
    parser.add_argument("--gateway-path")
    parser.add_argument(
        "--config-path",
        default=str(Path.home() / ".hermes" / "config.yaml"),
    )
    parser.add_argument("--provider-name", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--stream-timeout", type=float, default=DEFAULT_STREAM_TIMEOUT)
    parser.add_argument("--tts-timeout", type=int, default=180)
    parser.add_argument("--max-text-length", type=int, default=2000)
    parser.add_argument("--systemctl-timeout", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.restart_hermes and not args.apply:
        raise SystemExit("--restart-hermes requires --apply")
    if args.no_start and not args.apply:
        raise SystemExit("--no-start is only meaningful with --apply")
    if args.no_systemctl and not args.apply:
        raise SystemExit("--no-systemctl is only meaningful with --apply")

    plan = build_plan(args)
    result: dict[str, Any] = {
        "success": True,
        "applied": args.apply,
        "plan": plan,
    }
    if args.apply:
        result["apply_result"] = apply_plan(plan, args)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
