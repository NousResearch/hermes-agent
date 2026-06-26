#!/usr/bin/env python3
"""Validate the local Hermes + voice stack without touching the live gateway.

This aggregate preflight runs the existing focused verifiers in a safe order:

1. The local Hermes CLI entrypoint starts with an isolated HERMES_HOME.
2. Hermes command-provider TTS returns WhatsApp-ready Ogg/Opus from voice.
3. Hermes' stream command path returns raw 48 kHz mono 20 ms pcm_s16le frames.
4. Optionally, the voice WebRTC sidecar full-duplex smoke passes locally.

By default the script creates a temporary Hermes home and removes it after a
passing or failing run. Pass --keep-home to inspect the generated config.
Pass --live-hermes-root to also prove the checkout used by a local gateway has
the voice-native WhatsApp and sidecar surfaces from this branch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_COMMAND_TEXT = "Hermes local voice command preflight."
DEFAULT_STREAM_TEXT = "Hermes local voice stream preflight."
DEFAULT_FULL_DUPLEX_INBOUND_TEXT = "hello world"
DEFAULT_FULL_DUPLEX_OUTBOUND_TEXT = "Hello from Hermes through the local voice sidecar."

LIVE_ROOT_REQUIREMENTS = (
    {
        "path": "tools/tts_tool.py",
        "description": "command-provider voice-compatible Opus routing",
        "tokens": (
            "voice_compatible",
            "libopus",
            "-application",
            "voip",
        ),
    },
    {
        "path": "gateway/platforms/whatsapp_cloud.py",
        "description": "WhatsApp Cloud voice notes and calling sidecar client",
        "tokens": (
            "calling_sidecar_url",
            "voice.webrtc_sidecar",
            "_send_calling_sidecar_tts_stream_command",
            "-application",
            "voip",
        ),
    },
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def script_path(name: str) -> Path:
    path = repo_root() / "scripts" / name
    if not path.is_file():
        raise SystemExit(f"script not found: {path}")
    return path


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


def default_hermes_bin() -> str:
    local = repo_root() / ".venv" / "bin" / "hermes"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return "hermes"


def default_voice_repo() -> Path | None:
    configured = os.environ.get("VOICE_REPO")
    return Path(configured) if configured else None


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = stdout.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if text[index + end :].strip():
            continue
        if not isinstance(parsed, dict):
            raise ValueError("JSON root must be an object")
        return parsed
    raise ValueError("command did not print a JSON object")


def run_process(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def audit_live_hermes_root(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    failures: list[str] = []
    checked: list[dict[str, Any]] = []

    if not resolved.is_dir():
        failures.append(f"live Hermes root is not a directory: {resolved}")
        return {
            "success": False,
            "root": str(resolved),
            "checked": checked,
            "failures": failures,
        }

    for requirement in LIVE_ROOT_REQUIREMENTS:
        rel_path = str(requirement["path"])
        required_tokens = tuple(str(token) for token in requirement["tokens"])
        path = resolved / rel_path
        check = {
            "path": rel_path,
            "description": requirement["description"],
            "required_tokens": list(required_tokens),
        }
        if not path.is_file():
            check["present"] = False
            failures.append(f"{rel_path} is missing")
            checked.append(check)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [token for token in required_tokens if token not in text]
        check["present"] = True
        check["missing_tokens"] = missing
        if missing:
            failures.append(f"{rel_path} is missing tokens: {', '.join(missing)}")
        checked.append(check)

    return {
        "success": not failures,
        "root": str(resolved),
        "checked": checked,
        "failures": failures,
    }


def require_live_hermes_root(root: Path) -> dict[str, Any]:
    result = audit_live_hermes_root(root)
    if result["success"] is not True:
        raise SystemExit(
            "live Hermes root is missing voice-native integration surfaces:\n- "
            + "\n- ".join(result["failures"])
        )
    return result


def run_plain_step(
    name: str,
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = run_process(command, timeout=timeout, env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"{name} failed with exit code {completed.returncode}"
            + (f": {detail[:1000]}" if detail else "")
        )
    first_line = next(
        (line for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    return {
        "success": True,
        "command": shell_join(command),
        "stdout_first_line": first_line,
    }


def run_json_step(
    name: str,
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = run_process(command, timeout=timeout, env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"{name} failed with exit code {completed.returncode}"
            + (f": {detail[:1000]}" if detail else "")
        )
    try:
        result = parse_json_object(completed.stdout)
    except ValueError as exc:
        raise SystemExit(f"{name} did not return JSON: {exc}") from exc
    if result.get("success") is not True:
        raise SystemExit(
            f"{name} reported failure: "
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return {
        "success": True,
        "command": shell_join(command),
        "result": result,
    }


def command_tts_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    hermes_home: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_command_tts.py")),
        "--voice-bin",
        voice_bin,
        "--hermes-home",
        str(hermes_home),
        "--force",
        "--provider",
        args.provider,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.tts_timeout:g}",
    ]
    if args.keep_home:
        command.append("--keep-home")
    command.extend(["--text", args.command_text])
    return command


def stream_tts_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_stream_tts.py")),
        "--voice-bin",
        voice_bin,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.stream_timeout:g}",
        "--text",
        args.stream_text,
    ]
    if args.stream_command_template:
        command.extend(["--command-template", args.stream_command_template])
    return command


def resolve_voice_repo_for_full_duplex(args: argparse.Namespace) -> Path | None:
    if args.skip_full_duplex:
        return None
    if args.voice_repo is None:
        raise SystemExit(
            "full-duplex validation needs --voice-repo or VOICE_REPO; "
            "pass --skip-full-duplex to validate only command and stream TTS"
        )
    voice_repo = args.voice_repo.expanduser().resolve()
    smoke = voice_repo / "examples" / "webrtc-sidecar" / "full_duplex_loopback_smoke.py"
    if not smoke.is_file():
        raise SystemExit(f"voice full-duplex smoke not found: {smoke}")
    return voice_repo


def full_duplex_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    voice_repo: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_full_duplex_sidecar.py")),
        "--voice-repo",
        str(voice_repo),
        "--voice-bin",
        voice_bin,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.full_duplex_timeout:g}",
        "--inbound-text",
        args.full_duplex_inbound_text,
        "--outbound-text",
        args.full_duplex_outbound_text,
    ]
    if args.webrtc_python_bin:
        command.extend(["--python-bin", args.webrtc_python_bin])
    for word in args.expect_word:
        command.extend(["--expect-word", word])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--hermes-bin", default=os.environ.get("HERMES_BIN", default_hermes_bin()))
    parser.add_argument("--hermes-home", type=Path)
    parser.add_argument("--keep-home", action="store_true")
    parser.add_argument("--provider", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--tts-timeout", type=float, default=180.0)
    parser.add_argument("--stream-timeout", type=float, default=180.0)
    parser.add_argument("--full-duplex-timeout", type=float, default=90.0)
    parser.add_argument("--cli-timeout", type=float, default=10.0)
    parser.add_argument("--command-text", default=DEFAULT_COMMAND_TEXT)
    parser.add_argument("--stream-text", default=DEFAULT_STREAM_TEXT)
    parser.add_argument("--stream-command-template")
    parser.add_argument(
        "--live-hermes-root",
        type=Path,
        help=(
            "Optional checkout used by a running local gateway. When provided, "
            "fail unless that checkout contains the voice-native WhatsApp "
            "and calling sidecar integration surfaces."
        ),
    )
    parser.add_argument("--voice-repo", type=Path, default=default_voice_repo())
    parser.add_argument("--webrtc-python-bin", default=os.environ.get("VOICE_WEBRTC_PYTHON"))
    parser.add_argument("--skip-full-duplex", action="store_true")
    parser.add_argument(
        "--full-duplex-inbound-text",
        default=DEFAULT_FULL_DUPLEX_INBOUND_TEXT,
    )
    parser.add_argument(
        "--full-duplex-outbound-text",
        default=DEFAULT_FULL_DUPLEX_OUTBOUND_TEXT,
    )
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the inbound full-duplex transcript; repeatable",
    )
    args = parser.parse_args()
    if args.expect_word is None:
        args.expect_word = ["hello", "world"]
    return args


def main() -> int:
    args = parse_args()
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    hermes_bin = resolve_executable(args.hermes_bin, label="Hermes CLI")
    voice_repo = resolve_voice_repo_for_full_duplex(args)

    temporary_home = args.hermes_home is None
    hermes_home = (
        Path(tempfile.mkdtemp(prefix="hermes-voice-local-stack."))
        if temporary_home
        else args.hermes_home.expanduser().resolve()
    )
    hermes_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)

    checks: dict[str, Any] = {}
    try:
        if args.live_hermes_root is not None:
            checks["live_hermes_root"] = require_live_hermes_root(
                args.live_hermes_root
            )
        checks["hermes_cli"] = run_plain_step(
            "Hermes CLI",
            [hermes_bin, "--help"],
            timeout=args.cli_timeout,
            env=env,
        )
        checks["command_tts"] = run_json_step(
            "command TTS verifier",
            command_tts_command(args, voice_bin=voice_bin, hermes_home=hermes_home),
            timeout=args.tts_timeout + 30,
            env=env,
        )
        checks["stream_tts"] = run_json_step(
            "stream TTS verifier",
            stream_tts_command(args, voice_bin=voice_bin),
            timeout=args.stream_timeout + 30,
            env=env,
        )
        if voice_repo is None:
            checks["full_duplex_sidecar"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-full-duplex was provided",
            }
        else:
            checks["full_duplex_sidecar"] = run_json_step(
                "full-duplex sidecar verifier",
                full_duplex_command(args, voice_bin=voice_bin, voice_repo=voice_repo),
                timeout=args.full_duplex_timeout + 15,
                env=env,
            )

        print(
            json.dumps(
                {
                    "success": True,
                    "hermes_home": str(hermes_home),
                    "retained": bool(args.keep_home or not temporary_home),
                    "voice_bin": voice_bin,
                    "hermes_bin": hermes_bin,
                    "voice_repo": str(voice_repo) if voice_repo is not None else None,
                    "checks": checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if temporary_home and not args.keep_home:
            shutil.rmtree(hermes_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
