"""Codex CLI device-auth helper for operator-run parent VMs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Sequence, TextIO


DEFAULT_STATUS_COMMAND = ("codex", "login", "status")
DEFAULT_SIGNIN_COMMAND = ("codex", "login", "--device-auth")
DEFAULT_AUTH_TIMEOUT_SECONDS = 15 * 60
DEFAULT_CODE_TIMEOUT_SECONDS = 60

_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_CODE_PATTERNS = (
    re.compile(
        r"(?:user[_ -]?code|device[_ -]?code|code)\D{0,40}"
        r"([A-Z0-9]{4}(?:[- ][A-Z0-9]{4}){1,5})",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-Z0-9]{4}(?:-[A-Z0-9]{4}){1,5})\b"),
)
_SECRET_LINE_RE = re.compile(
    r"(?i)(access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|authorization|bearer)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class DevicePrompt:
    user_code: str
    verification_url: str


class GptAuthError(RuntimeError):
    """Expected operator-facing auth failure with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ProcessLike(Protocol):
    stdout: Optional[TextIO]

    def poll(self) -> Optional[int]: ...

    def wait(self, timeout: Optional[float] = None) -> int: ...

    def kill(self) -> None: ...


class Runner(Protocol):
    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult: ...

    def popen(self, argv: Sequence[str]) -> ProcessLike: ...


class SubprocessRunner:
    """Thin subprocess wrapper kept injectable for tests and dry-run seams."""

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GptAuthError("codex_cli_missing", f"Codex CLI not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GptAuthError(
                "status_timeout",
                "Timed out while checking Codex CLI login status.",
            ) from exc
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def popen(self, argv: Sequence[str]) -> ProcessLike:
        try:
            return subprocess.Popen(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise GptAuthError("codex_cli_missing", f"Codex CLI not found: {argv[0]}") from exc


def _parse_command(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    cleaned = (value or "").strip()
    if not cleaned:
        return tuple(default)
    try:
        parsed = tuple(shlex.split(cleaned))
    except ValueError as exc:
        raise GptAuthError("invalid_command", f"Invalid command override: {exc}") from exc
    if not parsed:
        raise GptAuthError("invalid_command", "Command override cannot be empty.")
    return parsed


def _redact_line(line: str) -> str:
    clean = _strip_ansi(line)
    if _SECRET_LINE_RE.search(clean):
        return "[redacted sensitive output]"
    return clean.strip()


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", value)


def _looks_authenticated(result: CommandResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        return False
    if "not logged in" in combined or "not authenticated" in combined:
        return False
    return True


def check_authenticated(
    runner: Runner,
    status_command: Sequence[str],
    *,
    timeout_seconds: float = 20.0,
) -> bool:
    return _looks_authenticated(runner.run(status_command, timeout=timeout_seconds))


def _extract_prompt_from_json(value: str) -> DevicePrompt | None:
    try:
        payload = json.loads(value)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    code = (
        payload.get("user_code")
        or payload.get("device_code")
        or payload.get("code")
    )
    url = (
        payload.get("verification_uri_complete")
        or payload.get("verification_url")
        or payload.get("verification_uri")
        or payload.get("url")
    )
    if isinstance(code, str) and isinstance(url, str) and code.strip() and url.strip():
        return DevicePrompt(_normalize_code(code), _normalize_url(url))
    return None


def _normalize_code(value: str) -> str:
    return value.strip().upper().replace(" ", "-")


def _normalize_url(value: str) -> str:
    return value.strip().rstrip(".,;")


def _extract_prompt_from_line(line: str) -> DevicePrompt | None:
    clean = _strip_ansi(line)
    parsed_json = _extract_prompt_from_json(clean.strip())
    if parsed_json is not None:
        return parsed_json

    url_match = _URL_RE.search(clean)
    url = _normalize_url(url_match.group(0)) if url_match else ""
    code = ""
    for pattern in _CODE_PATTERNS:
        match = pattern.search(clean)
        if match:
            code = _normalize_code(match.group(1))
            break
    if code and url:
        return DevicePrompt(code, url)
    return None


def _merge_prompt(current: DevicePrompt | None, line: str) -> DevicePrompt | None:
    clean = _strip_ansi(line)
    parsed = _extract_prompt_from_line(clean)
    current_code = current.user_code if current else ""
    current_url = current.verification_url if current else ""
    if parsed is not None:
        current_code = current_code or parsed.user_code
        current_url = current_url or parsed.verification_url
    else:
        url_match = _URL_RE.search(clean)
        if url_match and not current_url:
            current_url = _normalize_url(url_match.group(0))
        if not current_code:
            for pattern in _CODE_PATTERNS:
                match = pattern.search(clean)
                if match:
                    current_code = _normalize_code(match.group(1))
                    break
    if current_code and current_url:
        return DevicePrompt(current_code, current_url)
    if current_code or current_url:
        return DevicePrompt(current_code, current_url)
    return None


def _start_output_reader(process: ProcessLike) -> tuple[queue.Queue[str | None], threading.Thread]:
    lines: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        stream = process.stdout
        if stream is None:
            lines.put(None)
            return
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                lines.put(line)
        finally:
            lines.put(None)

    thread = threading.Thread(target=_reader, name="gpt-auth-codex-output", daemon=True)
    thread.start()
    return lines, thread


def _collect_until_prompt(
    process: ProcessLike,
    lines: queue.Queue[str | None],
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[DevicePrompt, list[str]]:
    prompt: DevicePrompt | None = None
    safe_tail: list[str] = []

    returncode: int | None = None
    while monotonic() < deadline:
        timeout = max(0.01, min(0.1, deadline - monotonic()))
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            returncode = process.poll()
            if returncode is not None:
                break
            continue

        if line is None:
            returncode = process.poll()
            if returncode is not None:
                break
            continue

        safe_tail.append(_redact_line(line))
        safe_tail = safe_tail[-20:]
        prompt = _merge_prompt(prompt, line)
        if prompt and prompt.user_code and prompt.verification_url:
            return prompt, safe_tail

    if returncode is not None:
        _drain_available(lines, safe_tail)
        raise _classify_signin_failure(returncode, safe_tail)

    raise GptAuthError(
        "device_code_timeout",
        "Timed out before Codex CLI printed a device code and verification URL.",
    )


def _drain_available(lines: queue.Queue[str | None], safe_tail: list[str]) -> None:
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return
        if line is None:
            continue
        safe_tail.append(_redact_line(line))
        del safe_tail[:-20]


def _classify_signin_failure(returncode: int, safe_tail: Iterable[str]) -> GptAuthError:
    text = "\n".join(safe_tail).lower()
    if "access_denied" in text or "authorization_denied" in text or "denied" in text:
        return GptAuthError("auth_denied", "Codex device authorization was denied.")
    if "expired_token" in text or "expired" in text:
        return GptAuthError("auth_code_expired", "Codex device code expired before authorization completed.")
    if "timeout" in text or "timed out" in text:
        return GptAuthError("auth_timeout", "Codex device authorization timed out.")
    return GptAuthError("signin_failed", f"Codex device auth command exited with status {returncode}.")


def _wait_for_signin(
    process: ProcessLike,
    lines: queue.Queue[str | None],
    safe_tail: list[str],
    *,
    deadline: float,
    monotonic: Callable[[], float],
    reader_thread: threading.Thread | None = None,
) -> None:
    remaining = max(0.0, deadline - monotonic())
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise GptAuthError("auth_timeout", "Codex device authorization timed out.") from exc
    if reader_thread is not None:
        reader_thread.join(timeout=1.0)
    _drain_available(lines, safe_tail)
    if returncode != 0:
        raise _classify_signin_failure(returncode, safe_tail)


def run(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
    input_func: Callable[[str], str] = input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = argparse.ArgumentParser(
        prog="gpt-auth",
        description="Validate or refresh Codex CLI device authentication.",
    )
    parser.add_argument(
        "--status-command",
        default=os.getenv("GPT_AUTH_STATUS_COMMAND", ""),
        help="Override status command, shell-split. Default: codex login status.",
    )
    parser.add_argument(
        "--signin-command",
        default=os.getenv("GPT_AUTH_SIGNIN_COMMAND", ""),
        help="Override device auth command, shell-split. Default: codex login --device-auth.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("GPT_AUTH_TIMEOUT_SECONDS", DEFAULT_AUTH_TIMEOUT_SECONDS)),
        help="Total seconds to wait for the operator to complete device auth.",
    )
    parser.add_argument(
        "--code-timeout-seconds",
        type=float,
        default=float(os.getenv("GPT_AUTH_CODE_TIMEOUT_SECONDS", DEFAULT_CODE_TIMEOUT_SECONDS)),
        help="Seconds to wait for Codex CLI to print the device code and URL.",
    )
    parse_argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(parse_argv)

    runner = runner or SubprocessRunner()

    try:
        status_command = _parse_command(args.status_command, DEFAULT_STATUS_COMMAND)
        signin_command = _parse_command(args.signin_command, DEFAULT_SIGNIN_COMMAND)

        if check_authenticated(runner, status_command):
            print("Codex CLI session is already authenticated.", file=out)
            return 0

        deadline = monotonic() + max(1.0, float(args.timeout_seconds))
        code_deadline = min(deadline, monotonic() + max(0.0, float(args.code_timeout_seconds)))

        process = runner.popen(signin_command)
        lines, reader_thread = _start_output_reader(process)
        try:
            prompt, safe_tail = _collect_until_prompt(
                process,
                lines,
                deadline=code_deadline,
                monotonic=monotonic,
            )
        except GptAuthError:
            process.kill()
            raise

        print(f"Device code: {prompt.user_code}", file=out, flush=True)
        print(f"Verification URL: {prompt.verification_url}", file=out, flush=True)
        print(
            "HITL PAUSE: complete Codex device auth and 2FA, then press Enter to verify.",
            file=out,
            flush=True,
        )

        try:
            input_func("")
        except KeyboardInterrupt:
            process.kill()
            raise GptAuthError("operator_cancelled", "Operator cancelled Codex device auth.")
        except EOFError as exc:
            process.kill()
            raise GptAuthError(
                "hitl_confirmation_unavailable",
                "No operator confirmation was available at the HITL pause.",
            ) from exc

        _wait_for_signin(
            process,
            lines,
            safe_tail,
            deadline=deadline,
            monotonic=monotonic,
            reader_thread=reader_thread,
        )

        if not check_authenticated(runner, status_command):
            raise GptAuthError(
                "auth_verification_failed",
                "Codex device auth finished, but login status is still unauthenticated.",
            )

        print("Codex CLI session verified.", file=out)
        return 0
    except GptAuthError as exc:
        print(f"ERROR {exc.code}: {exc.message}", file=err)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
