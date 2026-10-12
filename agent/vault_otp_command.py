"""Second-factor codes from a user-configured command (``vault.otp_commands``).

Some sites send the one-time code by email or SMS and the login has no authenticator seed. A headless
session (cron) cannot ask the user, and the model must never read the code. The user can name a helper
command per EXACT origin; ``browser_vault_enter_code`` runs it server-side and types the code into the
page through the same redacted fill path as a minted TOTP:

    vault:
      otp_commands:
        https://www.americanexpress.com: ~/.local/bin/amex-otp-from-gmail

Output contract (the ``key_cmd`` idiom): stdout is ONLY the code, bare or as JSON
``{"code": "...", "reference": "..."}``. ``reference`` is non-secret (a message id) and is returned to
the model so it can clean up the mailbox after the site accepts the code. Empty stdout with exit 0 means
"not delivered yet": the command is polled until a code arrives or the deadline passes. The helper runs
FOR the served profile (``served_profile_child_env``) and receives ``HERMES_OTP_ORIGIN`` and
``HERMES_OTP_SINCE`` (epoch seconds; ignore messages older than this). Its stdout and stderr never reach
the model or the logs: failures report the exit status only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

_RUN_TIMEOUT_SECONDS = 30
# Email delivery takes seconds to a minute; past this the code has usually expired anyway.
_DEADLINE_SECONDS = 120.0
_POLL_SECONDS = 5.0
# A request older than this is not "the code we just asked for".
_SINCE_WINDOW_SECONDS = 300
_RE_CODE = re.compile(r"[A-Za-z0-9]{4,12}")


class OtpCommandError(RuntimeError):
    """The configured helper failed. The message never contains helper output."""


@dataclass(frozen=True)
class FetchedOtp:
    code: str
    reference: str = ""


def otp_command_for(origin: str) -> str:
    """The helper configured for exactly ``origin``, or ""."""
    from agent.vault_backends.base import _cfg

    commands = _cfg().get("otp_commands") or {}
    return str(commands.get(origin) or "").strip() if isinstance(commands, dict) else ""


def _parse(stdout: str) -> Optional[FetchedOtp]:
    text = stdout.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            raise OtpCommandError("the OTP command printed invalid JSON") from None
        code, reference = str(data.get("code") or "").strip(), str(data.get("reference") or "").strip()
    else:
        code, reference = text, ""
    if not _RE_CODE.fullmatch(code):
        raise OtpCommandError("the OTP command printed something that is not a single code")
    return FetchedOtp(code, reference[:200])


def _run_once(command: str, origin: str, since: int) -> Optional[FetchedOtp]:
    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env(inherit_credentials=True)
    env.update({"HERMES_OTP_ORIGIN": origin, "HERMES_OTP_SINCE": str(since)})
    try:
        done = subprocess.run(os.path.expanduser(command), shell=True, capture_output=True, text=True,
                              errors="replace", timeout=_RUN_TIMEOUT_SECONDS, env=env)
    except subprocess.TimeoutExpired:
        raise OtpCommandError(f"the OTP command timed out after {_RUN_TIMEOUT_SECONDS}s") from None
    if done.returncode != 0:
        raise OtpCommandError(f"the OTP command exited with status {done.returncode}")
    return _parse(done.stdout)


def fetch_otp_code(origin: str, *, deadline_s: float = _DEADLINE_SECONDS,
                   poll_s: float = _POLL_SECONDS) -> Optional[FetchedOtp]:
    """Run the helper for ``origin`` until it prints a code. None when no helper is configured;
    :class:`OtpCommandError` when it fails or no code arrives before the deadline."""
    command = otp_command_for(origin)
    if not command:
        return None
    since = int(time.time()) - _SINCE_WINDOW_SECONDS
    deadline = time.monotonic() + deadline_s
    while True:
        fetched = _run_once(command, origin, since)
        if fetched is not None:
            return fetched
        if time.monotonic() + poll_s > deadline:
            raise OtpCommandError(f"no code arrived within {int(deadline_s)}s")
        time.sleep(poll_s)
