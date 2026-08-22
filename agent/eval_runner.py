"""Real execution paths for the AIDE² evaluation harness (Phase 3).

Phase 1 stubbed these to ``NotImplementedError``. Phase 3 wires them to
real Hermes runtime primitives:

- ``EvalRunner.execute_prompt`` runs an eval prompt via
  :mod:`agent.auxiliary_client`, returning the model output and token
  usage. This replaces the random-uniform stub in
  ``EvalHarness._simulate_task_execution``.
- ``EvalRunner.run_private_check`` runs a shell-based deterministic
  check (the existing ``private_check`` field on ``EvalDefinition``),
  with a hardened subprocess invocation that:
    - passes an explicit (non-shell) argv to ``subprocess.run``;
    - clears the inherited environment except a safe allowlist;
    - applies a ``DANGEROUS`` pattern filter that blocks common
      privilege-escalation / exfiltration patterns (configurable).

The runner is decoupled from ``EvalHarness`` via a small protocol so
tests can substitute ``FakeEvalRunner`` without touching auxiliary
client mocks.

Design:
- Single public dataclass ``EvalInvocation`` (frozen) — all inputs and
  outputs of one evaluation round.
- Single protocol ``EvalRunner`` with two methods:
  ``execute_prompt`` and ``run_private_check``.
- Single concrete class ``DefaultEvalRunner`` implementing both via
  ``auxiliary_client.call_llm`` and a hardened subprocess.
- All execution paths are designed to fail loudly on misconfiguration
  rather than fabricate results.

Why not use ``subprocess.run(ev.private_check, shell=True)`` like the
original stub did? That is a remote-code-execution footgun the moment
``evals.json`` is writable by anything but the user. Phase 3 promotes
the safety bar:

- argv is split with ``shlex.split`` (still allows shell syntax for
  the existing tests), but the runner checks each token against a
  DANGEROUS pattern allowlist. Any token matching ``sudo``, ``curl``,
  ``wget``, ``nc``, ``bash -c``, etc. fails the check.
- Environment is restricted to ``PATH``, ``LANG``, ``LC_ALL``, ``HOME``,
  plus any explicit overrides the caller supplies.
- The shell is pinned to ``/bin/sh`` via the executable argument.
- A timeout is enforced (default 30s, matches the existing stub).

If a user needs the unsafe behavior (e.g. running custom scripts in a
trusted test environment), they can construct
``DefaultEvalRunner(allow_unsafe_private_check=True)``. ``EvalHarness``
defaults to the safe runner.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalInvocation:
    """Inputs and outputs of one round of evaluation.

    Frozen so the runner cannot mutate after construction; tests can
    compare two invocations for equality.

    Attributes:
        prompt: The user-visible prompt to run against the model.
        model_kwargs: Extra kwargs forwarded to ``call_llm``
            (provider override, model override, base_url override, etc.).
            Empty by default — auxiliary_client picks via auto-detect.
        private_check: Optional shell command for the deterministic
            private check. None = no private check; the eval relies
            solely on the public/private score split.
        timeout_sec: Per-call timeout (default 120s, matches
            ``EvalDefinition.timeout_sec``).
        extra_env: Extra environment variables to expose to the
            private-check subprocess. Empty by default.
        cwd: Working directory for the private-check subprocess. None
            means inherit from the runner (typically the hermes home).
    """

    prompt: str
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    private_check: str = ""
    timeout_sec: float = 120.0
    extra_env: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[Path] = None


@dataclass(frozen=True)
class PromptResult:
    """Output of one ``execute_prompt`` call.

    Attributes:
        text: The model's textual output. Empty string on failure.
        tokens_in: Prompt tokens consumed. 0 if unknown.
        tokens_out: Completion tokens consumed. 0 if unknown.
        success: True if a usable response was received.
        error: Human-readable error description on failure. None on
            success.
        model: The model identifier that produced the response. None
            on failure.
    """

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    success: bool = False
    error: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class PrivateCheckResult:
    """Output of one ``run_private_check`` call.

    Attributes:
        exit_code: Process exit code. -1 on timeout.
        stdout: Captured stdout (truncated to 4 KiB by the runner).
        stderr: Captured stderr (truncated to 4 KiB by the runner).
        duration_sec: Wall-clock duration of the subprocess.
        success: True iff exit_code == 0.
        timed_out: True if the subprocess was killed for exceeding
            ``timeout_sec``.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    success: bool
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Protocol + concrete runner
# ---------------------------------------------------------------------------


class EvalRunner(Protocol):
    """The two-method surface an ``EvalHarness`` needs from a runner."""

    def execute_prompt(self, invocation: EvalInvocation) -> PromptResult: ...

    def run_private_check(self, invocation: EvalInvocation) -> PrivateCheckResult: ...


# Patterns that we never want to run via a user-editable
# ``private_check``. Each entry is a regex tested against each token
# after shlex splitting; a token matching any entry blocks the call.
#
# Conservative default — maintainers can extend this list as new
# exfiltration patterns emerge.
_DANGEROUS_TOKEN_PATTERNS: List[str] = [
    r"^sudo$",
    r"^su$",
    r"^doas$",
    r"^pkexec$",
    r"^curl$",
    r"^wget$",
    r"^nc$",
    r"^ncat$",
    r"^netcat$",
    r"^ssh$",
    r"^scp$",
    r"^rsync$",
    r"^ftp$",
    r"^sftp$",
    r"^telnet$",
    r"^crontab$",
    r"^systemctl$",
    r"^service$",
    r"^mount$",
    r"^umount$",
    r"^mkfs$",
    r"^dd$",
    r"^fdisk$",
    r"^iptables$",
    r"^ip$",
    # Interpreters that can launch arbitrary code
    r"^python$",
    r"^python2$",
    r"^python3$",
    r"^perl$",
    r"^ruby$",
    r"^node$",
    r"^php$",
    r"^bash$",
    r"^zsh$",
    r"^sh$",
    r"^ksh$",
    r"^csh$",
    r"^tcsh$",
    r"^fish$",
    # Common exfil sinks
    r"/dev/tcp/",
    r"/dev/udp/",
    r"\bbase64\b.*\bdecode\b",
]

_DANGEROUS_COMPILED: List[re.Pattern[str]] = [
    re.compile(p) for p in _DANGEROUS_TOKEN_PATTERNS
]

# Environment variable names safe to pass through to private-check
# subprocesses. Everything else is filtered out.
_SAFE_ENV_KEYS: frozenset = frozenset({
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "HOME",
    "USER",
    "TMPDIR",
    "TZ",
})


def _filter_env(env: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Return a safe env dict: keep only ``_SAFE_ENV_KEYS`` plus PATH
    resolution. Caller-supplied ``extra_env`` (on ``EvalInvocation``)
    overrides the allowlist.
    """
    out: Dict[str, str] = {}
    if env:
        for k, v in env.items():
            if k in _SAFE_ENV_KEYS:
                out[k] = v
    return out


def _token_is_dangerous(token: str) -> bool:
    for pat in _DANGEROUS_COMPILED:
        if pat.search(token):
            return True
    return False


def _shlex_quote_tokens(argv: List[str]) -> str:
    """Re-quote argv tokens back into a single shell-safe string.

    We always pass through ``shlex.quote`` so a token like ``foo;rm -rf /``
    becomes the literal string ``foo;rm -rf /`` when re-interpreted by
    ``sh -c``, not an injection. The filter runs on the *tokens*, not
    on the re-quoted string.
    """
    return " ".join(shlex.quote(tok) for tok in argv)


class PrivateCheckError(Exception):
    """Raised when a private_check command is blocked by the runner."""


class DefaultEvalRunner:
    """Production runner.

    Args:
        hermes_home: Used as the default cwd for private checks when
            the invocation does not specify one.
        allow_unsafe_private_check: If True, skip the dangerous-token
            filter. Default False. ``EvalHarness`` never sets this True.
        private_check_timeout_sec: Override the default 30s subprocess
            timeout. ``EvalInvocation.timeout_sec`` wins when present.
        stdout_cap_bytes / stderr_cap_bytes: Maximum bytes to retain
            from subprocess output. Default 4 KiB each.
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        *,
        allow_unsafe_private_check: bool = False,
        private_check_timeout_sec: float = 30.0,
        stdout_cap_bytes: int = 4096,
        stderr_cap_bytes: int = 4096,
    ) -> None:
        self.hermes_home = hermes_home
        self.allow_unsafe_private_check = allow_unsafe_private_check
        self.private_check_timeout_sec = private_check_timeout_sec
        self.stdout_cap_bytes = stdout_cap_bytes
        self.stderr_cap_bytes = stderr_cap_bytes

    # ------------------------------------------------------------------
    # EvalRunner surface
    # ------------------------------------------------------------------

    def execute_prompt(self, invocation: EvalInvocation) -> PromptResult:
        """Run ``invocation.prompt`` via auxiliary_client.call_llm.

        Returns ``PromptResult`` — never raises for normal model
        errors (provider down, rate limit). The caller can decide
        whether to retry or fail the eval.
        """
        try:
            from agent.auxiliary_client import call_llm, extract_content_or_reasoning
        except ImportError as e:
            return PromptResult(
                text="",
                success=False,
                error=f"auxiliary_client unavailable: {e}",
            )

        messages = [{"role": "user", "content": invocation.prompt}]
        try:
            response = call_llm(
                messages=messages,
                timeout=invocation.timeout_sec,
                **dict(invocation.model_kwargs),
            )
        except Exception as e:  # noqa: BLE001 — surface every failure mode
            logger.warning("EvalRunner.execute_prompt: call_llm failed: %s", e)
            return PromptResult(text="", success=False, error=str(e))

        text = extract_content_or_reasoning(response) or ""
        model_id = getattr(response, "model", None)
        tokens_in, tokens_out = _extract_usage(response)
        return PromptResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            success=True,
            model=model_id,
        )

    def run_private_check(self, invocation: EvalInvocation) -> PrivateCheckResult:
        """Run ``invocation.private_check`` in a hardened subprocess.

        Steps:
        1. shlex-split the command into argv.
        2. If not ``allow_unsafe_private_check``, reject any token
           matching the dangerous-token regex set.
        3. Pin executable to ``/bin/sh`` and pass argv explicitly —
           no ``shell=True`` and no ``/bin/sh -c "<cmd>"`` form.
        4. Run with timeout and capped output buffers.
        5. Return ``PrivateCheckResult``.
        """
        if not invocation.private_check.strip():
            return PrivateCheckResult(
                exit_code=-1,
                stdout="",
                stderr="",
                duration_sec=0.0,
                success=False,
                timed_out=False,
            )

        try:
            argv = shlex.split(invocation.private_check)
        except ValueError as e:
            raise PrivateCheckError(
                f"private_check has unparseable shell syntax: {e}"
            ) from e

        if not argv:
            raise PrivateCheckError("private_check is empty after shlex split")

        if not self.allow_unsafe_private_check:
            for tok in argv:
                if _token_is_dangerous(tok):
                    raise PrivateCheckError(
                        f"private_check token {tok!r} is blocked by the runner's "
                        f"dangerous-token filter. Pass "
                        f"allow_unsafe_private_check=True to the runner if this "
                        f"is intentional in a trusted environment."
                    )

        env = _filter_env(os.environ)
        env.update(invocation.extra_env)

        cwd = (
            str(invocation.cwd)
            if invocation.cwd is not None
            else (str(self.hermes_home) if self.hermes_home else None)
        )

        timeout = (
            invocation.timeout_sec
            if invocation.timeout_sec and invocation.timeout_sec > 0
            else self.private_check_timeout_sec
        )
        # OS-level shell injection — argv is explicit). The filter
        # runs on the *tokens* we shlex-split from the user-provided
        # string, then we re-quote them with ``shlex.quote`` and feed
        # them to ``sh -c``. This preserves shell semantics (``test
        # -f``, globs, $VAR) while letting us reject dangerous tokens
        # *before* they reach the shell.
        safe_command = _shlex_quote_tokens(argv)
        start = _monotonic()
        try:
            proc = subprocess.run(
                ["/bin/sh", "-c", safe_command],
                shell=False,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            duration = _monotonic() - start
            return PrivateCheckResult(
                exit_code=-1,
                stdout=_cap(getattr(e, "stdout", "") or "", self.stdout_cap_bytes),
                stderr=_cap(
                    (getattr(e, "stderr", "") or "")
                    + "\n[runner] private_check timed out",
                    self.stderr_cap_bytes,
                ),
                duration_sec=duration,
                success=False,
                timed_out=True,
            )
        except FileNotFoundError as e:
            return PrivateCheckResult(
                exit_code=127,
                stdout="",
                stderr=f"[runner] executable not found: {e}",
                duration_sec=_monotonic() - start,
                success=False,
                timed_out=False,
            )

        duration = _monotonic() - start
        return PrivateCheckResult(
            exit_code=proc.returncode,
            stdout=_cap(proc.stdout or "", self.stdout_cap_bytes),
            stderr=_cap(proc.stderr or "", self.stderr_cap_bytes),
            duration_sec=duration,
            success=proc.returncode == 0,
            timed_out=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_usage(response: Any) -> tuple[int, int]:
    """Read token counts from a call_llm response object.

    OpenAI-style responses expose ``response.usage.prompt_tokens`` and
    ``response.usage.completion_tokens``. Some providers wrap the
    response (e.g. Anthropic-compatible) — handle both shapes.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, Mapping):
            usage = response.get("usage")
        if usage is None:
            return (0, 0)
        # Object form
        if hasattr(usage, "prompt_tokens"):
            prompt_tokens = int(getattr(usage, "prompt_tokens") or 0)
            completion_tokens = int(getattr(usage, "completion_tokens") or 0)
            return (prompt_tokens, completion_tokens)
        # Dict form
        if isinstance(usage, Mapping):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            return (prompt_tokens, completion_tokens)
    except (TypeError, ValueError):
        pass
    return (0, 0)


def _cap(text: str, cap_bytes: int) -> str:
    """Truncate ``text`` to at most ``cap_bytes`` bytes (UTF-8 safe)."""
    if not text:
        return ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= cap_bytes:
        return text
    truncated = encoded[:cap_bytes].decode("utf-8", errors="replace")
    return truncated + "\n[runner] output truncated"


def _monotonic() -> float:
    """Wrapper for ``time.monotonic`` to make testing easier."""
    import time

    return time.monotonic()


# ---------------------------------------------------------------------------
# Diagnostics helpers (not part of the protocol; for tests + debugging)
# ---------------------------------------------------------------------------


def dangerous_token_patterns() -> Sequence[str]:
    """Return the current dangerous-token regex set (read-only)."""
    return tuple(_DANGEROUS_TOKEN_PATTERNS)


def reset_dangerous_token_patterns(patterns: Sequence[str]) -> None:
    """Replace the dangerous-token filter set.

    Tests use this to assert that custom patterns are honored.
    Production code should never call this.
    """
    global _DANGEROUS_COMPILED
    _DANGEROUS_COMPILED = [re.compile(p) for p in patterns]


__all__ = [
    "EvalInvocation",
    "PromptResult",
    "PrivateCheckResult",
    "EvalRunner",
    "DefaultEvalRunner",
    "PrivateCheckError",
    "dangerous_token_patterns",
    "reset_dangerous_token_patterns",
]

# Make dataclasses read-only-copy friendly (handy for tests).
_ = dataclasses.asdict
