"""Compatibility probe for the Claude Code CLI subprocess adapter.

Verifies the installed `claude` binary supports the protocol features the
adapter relies on. Runs at adapter init in production (results cached for
``probe_cache_seconds``) and in CI as a gating test.

PR 1 ships discovery + version helpers + the integration entry points.
Subsequent PRs use the probe; v1 adapter init refuses to start if the
probe fails any assertion.

Public surface:

  * ``discover_binary(path=None) -> str``
  * ``parse_version_string(s) -> tuple[int, int, int]``
  * ``check_version(version, min_version) -> tuple[int, int, int]``
  * ``check_env_hygiene(env, *, require_token=True) -> dict[str, str]``
  * ``run_probe(config) -> ProbeResult``
  * ``__main__`` runs the probe and prints results as JSON.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

from agent.claude_cli import errors

_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


def discover_binary(path: Optional[str] = None) -> str:
    """Locate the ``claude`` binary.

    If ``path`` is given, verify it exists and is executable. Otherwise
    look it up on PATH.

    Raises ``ClaudeCliUnavailable`` if the binary cannot be located.
    """
    if path is not None:
        if not os.path.isfile(path):
            raise errors.ClaudeCliUnavailable(
                f"configured claude path does not exist: {path}"
            )
        if not os.access(path, os.X_OK):
            raise errors.ClaudeCliUnavailable(
                f"configured claude path is not executable: {path}"
            )
        return path
    found = shutil.which("claude")
    if found is None:
        raise errors.ClaudeCliUnavailable(
            "`claude` not found on PATH; install Claude Code from "
            "https://docs.anthropic.com/en/docs/claude-code or set the "
            "binary path explicitly in provider config"
        )
    return found


def parse_version_string(text: str) -> tuple[int, int, int]:
    """Extract a (major, minor, patch) tuple from ``claude --version`` output.

    Accepts forms like ``"2.1.143 (Claude Code)"``, ``"2.1.143"``,
    ``"v2.1.143 (Claude Code)"``.

    Raises ``ClaudeCliIncompatible`` if no version can be parsed.
    """
    match = _VERSION_RE.search(text)
    if match is None:
        raise errors.ClaudeCliIncompatible(
            f"unexpected `claude --version` output: {text!r}"
        )
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_version(
    version: tuple[int, int, int],
    *,
    min_version: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Assert ``version`` is at or above ``min_version``.

    Returns the version tuple on success. Raises ``ClaudeCliVersionTooOld``
    on failure with a message naming both versions.
    """
    if version < min_version:
        raise errors.ClaudeCliVersionTooOld(
            f"installed claude is {'.'.join(map(str, version))}; "
            f"adapter requires at least {'.'.join(map(str, min_version))}"
        )
    return version


_DEFAULT_STRIP_ENV = ("ANTHROPIC_API_KEY",)


def check_env_hygiene(
    env: dict[str, str],
    *,
    require_token: bool = True,
    strip_env: Optional[list[str]] = None,
) -> dict[str, str]:
    """Sanitize a subprocess environment dict before spawning ``claude``.

    Strips ``ANTHROPIC_API_KEY`` (always) plus any names in ``strip_env``,
    because their presence may redirect Claude Code to API-key billing
    instead of the user's intended OAuth/Max-plan path.

    If ``require_token`` is True (default), asserts ``CLAUDE_CODE_OAUTH_TOKEN``
    is set to a non-empty value; raises ``ClaudeCliAuthMissing`` otherwise.

    Returns a new dict; does not mutate the input.
    """
    to_strip = set(_DEFAULT_STRIP_ENV)
    if strip_env:
        to_strip.update(strip_env)
    sanitized = {k: v for k, v in env.items() if k not in to_strip}
    if require_token:
        token = sanitized.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not token:
            raise errors.ClaudeCliAuthMissing(
                "CLAUDE_CODE_OAUTH_TOKEN is required for the Claude Code CLI "
                "subprocess adapter. Wire it via Infisical or set explicitly "
                "in /root/.hermes/.env. See docs/integrations/providers."
            )
    return sanitized
