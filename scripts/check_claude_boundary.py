#!/usr/bin/env python3
"""
Claude subscription boundary check.

Hermes routes a Claude subscription through the official ``claude-agent-sdk``.
The SDK owns the user's credentials; Hermes must not touch them, must not send
a subscription token to Anthropic's API itself, and must not dress its requests
up as Claude Code. This script makes those rules mechanical.

Four rules, all fatal:

  1. Credential access — no references to Claude/Anthropic credential stores
     (``.credentials.json``, ``~/.claude.json``, ``.claude/.credentials``,
     ``~/.anthropic/``).
  2. Direct OAuth inference — no code path that sends a Claude
     OAuth/subscription token to ``api.anthropic.com``. The API-key provider's
     ``/v1/models`` and ``/v1/messages`` calls are legitimate and allowed; what
     is forbidden is combining an Anthropic endpoint with
     ``CLAUDE_CODE_OAUTH_TOKEN`` or a Claude-credential-derived token.
  3. Identity spoofing — no ``claude-code/`` user-agent strings, no
     ``x-app: cli`` header, no Claude Code beta flags, and no system-prompt or
     tool-name rewriting that replaces "Hermes"/"Nous" to look like Claude Code.
  4. ``CLAUDE_CODE_OAUTH_TOKEN`` reads outside the allow-list.

Usage:
    python scripts/check_claude_boundary.py            # scan the tree
    python scripts/check_claude_boundary.py --verbose  # also list allow-list hits

Exit status:
    0 — no unallowed violations
    1 — at least one violation

Suppress a single intentional line with:
    ...  # claude-boundary: ok — <why>

ALLOW-LIST POLICY: every entry below names a file that still carries the
pre-SDK direct-OAuth path, with a justification. Entries tagged
``TODO(legacy-retirement)`` disappear when PR4 deletes that path — the check tightens
automatically, without anyone remembering to tighten it.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPRESS_MARKER = re.compile(r"#\s*claude-boundary\s*:\s*ok\b", re.IGNORECASE)

# Source roots that make up "Hermes source". Docs, the website, tests, and
# vendored trees are out of scope: this guards shipped behavior, and tests
# legitimately construct the very payloads the rules forbid in order to assert
# on them.
SOURCE_ROOTS = (
    "acp_adapter",
    "agent",
    "apps",
    "cron",
    "gateway",
    "hermes_cli",
    "model_tools.py",
    "plugins",
    "providers",
    "run_agent.py",
    "tools",
    "tui_gateway",
    "utils.py",
)

SCANNED_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs", ".cjs"}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "out",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
}


@dataclass
class Rule:
    """One boundary rule."""

    name: str
    pattern: re.Pattern
    message: str
    # Repo-relative path prefixes whose matches are tolerated. Each entry MUST
    # carry a justification in ALLOWLIST_REASONS.
    allowlist: tuple[str, ...] = ()
    # Optional second pattern that must ALSO be present for a match to count.
    require_also: re.Pattern | None = None
    # "line" — both patterns must be on the same line.
    # "file" — ``require_also`` may appear anywhere in the file. Needed for the
    #   direct-OAuth rule: a request's URL, headers, and token rarely share a
    #   line, so co-location in one module is the honest signal.
    scope: str = "line"
    hits: list[tuple[str, int, str]] = field(default_factory=list)


# ── Allow-list justifications ───────────────────────────────────────────────
# Keyed by repo-relative path prefix. Anything listed here is a KNOWN, CURRENT
# violation that PR4 removes; the entry is the audit trail for why the build is
# still green today.
ALLOWLIST_REASONS: dict[str, str] = {
    "agent/anthropic_adapter.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Holds the entire pre-SDK Claude path: the claude.ai PKCE flow, the "
        "keychain/credential-file readers and writer, the spoofed Claude Code "
        "user-agent / x-app / beta headers, and the prompt+tool-name rewriting. "
        "It must keep working until the Agent SDK runtime replaces it."
    ),
    "agent/account_usage.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Reads the Claude subscription usage endpoint with a Claude Code "
        "user-agent to render remaining plan quota."
    ),
    "agent/credential_pool.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Seeds and refreshes pooled Anthropic credentials from the Claude Code "
        "credential singleton."
    ),
    "agent/credential_sources.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Declares the claude_code credential source so users can suppress it."
    ),
    "agent/credential_persistence.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Names the Claude Code credential file among the persistence targets."
    ),
    "agent/agent_init.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Sets a Claude Code user-agent on the legacy Anthropic client."
    ),
    "agent/auxiliary_client.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Side-LLM clients inherit the legacy Anthropic OAuth headers and "
        "credential resolution."
    ),
    "run_agent.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Sets the legacy Anthropic client user-agent."
    ),
    "hermes_cli/doctor.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Probes the legacy Anthropic OAuth endpoint during diagnostics."
    ),
    "hermes_cli/auth.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "The anthropic registry entry still lists CLAUDE_CODE_OAUTH_TOKEN so "
        "existing users keep resolving while the old path is live."
    ),
    "hermes_cli/config.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Documents CLAUDE_CODE_OAUTH_TOKEN in OPTIONAL_ENV_VARS and offers the "
        "legacy 'use my Claude Code credentials' setup step."
    ),
    "hermes_cli/web_server.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Hosts the dashboard's Anthropic PKCE endpoints, which mint and store a "
        "claude.ai OAuth token."
    ),
    "hermes_cli/main.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "The `hermes model` Claude subscription flow shells out to the legacy "
        "token-setup command."
    ),
    "hermes_cli/model_setup_flows.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Same legacy Claude subscription setup flow as hermes_cli/main.py."
    ),
    "hermes_cli/credential_lifecycle.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Enumerates the claude_code credential source for list/remove verbs."
    ),
    "hermes_cli/models.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "Stats the Claude credential file's mtime to invalidate a provider "
        "cache key; it reads no contents, but the reference goes with the path."
    ),
    "hermes_cli/setup.py": (
        "TODO(legacy-retirement): remove when the direct-OAuth path is deleted (after subscription enablement; see decision record §3). "
        "The setup wizard skips prompting for CLAUDE_CODE_OAUTH_TOKEN because "
        "Claude Code sets it."
    ),
    "hermes_cli/agent_import.py": (
        "Not a boundary violation: `hermes import-agent claude-code` copies a "
        "user's own agent config (skills, memory, settings) out of ~/.claude. "
        "It reads no credential and performs no inference."
    ),
    "tools/environments/local.py": (
        "Not a boundary violation: the terminal sandbox's env blocklist "
        "deliberately does NOT strip CLAUDE_CODE_OAUTH_TOKEN, so an "
        "agent-spawned `claude` CLI keeps using the user's own login instead "
        "of falling through and clearing their credentials (#55878). This is "
        "passthrough of the user's environment, not Hermes reading a token."
    ),
}

# ── Rules ───────────────────────────────────────────────────────────────────

_CREDENTIAL_ALLOW = (
    "agent/anthropic_adapter.py",
    "agent/credential_pool.py",
    "agent/credential_sources.py",
    "agent/credential_persistence.py",
    "agent/auxiliary_client.py",
    "hermes_cli/agent_import.py",
    "hermes_cli/auth.py",
    "hermes_cli/config.py",
    "hermes_cli/credential_lifecycle.py",
    "hermes_cli/main.py",
    "hermes_cli/models.py",
    "hermes_cli/web_server.py",
    "run_agent.py",
    "tools/environments/local.py",
)

_OAUTH_INFERENCE_ALLOW = (
    "agent/anthropic_adapter.py",
    "agent/account_usage.py",
    "agent/auxiliary_client.py",
    "hermes_cli/auth.py",
    "hermes_cli/doctor.py",
    "hermes_cli/web_server.py",
)

_SPOOF_ALLOW = (
    "agent/anthropic_adapter.py",
    "agent/account_usage.py",
    "agent/agent_init.py",
    "agent/auxiliary_client.py",
    "hermes_cli/doctor.py",
    "run_agent.py",
)

_OAUTH_ENV_ALLOW = (
    "agent/anthropic_adapter.py",
    "agent/credential_pool.py",
    "hermes_cli/auth.py",
    "hermes_cli/config.py",
    "hermes_cli/setup.py",
    "hermes_cli/web_server.py",
    "tools/environments/local.py",
)

RULES: list[Rule] = [
    # ── Rule 1: credential access ───────────────────────────────────────────
    Rule(
        name="claude-credential-file",
        pattern=re.compile(
            r"\.credentials\.json"
            r"|\.claude/\.credentials"
            r"|~/\.claude\.json"
            r"|\.anthropic/"
        ),
        message=(
            "references a Claude/Anthropic credential store. The Agent SDK owns "
            "those files — Hermes never reads, writes, or deletes them."
        ),
        allowlist=_CREDENTIAL_ALLOW,
    ),
    Rule(
        name="claude-keychain-entry",
        pattern=re.compile(r"Claude Code-credentials"),
        message=(
            "touches the Claude Code macOS keychain entry. Use `claude auth "
            "login` / `claude auth logout` instead."
        ),
        allowlist=_CREDENTIAL_ALLOW,
    ),
    # ── Rule 2: direct OAuth inference ──────────────────────────────────────
    # api.anthropic.com by itself is fine (the API-key provider's /v1/models
    # and /v1/messages). The violation is an Anthropic endpoint on the same
    # line as a Claude subscription token.
    Rule(
        name="anthropic-endpoint-with-claude-token",
        pattern=re.compile(
            r"api\.anthropic\.com|claude\.ai/oauth|platform\.claude\.com/v1/"
        ),
        # Symbols that denote a CLAUDE SUBSCRIPTION credential specifically —
        # not "any Anthropic credential". Broader names (resolve_anthropic_token,
        # generic *oauth_token*) also cover the legitimate API-key path and
        # would flag every module that merely knows Anthropic's hostname.
        require_also=re.compile(
            r"CLAUDE_CODE_OAUTH_TOKEN"
            r"|read_claude_code_credentials"
            r"|_write_claude_code_credentials"
            r"|refresh_anthropic_oauth"
            r"|Claude Code-credentials"
        ),
        scope="file",
        message=(
            "sends a Claude OAuth/subscription token to an Anthropic endpoint. "
            "Subscription inference must go through claude-agent-sdk. (An "
            "Anthropic endpoint on its own is fine — the API-key provider's "
            "/v1/models and /v1/messages calls are legitimate.)"
        ),
        allowlist=_OAUTH_INFERENCE_ALLOW,
    ),
    # ── Rule 3: identity spoofing ───────────────────────────────────────────
    Rule(
        name="claude-code-user-agent",
        pattern=re.compile(r"""claude-code/(?:\d|\{|\$|"|')"""),
        message=(
            "sets a claude-code/<version> user-agent. Hermes must identify as "
            "Hermes; the SDK sets its own identity."
        ),
        allowlist=_SPOOF_ALLOW,
    ),
    Rule(
        name="claude-code-cli-app-header",
        pattern=re.compile(r"""["']x-app["']\s*:\s*["']cli["']"""),
        message="sets the Claude Code `x-app: cli` header.",
        allowlist=_SPOOF_ALLOW,
    ),
    Rule(
        name="claude-code-beta-flag",
        pattern=re.compile(r"claude-code-\d{8}|oauth-\d{4}-\d{2}-\d{2}"),
        message="sends a Claude Code OAuth-only beta flag.",
        allowlist=_SPOOF_ALLOW,
    ),
    Rule(
        name="claude-code-identity-rewrite",
        pattern=re.compile(
            r"""["'](?:Hermes(?:\s+Agent)?|Nous(?:\s+Research)?)["']\s*,\s*["']"""
            r"""(?:Claude Code|Anthropic)["']"""
        ),
        message=(
            "rewrites Hermes/Nous branding into Claude Code/Anthropic. Hermes "
            "does not impersonate another client to change how it is billed."
        ),
        allowlist=_SPOOF_ALLOW,
    ),
    Rule(
        name="claude-code-system-prompt",
        pattern=re.compile(r"You are Claude Code"),
        message="injects Claude Code's system-prompt identity.",
        allowlist=_SPOOF_ALLOW,
    ),
    # ── Rule 4: CLAUDE_CODE_OAUTH_TOKEN reads ───────────────────────────────
    Rule(
        name="claude-oauth-token-env",
        pattern=re.compile(r"CLAUDE_CODE_OAUTH_TOKEN"),
        message=(
            "reads CLAUDE_CODE_OAUTH_TOKEN. The Claude subscription provider "
            "holds no credential — the SDK resolves auth itself."
        ),
        allowlist=_OAUTH_ENV_ALLOW,
    ),
]


def _iter_source_files() -> Iterable[Path]:
    """Yield every scannable Hermes source file."""
    for root_name in SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        if root.is_file():
            if root.suffix in SCANNED_SUFFIXES:
                yield root
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            yield path


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def scan(rules: list[Rule]) -> int:
    """Populate ``rule.hits`` and return the number of unallowed violations."""
    violations = 0
    for path in _iter_source_files():
        rel = _rel(path)
        # This script names every pattern it forbids; scanning it would be a
        # guaranteed self-hit.
        if rel == "scripts/check_claude_boundary.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for rule in rules:
            if not rule.pattern.search(text):
                continue
            if (
                rule.scope == "file"
                and rule.require_also is not None
                and not rule.require_also.search(text)
            ):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not rule.pattern.search(line):
                    continue
                if (
                    rule.scope == "line"
                    and rule.require_also is not None
                    and not rule.require_also.search(line)
                ):
                    continue
                if SUPPRESS_MARKER.search(line):
                    continue
                allowed = any(rel == a or rel.startswith(a) for a in rule.allowlist)
                rule.hits.append((rel, lineno, line.strip()))
                if not allowed:
                    violations += 1
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print allow-listed hits (the PR4 removal backlog)",
    )
    args = parser.parse_args(argv)

    rules = RULES
    scan(rules)

    failed = False
    for rule in rules:
        unallowed = [
            (rel, lineno, line)
            for rel, lineno, line in rule.hits
            if not any(rel == a or rel.startswith(a) for a in rule.allowlist)
        ]
        if unallowed:
            failed = True
            print(f"FAIL [{rule.name}]: {rule.message}")
            for rel, lineno, line in unallowed:
                print(f"    {rel}:{lineno}: {line}")
            print()

    if args.verbose:
        for rule in rules:
            allowed = [
                (rel, lineno, line)
                for rel, lineno, line in rule.hits
                if any(rel == a or rel.startswith(a) for a in rule.allowlist)
            ]
            if allowed:
                print(f"allow-listed [{rule.name}]:")
                for rel, lineno, line in allowed:
                    print(f"    {rel}:{lineno}: {line}")
                print()

    if failed:
        print(
            "Claude boundary check FAILED.\n"
            "Hermes must not read Claude credentials, send a subscription token "
            "to Anthropic directly, or impersonate Claude Code. Route "
            "subscription inference through claude-agent-sdk, or add a "
            "justified allow-list entry to this script if the code is part of "
            "the legacy path being retired."
        )
        return 1

    print("Claude boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
