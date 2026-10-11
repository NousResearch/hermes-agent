"""Explicit cache routing for fresh, single-query CLI invocations (#136359).

Never reuse a gateway session key: it also identifies memory and persisted peers.
"""

import hashlib
import json
import sys
from pathlib import Path


def normalize_cache_scope(value: str) -> str:
    """Validate a local label; only its namespaced digest leaves the process."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--cache-scope must be a nonempty label")
    if len(value.encode("utf-8")) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("--cache-scope must be at most 256 UTF-8 bytes with no control characters")
    return value.strip()


def validate_cli_cache_scope(cache_scope, query, quiet, oneshot, resume, *, use_tui=False):
    """Reject accidental sharing on interactive or resumed conversations."""
    if cache_scope is None:
        return None
    label = normalize_cache_scope(cache_scope)
    if resume:
        raise ValueError("--cache-scope cannot be combined with --resume/--continue")
    if use_tui or not query or (not quiet and not oneshot and sys.stdin.isatty()):
        raise ValueError("--cache-scope requires a fresh single query (-Q or --oneshot on a TTY)")
    return label


def validate_chat_cache_scope(args, use_tui):
    """Validate before startup, session lookup, or reading --query-file."""
    try:
        return validate_cache_scope_args(args, use_tui=use_tui)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def validate_cache_scope_args(args, *, use_tui=False):
    """Shared validation for classic and thin gateway CLI launches."""
    return validate_cli_cache_scope(
        getattr(args, "cache_scope", None),
        getattr(args, "query", None) or getattr(args, "q", None) or
        getattr(args, "query_file", None) or getattr(args, "oneshot", None),
        getattr(args, "quiet", False) or getattr(args, "output_format", "text") == "stream-json",
        getattr(args, "oneshot_exit", False) or bool(getattr(args, "oneshot", None)),
        getattr(args, "resume", None) or getattr(args, "continue_last", None),
        use_tui=use_tui,
    )


def cli_cache_scope(label, cwd):
    """Only a profile/workspace-namespaced digest crosses the gateway boundary."""
    from hermes_constants import get_hermes_home

    if label is None:
        return None
    identity = [str(get_hermes_home().resolve()), str(Path(cwd).resolve()), label]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]
    return f"cli_{digest}"


def configure_cli_cache_scope(cli, label):
    """Namespace the opt-in by profile and workspace, never by physical session."""
    cli._cli_prompt_cache_scope = cli_cache_scope(label, Path.cwd())


def gateway_cache_policy(args, cwd):
    """The optional creation field; never silently drop it on an older owner."""
    scope = cli_cache_scope(getattr(args, "cache_scope", None), cwd)
    return {"cache_scope": scope} if scope is not None else {}
