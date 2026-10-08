"""Explicit, ordered context.external_files sources for a new prompt snapshot.

Paths retain #53766's user-home-relative contract. The configuration, safety policy and read budget
belong to the agent's profile; the process's launch profile must never supply another bot's rules.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Iterator

from agent.context_file_io import read_context_file

logger = logging.getLogger(__name__)
_BARE_ENV_REF = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_PERCENT_ENV_REF = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")


@dataclass(frozen=True)
class ExternalContextFile:
    label: str
    path: Path
    content: str
    identity: tuple | None
    status: str


@contextmanager
def external_context_scope(home_override: Path | None = None) -> Iterator[None]:
    """Bind an explicitly supplied profile for config, env references, budgets and read guards."""
    if home_override is None:
        yield
        return
    from hermes_constants import (
        get_hermes_home, hermes_home_key, reset_hermes_home_override, set_hermes_home_override,
    )
    from agent.secret_scope import (
        build_profile_secret_scope, current_secret_scope, current_secret_scope_home,
        reset_secret_scope, set_secret_scope,
    )
    previous_home = get_hermes_home()
    scope_home = current_secret_scope_home()
    matching_scope = current_secret_scope() is not None and hermes_home_key(
        scope_home or previous_home) == hermes_home_key(home_override)
    home_token = set_hermes_home_override(str(home_override))
    secret_token = None
    try:
        if not matching_scope:
            secret_token = set_secret_scope(build_profile_secret_scope(home_override), profile_home=str(home_override))
        yield
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def safe_context_metadata(value: str, fallback: str) -> str:
    """Keep filenames on one line, with the same threat policy as their content."""
    from tools.threat_patterns import scan_for_threats
    escaped = "".join(f"\\x{ord(c):02x}" if ord(c) < 32 or ord(c) == 127 else c for c in value)
    return fallback if not escaped or scan_for_threats(escaped, scope="context") else escaped


def _expanded_path(value: str) -> Path:
    from hermes_cli.config import _env_ref_lookup
    # ${VAR} was expanded by the canonical config loader. An unresolved placeholder stays unresolved;
    # os.path.expandvars here would fall back to another profile's process environment.
    def expand(match: re.Match) -> str:
        found = _env_ref_lookup(match.group(1))
        return match.group(0) if found is None else found

    expanded = _BARE_ENV_REF.sub(expand, value.strip())
    if os.name == "nt":
        expanded = _PERCENT_ENV_REF.sub(expand, expanded)
    if expanded == "~":
        return Path.home()
    path = Path.home() / expanded[2:] if expanded.startswith(("~/", "~\\")) else Path(os.path.expanduser(expanded))
    return path if path.is_absolute() else Path.home() / path


def _configured_paths() -> list[Path]:
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly()
    context = config.get("context")
    values = context.get("external_files") if isinstance(context, dict) else None
    if not isinstance(values, list):
        return []
    paths = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            paths.append(_expanded_path(value))
        except (OSError, ValueError, RuntimeError):
            logger.debug("Could not expand configured context path", exc_info=True)
    return paths


def _load_source(path: Path, timeout: float) -> ExternalContextFile:
    label = safe_context_metadata(str(path), "external context file")
    loaded = read_context_file(path, timeout, guarded=True)
    return ExternalContextFile(label, path, loaded.content, loaded.identity, loaded.status)


def load_external_context_files(home_override: Path | None = None) -> list[ExternalContextFile]:
    """Inspect each configured source in declaration order. Rendering owns scan/cap/dedup decisions."""
    from agent.prompt_builder import _get_context_file_read_timeout
    with external_context_scope(home_override):
        try:
            paths = _configured_paths()
        except (OSError, ValueError, RuntimeError):
            logger.debug("Could not load context.external_files configuration", exc_info=True)
            return []
        timeout = _get_context_file_read_timeout()
        return [_load_source(path, timeout) for path in paths]
