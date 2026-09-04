"""Hermes dotenv API with process-launch credential authority."""

from __future__ import annotations

import importlib
import os
import sys

_DESKTOP_MARKER_ENV = "HERMES_DESKTOP"
_DESKTOP_SESSION_TOKEN_ENV = "HERMES_DASHBOARD_SESSION_TOKEN"

# Snapshot the launch coordinates before importing dotenv mechanics. A later
# dotenv assignment to the same names is configuration, not launch provenance.
_launch_marker = os.environ.get(_DESKTOP_MARKER_ENV)
_launch_token = os.environ.get(_DESKTOP_SESSION_TOKEN_ENV)
_core = importlib.import_module("hermes_cli._env_loader_core")

# Keep the snapshot monotonic across unusual remove-and-reimport sequences.
if "_DESKTOP_LAUNCH_CREDENTIAL" not in vars(_core):
    _core._DESKTOP_LAUNCH_CREDENTIAL = (
        ("1", _launch_token) if _launch_marker == "1" and _launch_token else None
    )

# Likewise, retain one unwrapped core sink rather than stacking wrappers if the
# public module is deliberately re-imported during tests or plugin discovery.
if "_DESKTOP_ORIGINAL_LOAD_DOTENV_WITH_FALLBACK" not in vars(_core):
    _core._DESKTOP_ORIGINAL_LOAD_DOTENV_WITH_FALLBACK = (
        _core._load_dotenv_with_fallback
    )
_original_load_dotenv_with_fallback = (
    _core._DESKTOP_ORIGINAL_LOAD_DOTENV_WITH_FALLBACK
)


def _load_dotenv_with_fallback(path, *, override: bool) -> None:
    """Load one dotenv layer without changing marked launch authority."""
    launch_credential = _core._DESKTOP_LAUNCH_CREDENTIAL if override else None
    try:
        _original_load_dotenv_with_fallback(path, override=override)
    finally:
        if launch_credential is not None:
            marker, token = launch_credential
            os.environ[_DESKTOP_MARKER_ENV] = marker
            os.environ[_DESKTOP_SESSION_TOKEN_ENV] = token


# The core owns dotenv parsing and the public API. Replace only its common
# override-capable sink, then expose that module under the established import
# path so callers and tests retain one shared mutable module namespace.
_core._load_dotenv_with_fallback = _load_dotenv_with_fallback
sys.modules[__name__] = _core
