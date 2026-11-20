"""resolve_provider auto-detection must read provider keys through the
profile secret scope under multiplex (#86917).

A secondary profile whose config uses ``model.provider: auto`` and whose
API key lives only in its profile ``.env`` (installed per-turn as the
secret scope) failed with "No LLM provider configured": the auto path
read keys with bare ``os.getenv``, which under multiplex holds the
DEFAULT profile's values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.secret_scope as ss
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _install_profile(tmp_path: Path, provider: str, env_text: str) -> Path:
    """Create a profile home with a model section and a .env."""
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  provider: {provider}\n", encoding="utf-8"
    )
    (home / ".env").write_text(env_text, encoding="utf-8")
    return home


def test_auto_resolution_sees_profile_scoped_key(tmp_path):
    """The scoped provider key must be visible to auto-detection.

    Before the fix: the DEEPSEEK_API_KEY lived only in the scope, bare
    os.getenv found nothing, and auto-resolution reported no provider.
    """
    from hermes_cli.auth import resolve_provider

    home = _install_profile(tmp_path, "auto", "DEEPSEEK_API_KEY=sk-scoped\n")

    ss.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(home))
    try:
        scope_token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
        try:
            assert resolve_provider("auto") == "deepseek"
        finally:
            ss.reset_secret_scope(scope_token)
    finally:
        reset_hermes_home_override(home_token)
        ss.set_multiplex_active(False)


def test_auto_resolution_falls_back_to_os_environ_when_unscoped(monkeypatch):
    """Single-profile / CLI path unchanged: exported env keys still win."""
    from hermes_cli.auth import resolve_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert resolve_provider("auto") == "openrouter"


def test_auto_resolution_honors_explicit_config_provider(tmp_path):
    """An explicit config provider still wins over env-key detection."""
    from hermes_cli.auth import resolve_provider

    home = _install_profile(tmp_path, "opencode-go", "OPENCODE_GO_API_KEY=sk\n")

    ss.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(home))
    try:
        scope_token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
        try:
            assert resolve_provider("auto") == "opencode-go"
        finally:
            ss.reset_secret_scope(scope_token)
    finally:
        reset_hermes_home_override(home_token)
        ss.set_multiplex_active(False)
