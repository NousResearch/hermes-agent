"""Scoped Buzz ``check_requirements`` must see every config shape the runtime loader sees (#125985).

``_profile_buzz_extra`` (added for #98738) read the profile's buzz section only from
``gateway.platforms.buzz``. But the runtime platform-config loader
(``gateway/config_loader.py::platform_section``) resolves a section from a top-level
``buzz:`` block, ``gateway.platforms.buzz``, and ``platforms.buzz`` — and every documented
shape (and every profile Hermes itself writes) nests at top-level ``platforms.buzz``. Inside
a multiplexed secondary profile scope the gate therefore returned ``{}`` and the adapter
never started (``Platform 'Buzz' requirements not met``) even though
``platforms.buzz.extra.relay_url`` + ``BUZZ_PRIVATE_KEY`` were fully configured.

These tests drive the real ``_profile_buzz_extra`` / ``check_requirements`` under a real
installed scope (``set_multiplex_active`` + ``set_secret_scope``) with a real scoped
``HERMES_HOME`` config.yaml — mirroring the secondary-profile startup path — and assert the
section resolution matches the runtime loader's precedence for every shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz = load_plugin_adapter("buzz")

_profile_buzz_extra = _buzz._profile_buzz_extra
check_requirements = _buzz.check_requirements

_RELAY = "wss://relay.example"
_KEY = "a" * 63  # hex private key material; never a real secret


def _install_scope(tmp_path: Path, config_yaml: str, *, secrets: dict | None = None):
    """Install the secondary-profile runtime scope the adapter construction runs in."""
    from agent import secret_scope as ss

    home = tmp_path / "profile-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(secrets or {}, profile_home=str(home))
    return home, token


@pytest.fixture(autouse=True)
def _reset_scope():
    from agent import secret_scope as ss

    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture(autouse=True)
def _scoped_home(tmp_path, monkeypatch):
    """Point ``get_hermes_home()`` at the secondary profile's home via the context override."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "profile-home"
    home.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(home)
    yield home
    reset_hermes_home_override(token)


@pytest.mark.parametrize(
    "config_yaml",
    [
        # Documented shape (and what Hermes itself writes): top-level platforms.buzz.
        'platforms:\n  buzz:\n    enabled: true\n    extra:\n      relay_url: "wss://relay.example"\n',
        # Legacy top-level buzz: block (a top-level <name>: key wins in platform_section).
        'buzz:\n  enabled: true\n  extra:\n    relay_url: "wss://relay.example"\n',
        # Nested gateway.platforms.buzz — the only shape the old code saw (the #125985 workaround).
        'gateway:\n  platforms:\n    buzz:\n      enabled: true\n      extra:\n        relay_url: "wss://relay.example"\n',
    ],
    ids=["toplevel-platforms-buzz", "toplevel-buzz-key", "nested-gateway-platforms-buzz"],
)
def test_scoped_gate_reads_every_runtime_section_shape(config_yaml, tmp_path):
    """The scoped gate resolves the buzz section with the runtime loader's precedence."""
    _install_scope(tmp_path, config_yaml)
    assert _profile_buzz_extra().get("relay_url") == _RELAY


def test_scoped_requirements_gate_passes_with_documented_shape(tmp_path):
    """End-to-end gate: platforms.buzz.extra.relay_url + scoped key satisfy check_requirements."""
    _install_scope(
        tmp_path,
        'platforms:\n  buzz:\n    enabled: true\n    extra:\n      relay_url: "wss://relay.example"\n',
        secrets={"BUZZ_PRIVATE_KEY": _KEY},
    )
    assert check_requirements() is True


def test_precedence_matches_runtime_loader(tmp_path):
    """Invariant: when both nested shapes exist, the gate sees what ``platform_section`` sees.

    ``platform_section`` prefers ``gateway.platforms.buzz`` over ``platforms.buzz`` among the
    nested sources (top-level ``buzz:`` would win over both, covered above).
    """
    _install_scope(
        tmp_path,
        'gateway:\n  platforms:\n    buzz:\n      extra:\n        relay_url: "wss://nested"\n'
        'platforms:\n  buzz:\n    extra:\n      relay_url: "wss://toplevel"\n',
    )
    assert _profile_buzz_extra().get("relay_url") == "wss://nested"


def test_unconfigured_profile_still_fails_closed(tmp_path):
    """No buzz section anywhere: the gate keeps failing closed ({} — #98738 invariant)."""
    _install_scope(tmp_path, 'model:\n  provider: zai\n')
    assert _profile_buzz_extra() == {}
    assert check_requirements() is False


def test_non_dict_section_fails_closed(tmp_path):
    """A malformed (non-dict) buzz section must not crash the gate."""
    _install_scope(tmp_path, 'platforms:\n  buzz: not-a-dict\n')
    assert _profile_buzz_extra() == {}
