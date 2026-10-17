"""Signature-tolerant dispatch for third-party platform plugins (#97065).

Core's documented contract is ``PlatformEntry.setup_fn() -> None`` and
``BasePlatformAdapter.connect(*, is_reconnect: bool = False)``, but plugins
written against older/looser contracts register ``setup_fn(config)``
(keet-platform) or override ``connect()`` without the keyword. Core called
both with the documented shape, so a plugin TypeError aborted
``hermes gateway setup`` and left the platform stuck in ``retrying``.
"""

import asyncio

import pytest

from gateway.plugin_dispatch import connect_adapter, invoke_setup_fn


# ── invoke_setup_fn ──────────────────────────────────────────────────────────


def test_zero_arg_setup_fn_is_called_with_no_arguments():
    """The documented ``setup_fn()`` contract keeps receiving nothing."""
    calls = []

    def setup():
        calls.append("setup")

    invoke_setup_fn(setup, lambda: {"key": "keet"})
    assert calls == ["setup"]


def test_setup_fn_requiring_config_receives_it():
    """A legacy ``setup_fn(config)`` gets the config instead of a TypeError (#97065)."""
    seen = []

    def setup(config):
        seen.append(config)

    invoke_setup_fn(setup, lambda: {"key": "keet"})
    assert seen == [{"key": "keet"}]


def test_setup_fn_with_keyword_only_config_receives_it_by_name():
    """``setup_fn(*, config)`` is satisfied as a keyword argument."""
    seen = []

    def setup(*, config):
        seen.append(config)

    invoke_setup_fn(setup, lambda: "cfg")
    assert seen == ["cfg"]


def test_setup_fn_with_defaulted_config_is_called_with_no_arguments():
    """A default means the callable doesn't need the value — don't hand it over."""
    calls = []

    def setup(config=None):
        calls.append(config)

    invoke_setup_fn(setup, lambda: "cfg")
    assert calls == [None]


def test_config_factory_is_not_called_for_a_zero_arg_setup_fn():
    """No config is built when the signature doesn't ask for one."""
    calls = []

    def setup():
        calls.append("setup")

    def _factory():
        raise AssertionError("config_factory must not run for setup_fn()")

    invoke_setup_fn(setup, _factory)
    assert calls == ["setup"]


# ── connect_adapter ──────────────────────────────────────────────────────────


def test_connect_forwards_is_reconnect_when_accepted():
    """Adapters on the current contract still get ``is_reconnect`` (#46621)."""
    seen = []

    class Adapter:
        async def connect(self, *, is_reconnect: bool = False):
            seen.append(is_reconnect)
            return True

    assert asyncio.run(connect_adapter(Adapter(), is_reconnect=True)) is True
    assert seen == [True]


def test_connect_omits_is_reconnect_for_a_bare_connect():
    """A legacy ``connect()`` connects instead of raising TypeError (#97065)."""
    calls = []

    class Adapter:
        async def connect(self):
            calls.append("bare")
            return True

    assert asyncio.run(connect_adapter(Adapter(), is_reconnect=True)) is True
    assert calls == ["bare"]


def test_connect_forwards_is_reconnect_through_var_keywords():
    """``connect(**kwargs)`` still receives the keyword."""
    seen = []

    class Adapter:
        async def connect(self, **kwargs):
            seen.append(kwargs)
            return True

    assert asyncio.run(connect_adapter(Adapter(), is_reconnect=False)) is True
    assert seen == [{"is_reconnect": False}]


def test_connect_omits_the_kwarg_for_an_uninspectable_signature():
    """Unknown signature → the bare call (mirrors ``_accepts_kwarg(..., unknown=False)``).

    ``bool`` has no introspectable signature and rejects keyword arguments, so
    ``bool()`` → False proves the keyword was withheld.
    """

    class Adapter:
        connect = bool

    assert connect_adapter(Adapter(), is_reconnect=True) is False


# ── _legacy_setup_config ────────────────────────────────────────────────────
# The wizard must hand a legacy setup_fn(config) the same block the gateway will
# load on restart, whichever spelling the user wrote it in.


@pytest.fixture()
def _writable_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home, yaml_text):
    (home / "config.yaml").write_text(yaml_text, encoding="utf-8")


@pytest.mark.parametrize("yaml_text", [
    # The spelling `hermes config set gateway.platforms.keet...` writes.
    "gateway:\n  platforms:\n    keet:\n      enabled: true\n      token: TOK\n",
    # The top-level spelling.
    "platforms:\n  keet:\n    enabled: true\n    token: TOK\n",
    # Both, split across spellings — the merge must still find them.
    "gateway:\n  platforms:\n    keet:\n      enabled: true\nplatforms:\n  keet:\n      token: TOK\n",
])
def test_legacy_setup_config_sees_the_block_the_gateway_will_load(_writable_home, yaml_text):
    """A legacy setup_fn gating on config.enabled must not see a platform the user
    configured through `gateway.platforms.<key>` as disabled (#97065)."""
    from hermes_cli.gateway_setup_wizard import _legacy_setup_config

    _write_cfg(_writable_home, yaml_text)
    cfg = _legacy_setup_config({"key": "keet"})
    assert cfg.enabled is True
    assert cfg.token == "TOK"


def test_legacy_setup_config_defaults_when_the_platform_is_unset(_writable_home):
    from hermes_cli.gateway_setup_wizard import _legacy_setup_config

    _write_cfg(_writable_home, "model:\n  default: x\n")
    cfg = _legacy_setup_config({"key": "keet"})
    assert cfg.enabled is False
