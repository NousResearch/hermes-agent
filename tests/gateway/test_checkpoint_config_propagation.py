"""Regression tests for checkpoint config propagation to gateway agents.

``checkpoints: {enabled: true}`` in config.yaml was honored by the CLI lane
(``HermesCLI.__init__`` reads the block and passes ``checkpoints_enabled`` to
AIAgent) but silently ignored by every gateway-spawned agent:
``gateway/run.py`` never read the block, so AIAgent's ``checkpoints_enabled``
default (False) always won and filesystem checkpoints stayed OFF in
long-running gateways even when the user's config promised them.

``_resolve_checkpoint_kwargs`` mirrors the CLI semantics:
    mapping block -> its ``enabled``/limits; bool shorthand -> enabled flag;
    absent/malformed block -> disabled with default limits.
"""

import importlib
import sys
import textwrap

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a writable config.yaml and a clean module cache.

    Mirrors tests/gateway/test_max_tokens_propagation.py: re-import
    ``hermes_cli`` / ``gateway`` so each config write is read fresh, and
    restore the pre-test module snapshot on teardown so sibling test files
    keep their import-time mocks.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _saved = {
        k: v
        for k, v in sys.modules.items()
        if k.startswith(("hermes_cli", "gateway"))
    }

    def write_cfg(body: str) -> None:
        (hermes_home / "config.yaml").write_text(textwrap.dedent(body))

    def fresh_gateway():
        for mod in list(sys.modules.keys()):
            if mod.startswith(("hermes_cli", "gateway")):
                del sys.modules[mod]
        return importlib.import_module("gateway.run")

    try:
        yield write_cfg, fresh_gateway
    finally:
        for k in list(sys.modules.keys()):
            if k.startswith(("hermes_cli", "gateway")):
                del sys.modules[k]
        sys.modules.update(_saved)


# ---------------------------------------------------------------------------
# Pure resolver semantics (explicit config dicts)
# ---------------------------------------------------------------------------


def _resolver():
    from gateway.run import _resolve_checkpoint_kwargs

    return _resolve_checkpoint_kwargs


def test_config_on():
    kw = _resolver()({"checkpoints": {"enabled": True}})
    assert kw["checkpoints_enabled"] is True
    assert kw["checkpoint_max_snapshots"] == 20
    assert kw["checkpoint_max_total_size_mb"] == 500
    assert kw["checkpoint_max_file_size_mb"] == 10


def test_config_off():
    kw = _resolver()({"checkpoints": {"enabled": False}})
    assert kw["checkpoints_enabled"] is False


def test_config_absent():
    """No checkpoints block -> disabled (AIAgent's own default preserved)."""
    kw = _resolver()({})
    assert kw["checkpoints_enabled"] is False
    assert kw["checkpoint_max_snapshots"] == 20


def test_bool_shorthand():
    """``checkpoints: true`` shorthand matches the CLI lane's bool handling."""
    assert _resolver()({"checkpoints": True})["checkpoints_enabled"] is True
    assert _resolver()({"checkpoints": False})["checkpoints_enabled"] is False


def test_malformed_block_disables():
    """A non-dict, non-bool block must not crash agent construction."""
    kw = _resolver()({"checkpoints": "yes please"})
    assert kw["checkpoints_enabled"] is False
    assert kw["checkpoint_max_snapshots"] == 20


def test_limits_pass_through():
    kw = _resolver()(
        {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 7,
                "max_total_size_mb": 123,
                "max_file_size_mb": 4,
            }
        }
    )
    assert kw == {
        "checkpoints_enabled": True,
        "checkpoint_max_snapshots": 7,
        "checkpoint_max_total_size_mb": 123,
        "checkpoint_max_file_size_mb": 4,
    }


def test_kwargs_match_aiagent_signature():
    """Every resolved key must be a real AIAgent constructor parameter."""
    import inspect

    from run_agent import AIAgent

    params = set(inspect.signature(AIAgent.__init__).parameters)
    kw = _resolver()({"checkpoints": {"enabled": True}})
    missing = set(kw) - params
    assert not missing, f"resolved kwargs not accepted by AIAgent: {missing}"


# ---------------------------------------------------------------------------
# Real config path (config.yaml -> _load_gateway_config -> resolver)
# ---------------------------------------------------------------------------


def test_enabled_via_real_config_load(isolated_home):
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        checkpoints:
          enabled: true
          max_snapshots: 5
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_checkpoint_kwargs()
    assert kw["checkpoints_enabled"] is True
    assert kw["checkpoint_max_snapshots"] == 5


def test_absent_via_real_config_load(isolated_home):
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_checkpoint_kwargs()
    assert kw["checkpoints_enabled"] is False
