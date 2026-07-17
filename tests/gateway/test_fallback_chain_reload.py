"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_refresh_fallback_model_clears_when_config_removed(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=[{"provider": "stale", "model": "x"}],
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    assert bound() is not None

    cfg.write_text("model:\n  provider: nvidia\n")
    assert bound() is None
    assert runner._fallback_model is None


def test_refresh_fallback_model_keeps_last_known_good_on_read_failure(
    tmp_path, monkeypatch,
):
    """A transient config.yaml read/parse failure (user mid-edit, non-atomic
    write) must NOT wipe the last known-good chain — only a successful read
    that genuinely lacks the key clears it."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    good = bound()
    assert good == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]

    # Simulate a mid-edit torn write: invalid YAML.
    cfg.write_text("fallback_providers:\n  - provider: [unclosed\n")
    assert bound() == good
    assert runner._fallback_model == good


def test_refresh_fallback_model_reads_active_profile_config(tmp_path, monkeypatch):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    default_home = tmp_path / "default"
    secondary_home = tmp_path / "secondary"
    default_home.mkdir()
    secondary_home.mkdir()
    (default_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: default-provider\n"
        "    model: default-model\n",
        encoding="utf-8",
    )
    (secondary_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: secondary-provider\n"
        "    model: secondary-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)

    runner = SimpleNamespace(
        _fallback_model=None,
        _fallback_models_by_home={str(default_home): None},
    )
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)

    with gateway_run._profile_runtime_scope(secondary_home):
        assert bound() == [
            {"provider": "secondary-provider", "model": "secondary-model"}
        ]


def test_refresh_fallback_model_keeps_last_known_good_per_profile(
    tmp_path, monkeypatch
):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    default_home = tmp_path / "default"
    secondary_home = tmp_path / "secondary"
    default_home.mkdir()
    secondary_home.mkdir()
    (default_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: default-provider\n"
        "    model: default-model\n",
        encoding="utf-8",
    )
    secondary_config = secondary_home / "config.yaml"
    secondary_config.write_text(
        "fallback_providers:\n"
        "  - provider: secondary-provider\n"
        "    model: secondary-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)

    runner = SimpleNamespace(
        _fallback_model=[{"provider": "default-provider", "model": "default-model"}],
        _fallback_models_by_home={
            str(default_home): [
                {"provider": "default-provider", "model": "default-model"}
            ]
        },
    )
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)

    with gateway_run._profile_runtime_scope(secondary_home):
        secondary_chain = bound()
        secondary_config.write_text("fallback_providers: [", encoding="utf-8")
        assert bound() == secondary_chain

    assert runner._fallback_model == [
        {"provider": "default-provider", "model": "default-model"}
    ]


def test_refresh_fallback_model_serializes_load_and_lkg_publication(monkeypatch):
    from gateway.run import GatewayRunner

    runner = SimpleNamespace(
        _fallback_model=None,
        _fallback_models_by_home={},
        _fallback_refresh_lock=threading.RLock(),
    )
    entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_load():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            entered.set()
        release.wait(timeout=1)
        with state_lock:
            active -= 1
        return [{"provider": "test", "model": "model"}]

    monkeypatch.setattr("gateway.run.load_fallback_chain_strict", fake_load)
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    first = threading.Thread(target=bound)
    second = threading.Thread(target=bound)
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert max_active == 1


def test_apply_fallback_chain_updates_primary_agent():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
    )
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, chain)

    assert agent._fallback_chain == chain
    assert agent._fallback_model == chain[0]
    assert agent._fallback_index == 0


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True


def test_apply_fallback_chain_updates_after_cooldown_expires():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "deepseek", "model": "old"}],
        _fallback_model={"provider": "deepseek", "model": "old"},
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() - 1,
    )
    new_chain = [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, new_chain)

    assert agent._fallback_chain == new_chain
    assert agent._fallback_model == new_chain[0]
    # Activated agents keep their index; restore_primary_runtime owns reset.
    assert agent._fallback_index == 1


def test_apply_fallback_chain_clears_unavailable_memo_on_content_change():
    """A config edit must drop the session-scoped unavailability memo so a
    re-configured entry (credentials added mid-uptime) is retried instead of
    staying suppressed for the cached agent's lifetime."""
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "deepseek", "model": "old"}],
        _fallback_model={"provider": "deepseek", "model": "old"},
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys={("deepseek", "old", "")},
    )
    new_chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, new_chain)

    assert agent._fallback_chain == new_chain
    assert agent._unavailable_fallback_keys == set()


def test_apply_fallback_chain_keeps_unavailable_memo_when_unchanged():
    """The per-message no-op refresh must NOT clear the memo — it exists to
    rate-limit repeated activation attempts against dead entries."""
    from gateway.run import GatewayRunner

    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    memo = {("deepseek", "deepseek-v4-flash", "")}
    agent = SimpleNamespace(
        _fallback_chain=list(chain),
        _fallback_model=chain[0],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(memo),
    )
    GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))

    assert agent._unavailable_fallback_keys == memo


def test_apply_fallback_chain_clears_unavailable_memo_on_secret_rotation():
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.run import GatewayRunner

    chain = [
        {
            "provider": "custom:secondary",
            "model": "secondary-model",
            "key_env": "FB_KEY",
        }
    ]
    agent = SimpleNamespace(
        _fallback_chain=list(chain),
        _fallback_model=chain[0],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
    )
    token = set_secret_scope({})
    try:
        GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))
    finally:
        reset_secret_scope(token)

    agent._unavailable_fallback_keys = {
        ("custom:secondary", "secondary-model", "")
    }
    token = set_secret_scope({"FB_KEY": "rotated-profile-key"})
    try:
        GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))
    finally:
        reset_secret_scope(token)

    assert agent._unavailable_fallback_keys == set()


def test_apply_fallback_chain_clears_memo_on_openrouter_secret_rotation():
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.run import GatewayRunner

    chain = [{"provider": "openrouter", "model": "fallback-model"}]
    agent = SimpleNamespace(
        _fallback_chain=list(chain),
        _fallback_model=chain[0],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
    )
    token = set_secret_scope({})
    try:
        GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))
    finally:
        reset_secret_scope(token)

    agent._unavailable_fallback_keys = {("openrouter", "fallback-model", "")}
    token = set_secret_scope({"OPENROUTER_API_KEY": "rotated-openrouter-key"})
    try:
        GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))
    finally:
        reset_secret_scope(token)

    assert agent._unavailable_fallback_keys == set()


def test_apply_fallback_chain_compares_authoritative_snapshot_after_pruning():
    """Operational pruning alone must not invalidate unavailable-entry memoization."""
    from gateway.run import GatewayRunner

    authoritative = [
        {"provider": "custom:zenmux", "model": "zenmux-fallback"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    ]
    operational = [authoritative[1]]
    memo = {("openrouter", "anthropic/claude-sonnet-4.6", "")}
    agent = SimpleNamespace(
        _fallback_config_chain=list(authoritative),
        _fallback_chain=list(operational),
        _fallback_model=operational[0],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(memo),
    )

    GatewayRunner._apply_fallback_chain_to_agent(agent, list(authoritative))

    assert agent._fallback_config_chain == authoritative
    assert agent._fallback_chain == authoritative
    assert agent._unavailable_fallback_keys == memo


def test_load_fallback_model_static_unchanged_contract(tmp_path, monkeypatch):
    """_load_fallback_model remains a pure static reader used by refresh."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "fallback_model:\n"
        "  provider: nous\n"
        "  model: Hermes-4\n"
    )

    chain = GatewayRunner._load_fallback_model()
    assert chain == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "nous", "model": "Hermes-4"},
    ]


def test_load_fallback_model_applies_managed_overlay(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli import managed_scope

    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: user-provider\n"
        "    model: user-model\n",
        encoding="utf-8",
    )
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: managed-provider\n"
        "    model: managed-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed_dir)

    assert GatewayRunner._load_fallback_model() == [
        {"provider": "managed-provider", "model": "managed-model"}
    ]
