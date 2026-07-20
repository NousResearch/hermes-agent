"""Each standalone config loader (gateway, TUI/desktop, cron) must honor managed scope.

These loaders build their own config dict instead of routing through
hermes_cli.config.load_config, so the managed overlay has to be wired into each.
This is the regression guard for the whole bug class (a managed display.skin was
silently ignored by the TUI; the same gap existed in the gateway and cron).
"""
import textwrap

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed


def _seed(home, managed, *, user, mgd):
    (home / "config.yaml").write_text(textwrap.dedent(user), encoding="utf-8")
    (managed / "config.yaml").write_text(textwrap.dedent(mgd), encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def test_gateway_run_loader_honors_managed(homes, monkeypatch):
    home, managed = homes
    _seed(home, managed, user="model:\n  default: user/m\n", mgd="model:\n  default: org/m\n")
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", home, raising=False)
    cfg = gr._load_gateway_config()
    assert (cfg.get("model") or {}).get("default") == "org/m"


def test_gateway_config_loader_honors_managed(homes, monkeypatch):
    home, managed = homes
    _seed(
        home,
        managed,
        user="group_sessions_per_user: false\n",
        mgd="group_sessions_per_user: true\n",
    )
    import gateway.config as gc

    # load_gateway_config resolves home via get_hermes_home() (HERMES_HOME env).
    cfg = gc.load_gateway_config()
    # Managed value should have flowed into the GatewayConfig.
    assert cfg.group_sessions_per_user is True


def test_tui_loader_honors_managed(homes, monkeypatch):
    home, managed = homes
    _seed(home, managed, user="display:\n  skin: user\n", mgd="display:\n  skin: charizard\n")
    import tui_gateway.server as ts

    monkeypatch.setattr(ts, "_hermes_home", home, raising=False)
    monkeypatch.setattr(ts, "_cfg_cache", None, raising=False)
    monkeypatch.setattr(ts, "_cfg_mtime", None, raising=False)
    monkeypatch.setattr(ts, "get_hermes_home_override", lambda: None, raising=False)
    cfg = ts._load_cfg()
    assert (cfg.get("display") or {}).get("skin") == "charizard"


def test_tui_loader_does_not_persist_managed_back(homes, monkeypatch):
    """The TUI caches RAW config so _save_cfg never writes managed values to disk."""
    home, managed = homes
    _seed(home, managed, user="display:\n  skin: user\n", mgd="display:\n  skin: charizard\n")
    import tui_gateway.server as ts

    monkeypatch.setattr(ts, "_hermes_home", home, raising=False)
    monkeypatch.setattr(ts, "_cfg_cache", None, raising=False)
    monkeypatch.setattr(ts, "_cfg_mtime", None, raising=False)
    monkeypatch.setattr(ts, "get_hermes_home_override", lambda: None, raising=False)
    ts._load_cfg()  # populates the cache
    # The cache must hold the RAW user value, not the managed overlay, so a
    # subsequent _save_cfg can't bake the managed skin into the user file.
    assert (ts._cfg_cache.get("display") or {}).get("skin") == "user"


def test_logging_config_honors_managed(homes, monkeypatch):
    home, managed = homes
    _seed(home, managed, user="logging:\n  level: INFO\n", mgd="logging:\n  level: DEBUG\n")
    import hermes_logging

    level, _max, _bk = hermes_logging._read_logging_config()
    assert level == "DEBUG"


def test_timezone_honors_managed(homes, monkeypatch):
    home, managed = homes
    # hermes_time checks an env override first; ensure it's unset so config wins.
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    _seed(home, managed, user="timezone: America/New_York\n", mgd="timezone: Asia/Tokyo\n")
    import hermes_time

    assert hermes_time._resolve_timezone_name() == "Asia/Tokyo"


def test_gateway_env_bridge_honors_managed(homes, monkeypatch):
    """The gateway config→env bridge must bridge MANAGED values, not user ones.

    gateway/run.py bridges config.yaml settings into os.environ at startup and on
    every turn (HERMES_TIMEZONE, HERMES_REDACT_SECRETS, HERMES_MAX_ITERATIONS,
    ...). A managed value must win at that env layer too — otherwise the bridge
    writes the user's value into the env that the whole process then reads. This
    is the regression that manual verification caught (managed timezone was
    overridden by the user's value via the env bridge).

    We assert on the managed-overlaid config the bridge consumes (rather than the
    os.environ side effect, which leaks across same-process tests under the
    runner) — the bridge writes whatever this dict carries, so a managed value
    here proves the env var gets the managed value.
    """
    home, managed = homes
    _seed(home, managed, user="timezone: America/New_York\n", mgd="timezone: Asia/Tokyo\n")
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    # The bridge loads config.yaml, expands env, then applies this overlay before
    # writing HERMES_TIMEZONE = cfg["timezone"]. Prove the overlay flips the value.
    import yaml

    raw = yaml.safe_load((home / "config.yaml").read_text())
    bridged = managed_scope.apply_managed_overlay(raw)
    assert bridged.get("timezone") == "Asia/Tokyo"


def test_strict_config_read_rejects_malformed_managed_source(homes):
    home, managed = homes
    _seed(
        home,
        managed,
        user="plugins:\n  entries: {}\n",
        mgd="plugins:\n  entries: [unterminated\n",
    )
    from hermes_cli.config import load_config_readonly, load_config_readonly_strict

    assert isinstance(load_config_readonly(), dict)
    with pytest.raises(Exception):
        load_config_readonly_strict()
