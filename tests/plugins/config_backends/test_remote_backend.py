"""Remote config backend against a contract-faithful stub plane (``stub_plane.py``).

Covers the P4 behaviours: boot fetch fails closed with bounded retry, reads come from the plane and
never from a local config.yaml, writes are key-level diffs with CAS, locks and secret literals are
refused client-side (nothing sent), the poller keeps the in-memory document through an outage,
migrations run in memory only, and the managed config.yaml is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from utils import fast_safe_load

from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod
from hermes_cli.config_backend import (
    ConfigBackendUnavailable, ConfigLockedError, ConfigValueError, get_config_backend, write_config_key)

from .stub_plane import INSTANCE, TOKEN, StubPlane, remote_env


@pytest.fixture
def plane(tmp_path, monkeypatch):
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def _gets(plane):
    return [r for r in plane.requests if r["method"] == "GET"]


def _config_cmd(capsys, *argv):
    from hermes_cli.config import config_command
    ns = argparse.Namespace(config_command=argv[0], key=argv[1], value=argv[2] if len(argv) > 2 else None,
                            force=False)
    with pytest.raises(SystemExit) as exc:
        config_command(ns)
    return exc.value.code, capsys.readouterr().err


# --- selection and reads ------------------------------------------------------------------

def test_selected_from_env_and_file_tooling_off(plane):
    backend = get_config_backend()
    assert backend.name == "remote"
    assert backend.supports_file_tooling() is False
    assert backend.honors_managed_config() is False
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in backend.protected_env_names()


def test_reads_come_from_plane_not_local_file(plane):
    from hermes_cli.config import load_config
    plane.upper = {"model": {"default": "hermes-4", "provider": "nous"}, "terminal": {"timeout": 180}}
    local = plane.home / "config.yaml"
    local.write_text(json.dumps({"model": {"default": "LOCAL"}, "terminal": {"timeout": 1}}))

    cfg = load_config()

    assert cfg["model"]["default"] == "hermes-4"
    assert cfg["terminal"]["timeout"] == 180
    assert "display" in cfg  # DEFAULT_CONFIG stays under the remote user layer
    assert fast_safe_load(local.read_text())["model"]["default"] == "LOCAL"  # untouched
    gets = _gets(plane)
    assert gets and all(r["auth"] == f"Bearer {TOKEN}" and r["instance"] == INSTANCE for r in gets)
    assert gets[0]["profile"] == "default"


def test_boot_fetch_runs_inside_load_hermes_dotenv(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    load_hermes_dotenv(hermes_home=plane.home)  # idempotent: one fetch per process per profile
    assert len(_gets(plane)) == 1


# --- fail-closed boot ---------------------------------------------------------------------

def test_boot_retries_then_exits_on_outage(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    plane.fail_status = 503
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 3  # attempts at t=0, ~10 s, ~30 s (delays patched to 0)
    assert "does not start" in str(exc.value.code)
    assert isinstance(exc.value, SystemExit)


def test_boot_does_not_retry_a_refusal(plane, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv
    monkeypatch.setenv("HERMES_CONFIG_INSTANCE_ID", "someone-else")
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    assert "config_agent_unknown" in str(exc.value.code)


def test_boot_requires_instance_id(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("HERMES_CONFIG_INSTANCE_ID")
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_INSTANCE_ID"):
        load_config()
    assert plane.requests == []


def test_bad_idp_credentials_fail_closed(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "wrong")
    with pytest.raises(ConfigBackendUnavailable, match="IdP token request"):
        load_config()
    assert plane.requests == []  # no /self call without a token


def test_partial_idp_config_is_not_retried(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("GATEWAY_RELAY_IDP_CLIENT_SECRET")
    with pytest.raises(ConfigBackendUnavailable, match="CLIENT_SECRET"):
        load_config()
    assert plane.token_requests == 0


def test_secret_source_supplying_plane_credential_refuses_start(plane):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    report = SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()},
                             sources=[SimpleNamespace(skipped_existing=["HERMES_CONFIG_REMOTE_URL"])])
    with pytest.raises(ConfigBackendUnavailable) as exc:
        _refuse_protected_env_from_sources(report)
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in str(exc.value.code)
    assert "HERMES_CONFIG_REMOTE_URL" in str(exc.value.code)
    ok = SimpleNamespace(provenance={"OPENROUTER_API_KEY": object()}, sources=[])
    _refuse_protected_env_from_sources(ok)  # unrelated names pass


def test_boot_refuses_remote_secrets_source_that_maps_plane_credential(plane, monkeypatch):
    """End to end through load_hermes_dotenv: the REMOTE secrets: section drives the sources (D32)."""
    from types import SimpleNamespace

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    plane.upper = {"secrets": {"command": {"enabled": True}}}
    seen = {}

    def fake_apply_all(cfg, home_path, environ=None):
        seen["cfg"] = cfg
        return SimpleNamespace(sources=[SimpleNamespace(skipped_existing=[])], applied_any=True,
                               provenance={"HERMES_CONFIG_REMOTE_URL": SimpleNamespace(source="command")})

    monkeypatch.setattr(registry, "apply_all", fake_apply_all)
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)
    assert seen["cfg"] == {"command": {"enabled": True}}


def test_file_backend_has_no_protected_names(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    _refuse_protected_env_from_sources(
        SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()}, sources=[]))


# --- writes -------------------------------------------------------------------------------

def test_config_set_sends_only_the_changed_key(plane):
    from hermes_cli.config import load_config, set_config_value
    plane.upper = {"model": {"default": "hermes-4", "provider": "nous"}, "terminal": {"timeout": 180}}

    set_config_value("display.personality", "pirate")

    patches = plane.patches()
    assert len(patches) == 1
    body = patches[0]["body"]
    assert body["set"] == {"display": {"personality": "pirate"}}  # absent from the effective doc
    assert "unset" not in body
    assert body["expectedVersion"] == 0
    assert isinstance(body["writerConfigVersion"], int)
    assert plane.profile("default")["values"] == {"display": {"personality": "pirate"}}
    assert load_config()["display"]["personality"] == "pirate"
    assert not (plane.home / "config.yaml").exists()


def test_change_under_existing_section_is_a_leaf_set(plane):
    from hermes_cli.config import set_config_value
    plane.upper = {"display": {"personality": "concise", "compact": False}}
    set_config_value("display.personality", "pirate")
    assert plane.patches()[0]["body"]["set"] == {"display.personality": "pirate"}
    assert plane.profile("default")["values"] == {"display": {"personality": "pirate"}}


def test_write_conflict_rereads_and_retries_once(plane):
    from hermes_cli.config import load_config
    load_config()
    plane.profile("default")["version"] = 5  # another writer moved the profile level
    write_config_key(plane.home / "config.yaml", "display.personality", "pirate")
    patches = plane.patches()
    assert [p["body"]["expectedVersion"] for p in patches] == [0, 5]
    assert plane.profile("default")["version"] == 6


def test_locked_key_refused_by_config_set_without_sending(plane, capsys):
    plane.upper = {"terminal": {"timeout": 180}}
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]

    code, err = _config_cmd(capsys, "set", "terminal.timeout", "5")

    assert code == 1
    assert "locked by the tenant level" in err
    assert plane.patches() == []


def test_locked_key_refused_by_backend_single_key_write(plane):
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]
    with pytest.raises(ConfigLockedError) as exc:
        write_config_key(plane.home / "config.yaml", "terminal.backend", "local")
    assert (exc.value.path, exc.value.locked_by) == ("terminal", "tenant")
    assert plane.patches() == []


def test_r7_scalar_over_locked_subtree_refused(plane):
    plane.upper_locks = [{"path": "a.b", "level": "tenant"}]
    with pytest.raises(ConfigLockedError):
        write_config_key(plane.home / "config.yaml", "a", "scalar")
    assert plane.patches() == []


def test_bulk_save_strips_locked_keys_and_sends_the_rest(plane, capsys):
    from hermes_cli.config import read_raw_config, save_config
    plane.upper = {"terminal": {"timeout": 180}, "model": {"default": "hermes-4"}}
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]
    doc = read_raw_config()
    doc["terminal"]["timeout"] = 5
    doc["display"] = {"personality": "pirate"}

    save_config(doc)

    patches = plane.patches()
    assert len(patches) == 1
    assert patches[0]["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "terminal.timeout" not in json.dumps(patches[0]["body"])
    assert "locked by Remote Config were not saved: terminal" in capsys.readouterr().err


def test_secret_literal_refused_client_side(plane, capsys):
    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "mcp_servers.github.env.GITHUB_TOKEN", "ghp_live")
    assert exc.value.code == "config_secret_literal"
    code, err = _config_cmd(capsys, "set", "delegation.api_key", "sk-live-abc")
    assert code == 1 and "secret-shaped" in err
    assert plane.patches() == []
    write_config_key(plane.home / "config.yaml", "delegation.api_key", "${OPENROUTER_API_KEY}")
    assert plane.patches()[-1]["body"]["set"] == {"delegation": {"api_key": "${OPENROUTER_API_KEY}"}}


def test_unset_of_inherited_key_warns(plane, caplog):
    plane.upper = {"display": {"personality": "concise"}}
    plane.profile("default")["values"] = {"display": {"personality": "pirate"}}
    with caplog.at_level("WARNING"):
        write_config_key(plane.home / "config.yaml", "display.personality", None)
    assert plane.patches()[0]["body"]["unset"] == ["display.personality"]
    assert "still set by an upper level" in caplog.text


def test_yaml_only_values_are_converted_before_the_diff(plane):
    import datetime
    write_config_key(plane.home / "config.yaml", "x.when", datetime.date(2026, 9, 25))
    write_config_key(plane.home / "config.yaml", "x.big", 2**60)
    sent = [p["body"]["set"] for p in plane.patches()]
    assert sent == [{"x": {"when": "2026-09-25"}}, {"x.big": str(2**60)}]
    with pytest.raises(ConfigValueError, match="NaN"):
        write_config_key(plane.home / "config.yaml", "x.nan", float("nan"))


# --- poll, outage, versions ---------------------------------------------------------------

def test_poll_picks_up_changes_and_invalidates_caches(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    assert load_config()["display"]["personality"] == "concise"
    backend = remote_pkg.get_remote_backend()
    v1 = backend.version(plane.home)

    backend.poll_all()  # unchanged → 304
    assert plane.requests[-1]["if_none_match"] and backend.version(plane.home) == v1

    plane.upper = {"display": {"personality": "pirate"}}
    backend.poll_all()
    assert backend.version(plane.home) != v1
    assert load_config()["display"]["personality"] == "pirate"


def test_outage_while_running_keeps_in_memory_doc(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    load_config()
    backend = remote_pkg.get_remote_backend()
    plane.fail_status = 503

    backend.poll_all()  # never raises, never exits

    assert load_config()["display"]["personality"] == "concise"
    assert "503" in backend.describe(plane.home)
    plane.fail_status = None
    backend.poll_all()
    assert "last poll failed" not in backend.describe(plane.home)


def test_poll_refusal_logs_error_but_keeps_running(plane, caplog):
    from hermes_cli.config import load_config
    load_config()
    plane.fail_status = 403
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["ERROR"]
    plane.fail_status = 503
    caplog.clear()
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["WARNING"]


def test_forked_child_rearms_its_poller(plane):
    import os

    from hermes_cli.config import load_config
    load_config()
    backend = remote_pkg.get_remote_backend()
    first = backend._poller
    backend._poller_pid = -1  # as in a child after fork: the parent's thread did not survive
    load_config()
    assert backend._poller_pid == os.getpid() and backend._poller is not first
    assert backend._poller is not None and backend._poller.is_alive()


def test_poll_interval_floor(monkeypatch):
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "1")
    assert backend_mod.poll_interval() == backend_mod.MIN_POLL_SECONDS
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "600")
    assert backend_mod.poll_interval() == 600


def test_named_profile_fetches_its_own_layer(plane):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from hermes_cli.config import load_config
    work = plane.home / "profiles" / "work"
    work.mkdir(parents=True)
    plane.profile("work")["values"] = {"display": {"personality": "formal"}}
    token = set_hermes_home_override(work)
    try:
        assert load_config()["display"]["personality"] == "formal"
    finally:
        reset_hermes_home_override(token)
    assert {r["profile"] for r in _gets(plane)} == {"work"}


# --- migrations, unknown keys, managed scope ----------------------------------------------

def test_old_writer_version_migrates_in_memory_only(plane):
    from hermes_cli.config import read_raw_config
    from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION
    latest = backend_mod._latest_config_version()
    old = max(SUPPORT_FLOOR_VERSION, latest - 1)
    plane.profile("default").update(values={"display": {"personality": "pirate"}}, version=1, writer=old)

    doc = read_raw_config()

    assert doc["_config_version"] == latest
    assert doc["display"]["personality"] == "pirate"
    assert plane.patches() == []  # D12: never written back
    assert not (plane.home / "config.yaml").exists()


def test_migration_steps_persist_into_memory(plane, monkeypatch):
    """A migration step's ``_persist_migration`` lands in the in-memory layer, not on disk/plane."""
    from hermes_cli import config as config_mod
    from hermes_cli import config_migrations
    latest = backend_mod._latest_config_version()
    plane.profile("default").update(values={"old_key": 1}, version=1, writer=latest - 1)

    def fake_run_migrations(current, results, quiet, **kwargs):
        doc = config_mod.read_raw_config()
        doc["new_key"] = doc.pop("old_key")
        config_mod._persist_migration(doc)

    monkeypatch.setattr(config_migrations, "run_migrations", fake_run_migrations)
    doc = config_mod.read_raw_config()
    assert doc.get("new_key") == 1 and "old_key" not in doc
    assert plane.patches() == []


def test_unknown_top_level_key_warns_once(plane, caplog):
    from hermes_cli.config import load_config
    plane.upper = {"from_the_future": {"x": 1}}
    with caplog.at_level("WARNING"):
        load_config()
        load_config()
    assert caplog.text.count("unknown config key 'from_the_future'") == 1


def test_managed_config_yaml_is_ignored_and_flagged(plane, tmp_path, monkeypatch, capsys):
    from hermes_cli import managed_scope
    from hermes_cli.config import load_config
    from hermes_cli.env_loader import load_hermes_dotenv
    managed = tmp_path / "etc-hermes"
    managed.mkdir()
    (managed / "config.yaml").write_text(json.dumps({"display": {"personality": "managed"}}))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope._CONFIG_CACHE.clear()
    plane.upper = {"display": {"personality": "remote"}}

    load_hermes_dotenv(hermes_home=plane.home)

    assert "IGNORED" in capsys.readouterr().err
    assert managed_scope.load_managed_config() == {}
    assert load_config()["display"]["personality"] == "remote"


# --- tooling ------------------------------------------------------------------------------

def test_file_tooling_refused(plane):
    from hermes_cli.config_backend import require_file_tooling
    with pytest.raises(ConfigBackendUnavailable, match="Profile clone"):
        require_file_tooling("Profile clone")


def test_doctor_reports_backend_status(plane, capsys):
    from hermes_cli.doctor_config import _check_config_backend
    from hermes_cli.doctor_report import Finding
    f = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f) is True
    assert "Remote Config" in capsys.readouterr().out
    plane.fail_status = 503
    remote_pkg._reset_for_tests()
    f2 = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f2) is False
    assert f2.issues


def test_general_plugin_manager_never_loads_config_backends(plane, monkeypatch):
    from hermes_cli.plugins import PluginManager
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "file")
    mgr = PluginManager()
    mgr.discover_and_load()
    assert not any("config_backends" in key or key == "remote" for key in mgr._plugins)


# --- review round 1 regressions -----------------------------------------------------------

_BACKEND_FLIP_SOURCE = {"command": {"enabled": True, "override_existing": True,
                                    "command": "printf 'HERMES_CONFIG_BACKEND=file\\n'"}}


def test_secret_source_cannot_switch_remote_mode_off_at_boot(plane, monkeypatch):
    """D32 through the REAL startup path and a real command source: a source that writes
    HERMES_CONFIG_BACKEND=file must not disarm remote mode (and with it the protected-name check)
    and let the next read fall back to the local config.yaml."""
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    (plane.home / "config.yaml").write_text(json.dumps({"display": {"personality": "local"}}))
    plane.upper = {"display": {"personality": "remote"}, "secrets": _BACKEND_FLIP_SOURCE}

    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_BACKEND"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)

    import os
    assert os.environ["HERMES_CONFIG_BACKEND"] == "remote"  # the source's write was reverted
    assert get_config_backend().name == "remote"


def test_file_mode_ignores_a_source_that_selects_the_backend(monkeypatch, capsys):
    """The selector is never a source's to change under the file backend either: reverted, dropped
    from the report (so no later snapshot re-applies it), and startup continues."""
    import os
    from types import SimpleNamespace

    from hermes_cli.config_backend import get_config_backend as gcb
    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    before = dict(os.environ)
    file_backend = gcb()
    src = SimpleNamespace(applied=["HERMES_CONFIG_BACKEND", "OPENROUTER_API_KEY"], skipped_existing=[])
    report = SimpleNamespace(provenance={"HERMES_CONFIG_BACKEND": object(), "OPENROUTER_API_KEY": object()},
                             sources=[src])
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "remote")  # what the source wrote

    _refuse_protected_env_from_sources(report, file_backend, before)

    assert "HERMES_CONFIG_BACKEND" not in os.environ
    assert set(report.provenance) == {"OPENROUTER_API_KEY"} and src.applied == ["OPENROUTER_API_KEY"]
    assert "ignored HERMES_CONFIG_BACKEND" in capsys.readouterr().err


def _hold_gets(monkeypatch):
    """Pause the FIRST GET after its response arrived and before the backend installs it, until
    ``release`` is set; every later request passes straight through."""
    import threading

    from plugins.config_backends.remote import client
    fetched, release = threading.Event(), threading.Event()
    original = client.request

    def held(method, *args, **kwargs):
        resp = original(method, *args, **kwargs)
        if method == "GET" and not fetched.is_set():
            fetched.set()
            assert release.wait(10)
        return resp

    monkeypatch.setattr(client, "request", held)
    return fetched, release


def test_poll_in_flight_does_not_overwrite_an_acknowledged_write(plane, monkeypatch):
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "upper-change"
    fetched, release = _hold_gets(monkeypatch)

    poll = threading.Thread(target=backend.poll_one, args=(st,))
    poll.start()
    assert fetched.wait(10)  # the poll's GET (profile v0) has its response, not yet installed
    write_config_key(plane.home / "config.yaml", "display.personality", "my-write")
    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    release.set()
    poll.join(10)
    assert not poll.is_alive()

    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    assert plane.profile("default")["values"] == {"display": {"personality": "my-write"}}


def test_stale_fetch_with_same_profile_version_is_dropped(plane, monkeypatch):
    """Only an upper level changed, so both responses carry profileVersion 0: the guard must be
    'which fetch was installed last', not a profileVersion comparison."""
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "v1"
    fetched, release = _hold_gets(monkeypatch)

    slow = threading.Thread(target=backend.poll_one, args=(st,))
    slow.start()
    assert fetched.wait(10)  # holds an upper=v1 response
    plane.upper["display"]["personality"] = "v2"
    assert backend.poll_one(st) is True  # a newer fetch, started later, lands first
    assert st.doc["display"]["personality"] == "v2"
    release.set()
    slow.join(10)
    assert not slow.is_alive()

    assert st.profile_version == 0
    assert st.doc["display"]["personality"] == "v2"


def _old_compression_profile(plane):
    from hermes_cli.config import read_raw_config
    assert backend_mod._latest_config_version() > 46
    plane.upper = {"compression": {"threshold_tokens": 256000}}  # the old default the 46->47 step removes
    plane.profile("default")["writer"] = 46
    doc = read_raw_config()
    assert "threshold_tokens" not in (doc.get("compression") or {}), "precondition: migrated in memory"


def test_migration_is_not_sent_by_an_unrelated_write(plane):
    from hermes_cli.config import set_config_value
    _old_compression_profile(plane)

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert patch["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "unset" not in patch["body"]  # D12: the migration's removal is never written back


def test_migration_is_not_sent_after_a_cas_reread(plane):
    from hermes_cli.config import set_config_value
    _old_compression_profile(plane)
    plane.profile("default")["version"] = 3  # another writer moved the profile level: first PATCH 409s

    set_config_value("display.personality", "pirate")

    patches = plane.patches()
    assert [p["body"]["expectedVersion"] for p in patches] == [0, 3]
    for p in patches:
        assert p["body"]["set"] == {"display": {"personality": "pirate"}}
        assert "unset" not in p["body"]


def test_bulk_save_after_migration_sends_only_the_edit(plane):
    from hermes_cli.config import read_raw_config, save_config
    _old_compression_profile(plane)
    doc = read_raw_config()
    doc.setdefault("display", {})["personality"] = "pirate"

    save_config(doc)

    (p,) = plane.patches()
    assert p["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "unset" not in p["body"]
    assert "compression" not in json.dumps(p["body"])


def test_unrelated_write_keeps_an_old_profile_stamp_so_later_reads_still_migrate(plane):
    """Orchestrator ruling (option 1): the profile level stamped 46 still stores a pre-migration
    key; an unrelated write must not re-stamp it, or every later reader would skip 46->47."""
    from hermes_cli.config import read_raw_config, set_config_value
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, writer=46)
    assert "threshold_tokens" not in (read_raw_config().get("compression") or {}), "precondition"

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert "writerConfigVersion" not in patch["body"]
    assert plane.profile("default")["writer"] == 46
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 256000}

    remote_pkg._reset_for_tests()  # a later process reads the same profile level
    doc = read_raw_config()
    assert "threshold_tokens" not in (doc.get("compression") or {})  # 46->47 still ran
    assert doc["display"]["personality"] == "pirate"
    assert len(plane.patches()) == 1  # D12: the later read wrote nothing back


@pytest.mark.parametrize("stored", ["latest", None])
def test_write_stamps_a_current_or_unstamped_profile_level(plane, stored):
    from hermes_cli.config import set_config_value
    latest = backend_mod._latest_config_version()
    plane.profile("default")["writer"] = latest if stored == "latest" else None

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert patch["body"]["writerConfigVersion"] == latest
    assert plane.profile("default")["writer"] == latest


def test_write_never_lowers_a_newer_profile_stamp(plane):
    from hermes_cli.config import set_config_value
    newer = backend_mod._latest_config_version() + 1
    plane.profile("default")["writer"] = newer

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert "writerConfigVersion" not in patch["body"]
    assert plane.profile("default")["writer"] == newer


def _dashboard_put(config):
    import asyncio

    from fastapi import HTTPException

    from hermes_cli.web_routers.config_env import ConfigUpdate, update_config
    try:
        asyncio.run(update_config(ConfigUpdate(config=config)))
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return 200, None


def test_dashboard_save_of_a_newly_locked_key_returns_the_lock_message(plane):
    """The lock arrived on the plane after the last poll, so only the server refuses (403)."""
    from hermes_cli.config import read_raw_config
    plane.upper = {"display": {"personality": "old"}}
    read_raw_config()
    plane.upper_locks = [{"path": "display.personality", "level": "tenant"}]

    status, detail = _dashboard_put({"display": {"personality": "new"}})

    assert status == 403
    assert "display.personality" in detail and "tenant" in detail
    assert plane.profile("default")["values"] == {}


def test_dashboard_save_secret_literal_is_a_400(plane):
    status, detail = _dashboard_put({"model": {"api_key": "sk-live-not-a-ref-1234567890"}})
    assert status == 400
    assert "secret-shaped" in detail and "sk-live" not in detail
    assert plane.patches() == []


def test_dashboard_unexpected_config_failure_stays_opaque(plane):
    from hermes_cli.config import read_raw_config
    read_raw_config()
    plane.fail_status = 500  # the write reaches the plane, which fails: not an expected refusal
    status, detail = _dashboard_put({"display": {"personality": "new"}})
    assert (status, detail) == (500, "Internal server error")


# --- review round 2 regressions -----------------------------------------------------------

def test_refused_source_plane_url_is_never_visible_to_a_concurrent_request(plane, monkeypatch):
    """D32 credential boundary: a real command source supplies HERMES_CONFIG_REMOTE_URL pointing
    at another plane. A poll scheduled right after the sources ran (before the refusal) must still
    go to the configured plane: the forbidden URL is never published, so the bearer and instance id
    never reach the other server."""
    import os
    import threading

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with StubPlane() as other:
        plane.upper = {"secrets": {"command": {
            "enabled": True, "override_existing": True,
            "command": f"printf 'HERMES_CONFIG_REMOTE_URL={other.url}\\n'"}}}
        backend = get_config_backend()
        st = backend._state(plane.home)
        real_apply_all = registry.apply_all
        seen = {}

        def apply_then_poll(*args, **kwargs):
            report = real_apply_all(*args, **kwargs)
            seen["url_after_sources"] = os.environ.get("HERMES_CONFIG_REMOTE_URL")
            poll = threading.Thread(target=backend.poll_one, args=(st,))
            poll.start()
            poll.join(10)
            assert not poll.is_alive()
            return report

        monkeypatch.setattr(registry, "apply_all", apply_then_poll)
        gets_before = len(_gets(plane))
        with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
            env_loader.load_hermes_dotenv(hermes_home=plane.home)

        assert "url_after_sources" in seen, "precondition: the source ran"
        assert other.requests == []  # nothing — least of all the bearer — reached the other plane
        assert seen["url_after_sources"] == plane.url
        assert len(_gets(plane)) > gets_before  # the gap poll went to the configured plane
        assert os.environ["HERMES_CONFIG_REMOTE_URL"] == plane.url


def test_file_mode_publishes_permitted_source_values(monkeypatch, tmp_path):
    """Staging must not lose ordinary secrets: permitted names are published after the check."""
    import os

    from hermes_cli import env_loader
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    monkeypatch.setenv("CC_STAGED_SECRET", "x")
    monkeypatch.delenv("CC_STAGED_SECRET")
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: \"printf 'CC_STAGED_SECRET=from-source\\\\n'\"\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    env_loader._apply_external_secret_sources(home)

    assert os.environ.get("CC_STAGED_SECRET") == "from-source"


def test_migration_of_a_replaced_doc_does_not_touch_the_replacement(plane, monkeypatch):
    """A read starts migrating the stored-46 doc; before the migration begins, a poll installs the
    current-version doc that keeps compression.threshold_tokens on purpose. The stale 46->47 step
    must not run against it (the replacement stays intact, nothing is written back)."""
    import threading

    from hermes_cli.config import read_raw_config
    latest = backend_mod._latest_config_version()
    assert latest > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, writer=46)
    backend = get_config_backend()
    st = backend._state(plane.home)
    entered, resume = threading.Event(), threading.Event()
    real_migrate = backend._migrate_in_memory

    def paused(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        return real_migrate(*args, **kwargs)

    monkeypatch.setattr(backend, "_migrate_in_memory", paused)
    failures = []

    def read():
        try:
            read_raw_config()
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            failures.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    assert entered.wait(10)
    plane.profile("default").update(writer=latest, version=1)
    assert backend.poll_one(st)
    assert st.doc["compression"] == {"threshold_tokens": 256000}
    resume.set()
    reader.join(10)
    assert not reader.is_alive() and not failures, failures

    doc = read_raw_config()
    assert doc["_config_version"] == latest
    assert doc["compression"] == {"threshold_tokens": 256000}
    assert plane.patches() == []


def test_dashboard_save_refused_by_plane_value_check_is_a_400(plane):
    """The plane's own value refusal (config_value_invalid too_deep, contract §11.1) reaches the
    dashboard as a 400 with its reason, not an opaque 500."""
    from hermes_cli.config import read_raw_config
    read_raw_config()
    value = "leaf"
    for _ in range(33):
        value = [value]

    status, detail = _dashboard_put({"display": {"custom": value}})

    assert len(plane.patches()) == 1  # client sent it; the server refused
    assert status == 400
    assert "too_deep" in detail and "display" in detail  # the plane's reason and path pass through


@pytest.mark.parametrize("status,error,expected", [
    (400, "config_value_invalid", 400), (400, "config_path_invalid", 400), (400, "config_path_reserved", 400),
    (400, "config_secret_literal", 400), (413, "config_level_too_large", 413), (413, "config_body_too_large", 413),
    (400, "config_request_invalid", None), (500, "internal", None)])
def test_plane_refusals_map_to_http(status, error, expected):
    from hermes_cli.web_routers._common import config_refusal_http
    from plugins.config_backends.remote import client
    exc = backend_mod.RemoteBackend._write_error(
        client.Response(status=status, body={"error": error, "message": "refused", "path": "a.b"},
                        etag=None, retry_after=None))
    http = config_refusal_http(exc)
    assert (http.status_code if http else None) == expected
    assert exc.code == error


# --- portal-groups (multi-group plane, contract fixtureVersion 2 / ETag v2) ------------------

def _multi_group_plane(plane):
    plane.upper = {"terminal": {"backend": "docker"}, "model": {"default": "hermes-4"}}
    plane.upper_locks = [{"path": "terminal.backend", "level": "group", "groupId": "g-sec"}]
    plane.groups = [{"groupId": "g-sec", "priority": 10, "version": 3},
                    {"groupId": "g-eng", "priority": 5, "version": 1}]
    plane.group_provenance = {"terminal": {"backend": "g-sec"}}


def test_multi_group_response_fields_are_ignored_and_group_lock_refuses_client_side(plane):
    """portal-groups §7.5: several group levels, locks[].groupId and groupProvenance are ignorable;
    a group lock is just an upper lock (lockedBy "group"), refused before anything is sent."""
    from hermes_cli.config import read_raw_config
    _multi_group_plane(plane)

    doc = read_raw_config()

    assert doc["terminal"]["backend"] == "docker" and doc["model"]["default"] == "hermes-4"
    assert get_config_backend().locked(plane.home, "terminal.backend") == "group"
    with pytest.raises(ConfigLockedError) as exc:
        write_config_key(plane.home / "config.yaml", "terminal.backend", "local")
    assert (exc.value.path, exc.value.locked_by) == ("terminal.backend", "group")
    assert plane.patches() == []
    write_config_key(plane.home / "config.yaml", "model.default", "hermes-5")  # unlocked: written
    (p,) = plane.patches()
    assert p["body"]["set"] == {"model.default": "hermes-5"}


def test_plane_group_lock_refusal_maps_to_locked_error_and_403():
    """A server-side config_key_locked from a group lock carries lockedByGroupId (portal-groups
    §6.6); the agent still maps it to ConfigLockedError(path, "group") and the dashboard's 403."""
    from hermes_cli.web_routers._common import config_refusal_http
    from plugins.config_backends.remote import client
    exc = backend_mod.RemoteBackend._write_error(client.Response(
        status=403, body={"error": "config_key_locked", "path": "terminal.backend", "lockedBy": "group",
                          "lockedByGroupId": "g-sec", "message": "terminal.backend is locked by group g-sec"},
        etag=None, retry_after=None))
    assert isinstance(exc, ConfigLockedError)
    assert (exc.path, exc.locked_by) == ("terminal.backend", "group")
    http = config_refusal_http(exc)
    assert http is not None and http.status_code == 403 and "locked by group" in http.detail


def test_etag_is_opaque_and_echoed_verbatim(plane, monkeypatch):
    """The plane's ETag (hermes-config-etag/2 since portal-groups §6.8) is never parsed: whatever
    bytes it sends come back unchanged in If-None-Match."""
    from hermes_cli.config import read_raw_config
    _multi_group_plane(plane)
    real_effective = plane.effective
    opaque = 'W/"v2:any-bytes/at all"'

    def effective(name):
        body = real_effective(name)
        body["etag"] = opaque
        return body

    monkeypatch.setattr(plane, "effective", effective)
    read_raw_config()
    backend = get_config_backend()
    st = backend._state(plane.home)
    assert backend.poll_one(st) is False and not st.last_error  # 304: nothing changed
    assert _gets(plane)[-1]["if_none_match"] == opaque


def test_stub_plane_deep_merge_follows_contract_6_2():
    """The test double resolves like the plane: null over a mapping is ignored (contract §6.2)."""
    from .stub_plane import deep_merge
    upper = {"display": {"personality": "concise"}, "model": "a", "tags": [1, 2]}
    assert deep_merge(upper, {"display": None}) == upper
    assert deep_merge(upper, {"model": None})["model"] is None
    assert deep_merge(upper, {"tags": [3]})["tags"] == [3]


# --- review round 3 regressions -----------------------------------------------------------

def _drop_process_deployment(plane, monkeypatch, tmp_path):
    """Move the deployment out of the process env (as on a host where it lives in .env files);
    returns the managed dir. Every name stays recorded by monkeypatch, so whatever a dotenv load
    publishes is undone at teardown."""
    from hermes_cli import config_backend
    for name in remote_env(plane):
        monkeypatch.delenv(name, raising=False)
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", True)  # this process is past its first read
    return managed


def test_cas_retry_of_a_document_write_replays_only_its_own_edit(plane, monkeypatch):
    """F1: another writer commits between our read and our PATCH. The 409 retry re-applies OUR
    edit to the re-read doc; it must not turn the other writer's changes into edits of ours."""
    from hermes_cli.config import set_config_value
    from plugins.config_backends.remote import client
    plane.profile("default").update(values={"display": {"personality": "concise", "compact": False}}, version=1)
    real_request = client.request
    raced = []

    def other_writer_lands_first(method, *args, **kwargs):
        if method == "PATCH" and not raced:
            raced.append(True)
            prof = plane.profile("default")
            prof["values"]["display"]["compact"] = True
            prof["values"]["terminal"] = {"timeout": 321}
            prof["version"] += 1
        return real_request(method, *args, **kwargs)

    monkeypatch.setattr(client, "request", other_writer_lands_first)

    set_config_value("display.personality", "pirate")

    assert plane.profile("default")["values"] == {
        "display": {"personality": "pirate", "compact": True}, "terminal": {"timeout": 321}}
    first, retry = plane.patches()
    assert first["body"]["expectedVersion"] == 1 and retry["body"]["expectedVersion"] == 2
    assert "unset" not in retry["body"] and "compact" not in json.dumps(retry["body"])


def test_edit_during_an_in_memory_migration_never_sends_the_migration(plane, monkeypatch):
    """F2: another thread's in-memory migration has run its steps but not finished when a public
    edit reads and saves. The edit must diff against the doc it read, so the migration's removal
    of compression.threshold_tokens is not sent (D12)."""
    import threading

    from hermes_cli import config_migrations
    from hermes_cli.config import read_raw_config, set_config_value
    from hermes_cli.config_backend import read_config_doc
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, version=1, writer=46)
    get_config_backend()._state(plane.home)
    steps_ran, resume = threading.Event(), threading.Event()
    real_run = config_migrations.run_migrations

    def paused(*args, **kwargs):
        real_run(*args, **kwargs)
        if threading.current_thread().name == "migrator":  # the edit's own checks run unpaused
            steps_ran.set()
            assert resume.wait(10), "the edit waited for the migration"

    monkeypatch.setattr(config_migrations, "run_migrations", paused)
    failures = []

    def migrate():  # a lock-free reader (read_config_doc) starts the migration
        try:
            read_config_doc(plane.home / "config.yaml")
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            failures.append(exc)

    migrator = threading.Thread(target=migrate, name="migrator")
    migrator.start()
    try:
        assert steps_ran.wait(10)
        set_config_value("display.personality", "pirate")
    finally:
        resume.set()
        migrator.join(10)
    assert not migrator.is_alive() and not failures, failures

    (patch,) = plane.patches()
    assert "unset" not in patch["body"] and "compression" not in json.dumps(patch["body"])
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 256000}
    later = read_raw_config()
    assert later["display"]["personality"] == "pirate"
    assert "threshold_tokens" not in (later.get("compression") or {})  # still migrated, in memory


def test_profile_added_while_the_poller_iterates_does_not_kill_it(plane):
    """F3: a first read of another profile inserts its state while the poll loop walks the
    roster; the only poller thread must survive it."""
    import dataclasses
    import threading

    backend = get_config_backend()
    st = backend._state(plane.home)
    work = plane.home / "profiles" / "work"
    work.mkdir(parents=True)
    inserted = threading.Event()

    class AddsAProfileMidPass(backend_mod._ProfileState):
        reads = 0

        @property
        def next_poll(self):
            AddsAProfileMidPass.reads += 1
            if AddsAProfileMidPass.reads == 2 and not inserted.is_set():  # mid-pass: the deadline scan
                backend._state(work)
                inserted.set()
            return time.monotonic() + 1000

        @next_poll.setter
        def next_poll(self, value):
            pass

    import time
    key = backend._key(plane.home)
    backend._states[key] = AddsAProfileMidPass(**{
        f.name: getattr(st, f.name) for f in dataclasses.fields(st) if f.name != "next_poll"})
    loop = threading.Thread(target=backend._poll_loop, daemon=True)
    loop.start()
    assert inserted.wait(10)
    loop.join(2)
    assert loop.is_alive()
    assert backend._key(work) in backend._states
    backend._stop.set()


def test_a_dead_poller_is_rearmed_by_a_cached_read(plane):
    """F3: a poller that died must not leave cached profiles without updates."""
    import threading

    from hermes_cli.config import load_config
    load_config()
    backend = remote_pkg.get_remote_backend()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    backend._poller = dead
    load_config()
    assert backend._poller is not dead and backend._poller.is_alive()


def test_instance_id_from_the_managed_dotenv_boots(plane, monkeypatch, tmp_path):
    """F4: the deployment in the home's .env, the instance id only in the managed .env."""
    import os

    from hermes_cli.env_loader import load_hermes_dotenv
    managed = _drop_process_deployment(plane, monkeypatch, tmp_path)
    env = remote_env(plane)
    (managed / ".env").write_text(f"HERMES_CONFIG_INSTANCE_ID={env.pop('HERMES_CONFIG_INSTANCE_ID')}\n")
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    plane.upper = {"display": {"personality": "managed"}}

    load_hermes_dotenv(hermes_home=plane.home)

    gets = _gets(plane)
    assert gets and gets[0]["instance"] == INSTANCE
    assert os.environ["HERMES_CONFIG_INSTANCE_ID"] == INSTANCE


def test_child_for_another_profile_keeps_the_remote_deployment(plane, monkeypatch, tmp_path):
    """F5: the deployment came from the launch profile's .env. A Hermes child built for another
    profile must still read that profile's config remotely, never fall back to its local file."""
    import os
    import subprocess
    import sys

    from hermes_cli.env_loader import load_hermes_dotenv
    from tools.environments.local import served_profile_child_env
    _drop_process_deployment(plane, monkeypatch, tmp_path)
    env = remote_env(plane)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    load_hermes_dotenv(hermes_home=plane.home)
    beta = plane.home / "profiles" / "beta"
    beta.mkdir(parents=True)
    (beta / "config.yaml").write_text("display:\n  personality: local-child\n")
    # beta's own plane credential (credentials are never carried into another profile's child)
    (beta / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("GATEWAY_RELAY_IDP_")))
    plane.profile("beta")["values"] = {"display": {"personality": "remote-child"}}

    child_env = served_profile_child_env(target_home=beta, inherit_credentials=True)
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    code = ("import json\n"
            "from hermes_cli.env_loader import load_hermes_dotenv\n"
            "load_hermes_dotenv()\n"
            "from hermes_cli.config_backend import get_config_backend\n"
            "from hermes_cli.config import load_config\n"
            "print('RESULT=' + json.dumps([get_config_backend().name, load_config()['display']['personality']]))\n")
    proc = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True,
                          timeout=120, stdin=subprocess.DEVNULL, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT=")][-1]
    assert json.loads(line[len("RESULT="):]) == ["remote", "remote-child"]
    assert "beta" in {r["profile"] for r in _gets(plane)}
    assert not any(k.startswith("GATEWAY_RELAY_IDP_") and v != env[k] for k, v in child_env.items())
    assert os.environ["HERMES_CONFIG_BACKEND"] == "remote"


def test_setting_a_value_that_old_schema_migration_rewrites_is_refused(plane, capsys):
    """F8: on a level stamped 46 (option 1 keeps that stamp), 256000 is exactly what the 46->47
    step removes on every read: acknowledging it would hide it. Refused, nothing sent. A value
    the migration keeps is saved as usual."""
    from hermes_cli.config import read_raw_config
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"display": {"personality": "old"}}, version=1, writer=46)
    read_raw_config()

    code, err = _config_cmd(capsys, "set", "compression.threshold_tokens", "256000")

    assert code == 1 and "compression.threshold_tokens" in err and "schema v46" in err
    assert plane.patches() == []
    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 256000)
    assert exc.value.code == "config_migration_conflict"

    write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 300000)
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 300000}
    remote_pkg._reset_for_tests()  # a fresh process reads it back
    assert read_raw_config()["compression"]["threshold_tokens"] == 300000


# --- profiles, TUI and RPC in remote mode -------------------------------------------------

@pytest.fixture
def profile_plane(tmp_path, monkeypatch):
    """``plane`` with a real profiles root (``Path.home()/.hermes``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        get_config_backend()._state(home)
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def test_fresh_profile_creation_succeeds_against_a_healthy_plane(profile_plane):
    """F6: the staging home is never read through the remote backend."""
    from hermes_cli.profiles import create_profile
    before = len(profile_plane.requests)

    path = create_profile("fresh", no_alias=True, no_skills=True)

    assert path.is_dir() and (path / ".env").exists()
    assert not list(path.parent.glob(".fresh.staging-*"))
    assert not any(r["profile"].startswith(".") for r in profile_plane.requests[before:])


def test_rename_is_refused_before_anything_moves(profile_plane, monkeypatch):
    """F7: the plane keys settings by profile name and has no rename."""
    from hermes_cli import profiles
    for name in ("_check_gateway_running", "_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_stop_bot_desktop", "_live_default_multiplexer", "_maybe_register_gateway_service"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: False)
    old = profile_plane.home / "profiles" / "old"
    old.mkdir(parents=True)
    profile_plane.profile("old").update(values={"model": {"default": "profile-model"}}, version=1)
    profile_plane.profile("new").update(values={"model": {"default": "target-model"}}, version=1)

    with pytest.raises(ValueError, match="not supported"):
        profiles.rename_profile("old", "new")

    assert old.is_dir() and not (profile_plane.home / "profiles" / "new").exists()
    assert profile_plane.profile("old")["values"] == {"model": {"default": "profile-model"}}
    assert profile_plane.patches() == []


def test_roster_model_follows_the_remote_layer_not_the_local_file(profile_plane):
    """F9: a poll changes the profile's model while an ignored local config.yaml stays put."""
    from hermes_cli.profiles import list_profiles
    named = profile_plane.home / "profiles" / "roster"
    named.mkdir(parents=True)
    (named / "config.yaml").write_text("model:\n  default: ignored-local\n")
    profile_plane.profile("roster").update(values={"model": {"default": "model-A", "provider": "nous"}}, version=1)

    def roster():
        return {p.name: (p.model, p.provider) for p in list_profiles()}["roster"]

    assert roster() == ("model-A", "nous")
    profile_plane.profile("roster").update(values={"model": {"default": "model-B", "provider": "openrouter"}}, version=2)
    backend = get_config_backend()
    assert backend.poll_one(backend._state(named))
    assert roster() == ("model-B", "openrouter")


def _tui_server(monkeypatch, home):
    import hermes_cli.banner as banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    from tui_gateway import server
    monkeypatch.setattr(server, "_hermes_home", home)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    return server


def test_tui_config_set_of_a_locked_key_is_refused_not_reported_saved(plane, monkeypatch):
    """F10: a keyed TUI edit under a lock answers with the lock, sends nothing, and the raw cache
    keeps the accepted value."""
    plane.upper = {"display": {"tui_theme": "dark"}}
    plane.upper_locks = [{"path": "display.tui_theme", "level": "tenant"}]
    server = _tui_server(monkeypatch, plane.home)

    answer = server._methods["config.set"](1, {"key": "theme", "value": "light"})

    assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
    assert plane.patches() == []
    assert server._load_cfg_raw()["display"]["tui_theme"] == "dark"

    plane.upper_locks = []
    backend = get_config_backend()
    assert backend.poll_one(backend._state(plane.home))
    assert server._methods["config.set"](2, {"key": "theme", "value": "light"})["result"]["value"] == "light"
    assert server._load_cfg_raw()["display"]["tui_theme"] == "light"
    assert plane.profile("default")["values"] == {"display": {"tui_theme": "light"}}


def test_rpc_refused_by_the_config_backend_still_gets_its_one_answer(plane, monkeypatch):
    """F11: ConfigBackendUnavailable is a SystemExit; an admitted request (inline or pooled) still
    answers once, with its own id."""
    import copy
    import threading

    server = _tui_server(monkeypatch, plane.home)

    def refused(rid, params):
        raise ConfigBackendUnavailable("Remote Config: cannot load the config for profile 'lazy'")

    monkeypatch.setitem(server._methods, "cc.refused", refused)
    inline = server.handle_request({"jsonrpc": "2.0", "id": "inline-1", "method": "cc.refused", "params": {}})
    assert inline["id"] == "inline-1" and "cannot load the config" in inline["error"]["message"]

    monkeypatch.setattr(server, "_LONG_HANDLERS", server._LONG_HANDLERS | {"cc.refused"})
    written = threading.Event()

    class Recorder:
        frames = []

        def write(self, obj):
            self.frames.append(copy.deepcopy(obj))
            written.set()
            return True

        def close(self):
            pass

    transport = Recorder()
    assert server.dispatch({"jsonrpc": "2.0", "id": "pooled-1", "method": "cc.refused", "params": {}}, transport) is None
    assert written.wait(10)
    (frame,) = transport.frames
    assert frame["id"] == "pooled-1" and "cannot load the config" in frame["error"]["message"]



# --- review round 4 regressions -----------------------------------------------------------

def _absent_section_race(plane, monkeypatch):
    """Another writer adds ``terminal.persistent`` to a profile with no ``terminal`` section
    between our read and our first PATCH (so that PATCH gets a 409)."""
    from plugins.config_backends.remote import client
    plane.profile("default").update(values={"display": {"personality": "concise"}}, version=1)
    get_config_backend().read_user_layer(plane.home)
    real_request = client.request
    raced = []

    def other_writer_lands_first(method, *args, **kwargs):
        if method == "PATCH" and not raced:
            raced.append(True)
            prof = plane.profile("default")
            prof["values"]["terminal"] = {"persistent": True}
            prof["version"] += 1
        return real_request(method, *args, **kwargs)

    monkeypatch.setattr(client, "request", other_writer_lands_first)


@pytest.mark.parametrize("entry", ["set_config_value", "write_config_key"])
def test_cas_retry_keeps_a_sibling_added_to_a_section_the_doc_lacked(plane, monkeypatch, entry):
    """Round 4 #1: our edit adds terminal.timeout to a doc with no terminal section; the 409 retry
    must add that one key to the re-read doc, not replace the section another writer created."""
    from hermes_cli.config import set_config_value
    _absent_section_race(plane, monkeypatch)

    if entry == "set_config_value":
        set_config_value("terminal.timeout", "321")
    else:
        write_config_key(plane.home / "config.yaml", "terminal.timeout", 321)

    assert plane.profile("default")["values"]["terminal"] == {"persistent": True, "timeout": 321}
    first, retry = plane.patches()
    assert retry["body"]["expectedVersion"] == 2
    assert retry["body"]["set"] == {"terminal.timeout": 321} and "unset" not in retry["body"]


def test_cas_retry_of_a_new_empty_section_keeps_a_concurrent_one(plane, monkeypatch):
    """An empty mapping the edit adds means "make this a mapping": on retry it must not replace
    the mapping (with keys) another writer created meanwhile."""
    _absent_section_race(plane, monkeypatch)
    from hermes_cli.config_backend import Changes
    get_config_backend().write_changes(plane.home, Changes(set={"terminal": {}}))
    assert plane.profile("default")["values"]["terminal"] == {"persistent": True}
    assert len(plane.patches()) == 1  # nothing left to send after the re-read: one PATCH, the 409


def test_explicit_section_replacement_stays_a_replacement(plane):
    """Control: replacing a present section with a scalar, or unsetting it, is still sent whole."""
    from hermes_cli.config_backend import Changes
    plane.profile("default").update(values={"terminal": {"persistent": True, "timeout": 3}}, version=1)
    backend = get_config_backend()
    backend.write_changes(plane.home, Changes(unset=("terminal",)))
    assert plane.patches()[-1]["body"]["unset"] == ["terminal"]
    assert "terminal" not in plane.profile("default")["values"]


def _routed_profile(profile_plane, source_env_line):
    named = profile_plane.home / "profiles" / "routed"
    named.mkdir(parents=True)
    (named / ".env").write_text("# synthetic\n")
    profile_plane.profile("routed").update(values={"secrets": {"command": {
        "enabled": True, "override_existing": True,
        "command": f"printf '{source_env_line}\\n'"}}}, version=1)
    return named


def test_routed_hydration_refuses_a_source_that_supplies_a_protected_name(profile_plane, monkeypatch):
    """Round 4 #2: a routed profile's remote secrets: source supplies HERMES_PORTAL_BASE_URL (where
    auth.json's refresh token is sent). Refused before any snapshot, ownership or scope publication;
    every retry refuses again; the profile hydrates once the mapping is gone."""
    import os

    from hermes_cli import env_loader
    from hermes_cli.web_server_profiles import _config_profile_scope
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", set())
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    named = _routed_profile(profile_plane, "HERMES_PORTAL_BASE_URL=http://127.0.0.1:1")

    for _attempt in range(2):  # nothing is cached by a refusal: a retry refuses again
        with pytest.raises(ConfigBackendUnavailable, match="HERMES_PORTAL_BASE_URL"):
            with _config_profile_scope("routed"):
                pass
        assert env_loader.get_secret_source_values(named) == {}
        assert str(named.resolve()) not in env_loader._APPLIED_HOMES
        assert "HERMES_PORTAL_BASE_URL" not in env_loader._SOURCE_SUPPLIED_NAMES
        assert "HERMES_PORTAL_BASE_URL" not in os.environ

    profile_plane.profile("routed").update(values={"secrets": {"command": {
        "enabled": True, "override_existing": True, "command": "printf 'SYNTHETIC_KEY=routed-ok\\n'"}}}, version=2)
    backend = get_config_backend()
    assert backend.poll_one(backend._state(named))
    assert env_loader.hydrate_profile_secret_sources(named) == {"SYNTHETIC_KEY": "routed-ok"}


def test_file_mode_routed_hydration_drops_only_the_selector(monkeypatch, tmp_path, capsys):
    """Control: under the file backend a routed source may not switch the backend either, but its
    other values (a Portal URL included: file mode has no plane credential) still hydrate."""
    from hermes_cli import env_loader
    for name in ("HERMES_CONFIG_BACKEND", "HERMES_CONFIG_REMOTE_URL", "HERMES_CONFIG_INSTANCE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    home = tmp_path / "routed"
    home.mkdir()
    (home / ".env").write_text("# synthetic\n")
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    override_existing: true\n"
        "    command: \"printf 'HERMES_CONFIG_BACKEND=remote\\\\nHERMES_PORTAL_BASE_URL=http://portal.test\\\\n'\"\n")

    values = env_loader.hydrate_profile_secret_sources(home)

    assert values == {"HERMES_PORTAL_BASE_URL": "http://portal.test"}
    assert "ignored HERMES_CONFIG_BACKEND" in capsys.readouterr().err


_LEGACY_SOUL = "# Beta identity\n\n## Messaging other agents\nKEEP THIS PROFILE SECTION\n"


def _sibling_with_legacy_soul(home):
    beta = home / "profiles" / "beta"
    beta.mkdir(parents=True)
    (beta / "SOUL.md").write_text(_LEGACY_SOUL)
    return beta / "SOUL.md"


def test_remote_read_migration_changes_no_local_file(plane):
    """Round 4 #3: reading default's remote config stamped 40 runs the 40->41 step in memory. That
    step's only effect is on profile SOUL.md files; a config read must not rewrite a sibling's."""
    from hermes_cli.config_backend import read_config_doc
    soul = _sibling_with_legacy_soul(plane.home)
    plane.profile("default").update(values={"display": {"personality": "old"}}, writer=40, version=1)

    doc = read_config_doc(plane.home / "config.yaml")

    assert doc["_config_version"] == backend_mod._latest_config_version()  # it did migrate
    assert soul.read_text() == _LEGACY_SOUL
    assert plane.patches() == []


def test_refused_write_validation_changes_no_local_file(plane):
    """Round 4 #3: the config_migration_conflict check simulates the migration of the stored doc;
    a refused edit must leave every local file as it was."""
    from hermes_cli.config import read_raw_config
    plane.profile("default").update(values={}, writer=40, version=1)
    read_raw_config()
    soul = _sibling_with_legacy_soul(plane.home)  # created after the read's own migration

    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 256000)

    assert exc.value.code == "config_migration_conflict"
    assert soul.read_text() == _LEGACY_SOUL
    assert plane.patches() == []


def _home_files(root):
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


@pytest.mark.parametrize("config_only", [True, False])
def test_config_only_migration_ladder_touches_no_file(tmp_path, monkeypatch, config_only):
    """The whole ladder from the support floor, config-only, over a home whose files every
    file-touching step would change (.env dead/legacy values, a legacy SOUL.md section, no logs/):
    the only output is the config document. Control: a file-mode run does change them."""
    from hermes_cli import config as config_mod
    from hermes_cli import config_migrations
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_CONFIG_BACKEND", "LLM_MODEL", "OPENAI_MODEL", "TERMINAL_VERCEL_RUNTIME"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli.config_defaults import LEGACY_VERCEL_RUNTIME
    (home / ".env").write_text(f"LLM_MODEL=old-model\nTERMINAL_VERCEL_RUNTIME={LEGACY_VERCEL_RUNTIME}\n")
    (home / "SOUL.md").write_text(_LEGACY_SOUL)
    _sibling_with_legacy_soul(home)
    doc = {"_config_version": config_migrations.SUPPORT_FLOOR_VERSION, "model": {"default": "m"}}
    persisted = []
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: dict(persisted[-1] if persisted else doc))
    monkeypatch.setattr(config_mod, "_persist_migration", lambda cfg: persisted.append(dict(cfg)))
    before = _home_files(home)

    config_migrations.run_migrations(config_migrations.SUPPORT_FLOOR_VERSION,
                                     {"env_added": [], "config_added": [], "warnings": []}, True,
                                     config_only=config_only)

    assert persisted  # the document itself was migrated either way
    assert (_home_files(home) == before) is config_only



def test_concurrent_first_reader_waits_for_the_bootstrap(plane, monkeypatch, tmp_path):
    """Round 4 #4: while one thread publishes the deployment from .env, a concurrent first reader
    must wait for it, not select the file backend and read the local config.yaml meanwhile."""
    import threading

    from hermes_cli import config_backend, env_loader
    from hermes_cli.config_backend import read_config_doc
    _drop_process_deployment(plane, monkeypatch, tmp_path)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in remote_env(plane).items()))
    (plane.home / "config.yaml").write_text("display:\n  personality: forbidden-local\n")
    plane.profile("default").update(values={"display": {"personality": "remote"}}, version=1)
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", False)
    entered, resume = threading.Event(), threading.Event()
    real_apply = env_loader.apply_config_bootstrap_env

    def paused(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(env_loader, "apply_config_bootstrap_env", paused)
    results = {}

    def run(name, fn):
        threading.Thread(target=lambda: results.__setitem__(name, fn()), name=name, daemon=True).start()

    run("first", lambda: get_config_backend().name)
    assert entered.wait(10)
    run("concurrent", lambda: read_config_doc(plane.home / "config.yaml")["display"]["personality"])
    import time
    time.sleep(0.3)
    assert "concurrent" not in results  # waiting for the bootstrap, not reading the local file
    resume.set()
    deadline = time.monotonic() + 10
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert results == {"first": "remote", "concurrent": "remote"}


def test_same_thread_reentry_during_the_bootstrap_does_not_deadlock(monkeypatch):
    """Control: an import-time config read on the bootstrapping thread itself returns at once."""
    from hermes_cli import config_backend, env_loader
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", False)
    seen = []

    def reentrant(*args, **kwargs):
        seen.append(config_backend.get_config_backend().name)  # would block on a plain Lock

    monkeypatch.setattr(env_loader, "apply_config_bootstrap_env", reentrant)
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    assert config_backend.get_config_backend().name == "file"
    assert seen == ["file"] and config_backend._BOOTSTRAPPED


def _fresh_interpreter_boot(tmp_path, env, project_env):
    import os
    import subprocess
    import sys
    code = ("import json\n"
            "from hermes_cli.env_loader import load_hermes_dotenv\n"
            f"load_hermes_dotenv(project_env={str(project_env)!r}, load_external_secrets=False)\n"
            "from hermes_cli.config_backend import get_config_backend\n"
            "from hermes_cli.config import load_config\n"
            "print('RESULT=' + json.dumps([get_config_backend().name, load_config()['display']['personality']]))\n")
    child = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR", "SYSTEMROOT")}
    child.update(env, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    return subprocess.run([sys.executable, "-c", code], env=child, capture_output=True, text=True,
                          timeout=120, stdin=subprocess.DEVNULL, cwd=str(tmp_path))


def test_plane_credential_from_the_project_dotenv_boots(plane, tmp_path):
    """Round 4 #5: user .env selects remote and names the plane and instance; the project .env (a
    supported load_hermes_dotenv layer) holds the IdP credential. A fresh process boots and fetches:
    the early bootstrap sees the project layer before the sanitizer's config import reads config."""
    env = remote_env(plane)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("HERMES_CONFIG_")))
    project = tmp_path / "project.env"
    project.write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("GATEWAY_RELAY_IDP_")))
    (plane.home / "config.yaml").write_text("display:\n  personality: forbidden-local\n")
    plane.profile("default").update(values={"display": {"personality": "remote"}}, version=1)

    proc = _fresh_interpreter_boot(tmp_path, {"HERMES_HOME": str(plane.home), "HOME": str(tmp_path)}, project)

    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")][-1]
    assert json.loads(line[len("RESULT="):]) == ["remote", "remote"]
    assert _gets(plane)


def test_bootstrap_keeps_the_dotenv_precedence(tmp_path, monkeypatch):
    """The bootstrap composes the layers as load_hermes_dotenv does: user .env overrides the
    process, the project .env only fills gaps when a user .env exists, managed .env wins last."""
    from hermes_cli.env_loader import _bootstrap_env
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("CC_A", "process")
    monkeypatch.setenv("CC_D", "process")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    (home / ".env").write_text("CC_A=user\nCC_B=user\n")
    (home / ".op.env").write_text("CC_B=op\nCC_E=op\n")
    project = tmp_path / "project.env"
    project.write_text("CC_B=project\nCC_C=project\nCC_D=project\n")
    (managed / ".env").write_text("CC_C=managed\n")

    env = _bootstrap_env(home, project)

    assert (env["CC_A"], env["CC_B"], env["CC_C"], env["CC_D"], env["CC_E"]) == (
        "user", "user", "managed", "process", "op")
    (home / ".env").unlink()  # no user .env: the project layer overrides the process
    assert _bootstrap_env(home, project)["CC_D"] == "project"


def test_tui_section_and_prompt_setters_refuse_locked_edits(plane, monkeypatch):
    """Round 4 #6: prompt / reasoning / details_mode / details_mode.<section> under an upper lock
    answer with the lock, send nothing, and leave the session as it was. Unlocked, they write."""
    plane.upper = {"custom_prompt": "tenant-prompt",
                   "display": {"show_reasoning": False, "sections": {"thinking": "hidden"}, "details_mode": "collapsed"}}
    plane.upper_locks = [{"path": "custom_prompt", "level": "tenant"}, {"path": "display", "level": "group"}]
    server = _tui_server(monkeypatch, plane.home)
    session = {"show_reasoning": False, "session_key": "k"}
    server._sessions["s1"] = session
    try:
        _locked_then_unlocked_tui_setters(plane, server, session)
    finally:
        server._sessions.pop("s1", None)  # never torn down by the TUI fixture after the plane stops


def _locked_then_unlocked_tui_setters(plane, server, session):
    calls = [("prompt", "synthetic-new-prompt"), ("prompt", "clear"), ("reasoning", "show"),
             ("details_mode", "expanded"), ("details_mode.thinking", "expanded"), ("details_mode.thinking", "")]

    for rid, (key, value) in enumerate(calls):
        answer = server._methods["config.set"](rid, {"key": key, "value": value, "session_id": "s1"})
        assert answer.get("error", {}).get("code") == 4002, (key, value, answer)
        assert "locked" in answer["error"]["message"]
    assert plane.patches() == []
    assert session["show_reasoning"] is False
    raw = server._load_cfg_raw()
    assert raw["custom_prompt"] == "tenant-prompt" and raw["display"]["show_reasoning"] is False

    plane.upper_locks = []
    backend = get_config_backend()
    assert backend.poll_one(backend._state(plane.home))
    for rid, (key, value) in enumerate(calls[:1] + calls[2:5]):
        assert "result" in server._methods["config.set"](100 + rid, {"key": key, "value": value, "session_id": "s1"})
    stored = plane.profile("default")["values"]
    assert stored["custom_prompt"] == "synthetic-new-prompt"
    assert stored["display"]["show_reasoning"] is True and stored["display"]["details_mode"] == "expanded"
    assert stored["display"]["sections"]["thinking"] == "expanded"
    assert session["show_reasoning"] is True
    assert all("unset" not in p["body"] for p in plane.patches())


def test_tui_shared_metrics_set_refuses_a_locked_consent(plane, monkeypatch):
    """Round 4 #6 (same class): the consent setter is explicit too; a locked answer is refused,
    not saved-minus-the-lock while the consent bookkeeping records it."""
    plane.upper = {"telemetry": {"shared_metrics": {"enabled": False, "send": False}}}
    plane.upper_locks = [{"path": "telemetry.shared_metrics", "level": "tenant"}]
    server = _tui_server(monkeypatch, plane.home)
    recorded = []
    from hermes_cli import setup as setup_mod
    monkeypatch.setattr(setup_mod, "_record_send_consent_change", lambda **kw: recorded.append(kw))

    answer = server._methods["shared_metrics.set"](1, {"enabled": True, "send": True})

    assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
    assert plane.patches() == [] and recorded == []
