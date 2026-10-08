"""Remote config: key-level CAS writes, locks, value checks and the dashboard's HTTP mapping."""
from __future__ import annotations

import json

import pytest

from hermes_cli.config_backend import (
    ConfigLockedError,
    ConfigValueError,
    get_config_backend,
    write_config_key,
)
from plugins.config_backends.remote import backend as backend_mod

from .conftest import _config_cmd


def test_file_backend_has_no_protected_names(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    _refuse_protected_env_from_sources(
        SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()}, sources=[]))


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


@pytest.mark.parametrize("stored", ["latest", None])
def test_write_stamps_a_current_or_unstamped_profile_level(plane, stored):
    from hermes_cli.config import set_config_value
    latest = backend_mod._latest_config_version()
    plane.profile("default")["writer"] = latest if stored == "latest" else None

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert patch["body"]["writerConfigVersion"] == latest
    assert plane.profile("default")["writer"] == latest


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
