"""``gateway.standalone`` and the multiplexer refusal: ``_named_profile_refused_under_multiplexer``.

A profile that authored ``gateway.standalone: true`` runs (or may run) its own gateway without
``--force``; the only refusal left is a RUNNING host record that still lists it (the host gateway
has not rescanned since the key was set) — and that refusal names the rescan instead of ``--force``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

import hermes_constants


@pytest.fixture
def standalone_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    home = root / "profiles" / "coder"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from hermes_cli import gateway as gw
    # The probe seams live on the hermes_cli.gateway facade, like the other refusal tests.
    monkeypatch.setattr(gw, "_is_service_installed", lambda: False)
    monkeypatch.setattr(gw, "_served_by_another_host_gateway", lambda name=None: None)
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda name=None: False)
    return gw, home


def _refusal(gw) -> tuple[bool, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        refused = gw._named_profile_refused_under_multiplexer()
    return refused, buf.getvalue()


def test_standalone_named_home_is_not_refused_without_force(standalone_home):
    gw, _home = standalone_home
    refused, out = _refusal(gw)
    assert refused is False
    assert out == ""


def test_standalone_named_home_still_served_by_host_record_is_refused_with_rescan(standalone_home, monkeypatch):
    gw, _home = standalone_home
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda name=None: True)
    refused, out = _refusal(gw)
    assert refused is True
    assert "still serves profile 'coder'" in out
    assert "gateway.standalone" in out
    assert "rescan-profiles" in out and "30s" in out
    assert "--force" not in out


def test_non_standalone_refusal_names_the_opt_out(standalone_home):
    gw, home = standalone_home
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    refused, out = _refusal(gw)
    assert refused is True
    assert "Or opt this profile out of the host gateway for good" in out
    assert "gateway.standalone: true" in out


def test_setup_stale_host_record_names_rescan(standalone_home, monkeypatch, capsys):
    gw, _home = standalone_home
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda name=None: True)
    assert gw._served_profile_needs_no_service() is True
    assert "rescan-profiles" in capsys.readouterr().out


def test_dashboard_standalone_refusal_resolves_once_and_preserves_invalid_profile(standalone_home, monkeypatch):
    from fastapi import HTTPException
    from hermes_cli import web_server_gateway as web
    from hermes_cli import web_server_profiles

    _gw, home = standalone_home
    resolved = []

    def resolve(name):
        resolved.append(name)
        if name != "coder":
            raise HTTPException(status_code=404, detail="Unknown profile")
        return home

    monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", resolve)
    monkeypatch.setattr(web, "_profile_is_multiplexed", lambda name: True)
    assert "rescan-profiles" in web.multiplexed_profile_refusal("coder", "start")
    assert resolved == ["coder"]
    monkeypatch.setattr(web, "_profile_is_multiplexed", lambda name: False)
    assert web.multiplexed_profile_refusal("coder", "start") is None
    with pytest.raises(HTTPException) as exc:
        web.multiplexed_profile_refusal("missing", "stop")
    assert exc.value.status_code == 404
