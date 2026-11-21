"""An inherited cwd is provenance, not a pick — a foreign one must never refuse the create.

The Desktop rides its remembered app-global workspace as ``cwd`` with ``cwd_explicit: false`` on
every ``session.create``. When the gateway runs on a different host than the client (a Windows
Desktop dialing a Linux container), that path is a foreign spelling that can never pass the host
``is_dir()`` check — and ``build_policy`` validated BEFORE honoring the ``cwd_explicit: false``
override, so the create died with ``invalid_params field=cwd`` and no chat could be created at
all. An explicit pick stays strict; an inherited cwd that does exist on this host is kept as-is.
"""
import pytest

import gateway.session_policy as sp
from persistence.core.errors import AuthorityRefusal

FOREIGN_CWD = r"C:\Users\markl\hermes"
CFG = {"platform_toolsets": {"gui": []}}


@pytest.mark.platforms("linux", "macos")
def test_an_inherited_foreign_cwd_falls_back_to_the_host_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_profile_workspace_cwd", lambda home: None)
    monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: tmp_path)
    policy = sp.build_policy({"source": "gui", "cwd": FOREIGN_CWD, "cwd_explicit": False,
                              "model": "fixture"}, CFG)
    assert policy.cwd == str(tmp_path)


@pytest.mark.platforms("linux", "macos")
def test_a_profile_workspace_beats_an_inherited_foreign_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "profile-workspace"
    workspace.mkdir()
    monkeypatch.setattr(sp, "_profile_workspace_cwd", lambda home: str(workspace))
    policy = sp.build_policy({"source": "gui", "cwd": FOREIGN_CWD, "cwd_explicit": False,
                              "model": "fixture"}, CFG)
    assert policy.cwd == str(workspace)


@pytest.mark.platforms("linux", "macos")
def test_an_inherited_cwd_that_exists_here_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_profile_workspace_cwd", lambda home: None)
    kept = tmp_path / "real-workspace"
    kept.mkdir()
    policy = sp.build_policy({"source": "gui", "cwd": str(kept), "cwd_explicit": False,
                              "model": "fixture"}, CFG)
    assert policy.cwd == str(kept)


@pytest.mark.platforms("linux", "macos")
def test_an_explicit_pick_of_a_foreign_cwd_is_still_refused(monkeypatch):
    with pytest.raises(AuthorityRefusal, match="invalid_params") as caught:
        sp.build_policy({"source": "gui", "cwd": FOREIGN_CWD, "cwd_explicit": True,
                         "model": "fixture"}, CFG)
    assert caught.value.field == "cwd"


@pytest.mark.platforms("linux", "macos")
def test_a_legacy_spelling_without_cwd_explicit_is_still_refused(monkeypatch):
    """No ``cwd_explicit`` at all: the value is a deliberate pick, strictness unchanged."""
    with pytest.raises(AuthorityRefusal, match="invalid_params") as caught:
        sp.build_policy({"source": "gui", "cwd": FOREIGN_CWD, "model": "fixture"}, CFG)
    assert caught.value.field == "cwd"
