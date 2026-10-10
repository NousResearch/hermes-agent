"""``onboarding.run`` lives in the root profile's config.yaml: unset runs the questionnaire only on a
fresh install, an explicit bool wins, and a backend launched under a named profile reads and writes
the same root file."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import onboarding_run, profiles
from hermes_cli.config import read_user_config_raw
from hermes_state import SessionDB


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _add_session(home: Path) -> None:
    db = SessionDB(home / "state.db")
    try:
        db.create_session("old-session", "desktop")
    finally:
        db.close()


def _launch_under(monkeypatch, root: Path, name: str) -> Path:
    """A pooled backend launched under ``-p <name>``: HERMES_HOME is the profile's home."""
    home = profiles.create_profile(name, no_alias=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert onboarding_run.get_default_hermes_root() == root
    return home


def test_fresh_install_runs(root):
    SessionDB(root / "state.db").close()  # a boot leaves an empty state.db behind

    assert onboarding_run.should_run() is True


def test_root_with_a_session_does_not_run(root):
    _add_session(root)

    assert onboarding_run.should_run() is False


def test_a_user_profile_is_history_but_a_setup_profile_is_not(root):
    setup = profiles.create_profile("hermes-setup", no_alias=True)
    (setup / ".setup-profile.json").write_text("{}")
    assert onboarding_run.should_run() is True

    profiles.create_profile("work", no_alias=True)
    assert onboarding_run.should_run() is False


def test_a_managed_run_false_wins_on_a_fresh_install(root, tmp_path, monkeypatch):
    # IT can pin the questionnaire off through the managed overlay; the root file stays unset.
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("onboarding:\n  run: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()

    assert onboarding_run.should_run() is False


def test_explicit_true_runs_despite_history(root):
    _add_session(root)
    onboarding_run.set_run(True)

    assert onboarding_run.should_run() is True


def test_set_run_false_persists_and_keeps_comments(root):
    (root / "config.yaml").write_text("# my notes\nmodel:\n  default: some-model  # pinned\n", encoding="utf-8")

    onboarding_run.set_run(False)

    text = (root / "config.yaml").read_text(encoding="utf-8")
    assert "# my notes" in text and "# pinned" in text
    assert read_user_config_raw(root / "config.yaml")["model"] == {"default": "some-model"}
    assert onboarding_run.should_run() is False


def test_named_profile_launch_uses_the_root_file(root, monkeypatch):
    work = _launch_under(monkeypatch, root, "work")

    onboarding_run.set_run(True)

    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is True
    assert "run" not in (read_user_config_raw(work / "config.yaml").get("onboarding") or {})
    assert onboarding_run.should_run() is True

    (work / "config.yaml").write_text("onboarding:\n  run: true\n", encoding="utf-8")
    onboarding_run.set_run(False)
    assert onboarding_run.should_run() is False


def test_named_profile_launch_reads_the_root_state_db(root, monkeypatch):
    # Launched under the setup profile, the one named profile that is not history.
    setup = _launch_under(monkeypatch, root, "hermes-setup")
    (setup / ".setup-profile.json").write_text("{}")
    _add_session(setup)
    assert onboarding_run.should_run() is True

    _add_session(root)
    assert onboarding_run.should_run() is False
