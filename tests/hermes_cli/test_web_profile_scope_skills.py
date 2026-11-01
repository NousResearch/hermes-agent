"""A dashboard ``?profile=B`` request never moves another profile's skill reads/writes into B.

``_profile_scope(B)`` used to retarget the process-global ``SKILLS_DIR`` of ``tools.skills_tool``
and ``tools.skill_manager_tool`` for the whole request. Readers never took the lock, and
``_skills_dir()`` prefers a patched global over the live profile home (#40677), so an agent turn
for the DEFAULT profile overlapping any profile-scoped request (skills list, model/MCP/config
routes, console) created its new skill in B and loaded B's skills instead of its own.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    b = root / "profiles" / "b"
    (b / "skills").mkdir(parents=True)
    (root / "skills").mkdir()
    for home in (root, b):
        (home / "config.yaml").write_text("model:\n  default: openai/gpt-4o-mini\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    from agent import secret_scope
    from tui_gateway import launch_profile_policy as lpp
    # A named-profile scope flips the process to multi-profile hosting; undo it after the test.
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(lpp, "_snapshot", None)
    return root, b


@contextmanager
def _profile_b_request_in_flight():
    """Hold ``_profile_scope("b")`` on another thread, as a ``?profile=b`` route handler does."""
    from hermes_cli.web_server_profiles import _profile_scope

    entered, release, errors = threading.Event(), threading.Event(), []

    def handler():
        try:
            with _profile_scope("b"):
                entered.set()
                release.wait(30)
        except BaseException as exc:  # surface failures instead of a silent timeout
            errors.append(exc)
            entered.set()

    t = threading.Thread(target=handler, daemon=True)
    t.start()
    assert entered.wait(30) and not errors, errors
    try:
        yield
    finally:
        release.set()
        t.join(30)


@contextmanager
def _default_profile_turn(root: Path):
    """An agent turn bound to the default profile's home (as every session is)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(root))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_skill_created_by_default_profile_turn_stays_in_default_profile(homes):
    from tools.skill_manager_tool import skill_manage

    root, b = homes
    content = "---\nname: default-made\ndescription: made by a default-profile turn\n---\n\nbody\n"
    with _profile_b_request_in_flight(), _default_profile_turn(root):
        result = skill_manage(action="create", name="default-made", content=content)

    assert '"success": true' in result, result
    assert (root / "skills" / "default-made" / "SKILL.md").is_file()
    assert not list((b / "skills").rglob("default-made"))


def test_default_profile_turn_cannot_load_profile_b_skill_during_b_request(homes):
    from hermes_cli.web_server_profiles import _profile_scope
    from tools.skills_tool import skill_view

    root, b = homes
    skill = b / "skills" / "b-only"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: b-only\ndescription: only in profile b\n---\n\nB-ONLY-BODY\n", encoding="utf-8")

    with _profile_b_request_in_flight(), _default_profile_turn(root):
        assert "B-ONLY-BODY" not in str(skill_view("b-only"))
    # The request's own scope still resolves profile b's skills.
    with _profile_scope("b"):
        assert "B-ONLY-BODY" in str(skill_view("b-only"))
