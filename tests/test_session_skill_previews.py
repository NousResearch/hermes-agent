"""Session previews must never surface a /skill's own body.

`preview` is the head of the first user message, and it is the TITLE FALLBACK on
every surface (sidebar rows, pickers, exports, desktop `sessionTitle`). A /skill
invocation expands into a message that embeds the whole skill body, so an
untitled skill session used to read `[IMPORTANT: The user has invoked the "work"
skill, indicatin...` in the sidebar.

These drive the real SQL/shaping path through SessionDB rather than calling the
shaper directly, so the CASE expression and the Python side are covered together.
"""

import pytest

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from hermes_state import SessionDB

SKILL_BODY = (
    "Kick off a task in a fresh isolated git worktree instead of the current checkout. "
    "Look at what the repo already does, and copy it. Create the worktree and branch. "
)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _install_skill(tmp_path, monkeypatch, name="work", body=SKILL_BODY):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    skill_commands.scan_skill_commands()
    return skills_dir


def _seed(db, session_id, content, *, title=None, reply="On it."):
    db.create_session(session_id=session_id, source="cli", model="m")
    db.append_message(session_id, role="user", content=content)
    db.append_message(session_id, role="assistant", content=reply)
    if title:
        db.set_session_title(session_id, title)


class TestSkillPreview:


    def test_skill_preview_shows_the_typed_instruction(self, db, tmp_path, monkeypatch):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        _seed(db, "s1", message)
        (row,) = db.list_sessions_rich(limit=10)
        assert row["preview"] == "/work — fix the title leak"

    def test_skill_preview_never_leaks_the_body(self, db, tmp_path, monkeypatch):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        _seed(db, "s1", message)
        (row,) = db.list_sessions_rich(limit=10)
        assert "IMPORTANT" not in row["preview"]
        assert "worktree" not in row["preview"]

    def test_bare_skill_preview_is_the_command(self, db, tmp_path, monkeypatch):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message("/work")
        _seed(db, "s1", message)
        (row,) = db.list_sessions_rich(limit=10)
        assert row["preview"] == "/work"



    def test_rewind_picker_shows_the_typed_instruction(
        self, db, tmp_path, monkeypatch
    ):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        _seed(db, "s1", message)
        (entry,) = db.list_recent_user_messages("s1")
        assert entry["preview"] == "/work — fix the title leak"


class TestSkillScaffoldedSessionLookup:
    """Backing queries for `hermes sessions retitle-skills`."""

    def test_finds_only_titled_skill_sessions(self, db, tmp_path, monkeypatch):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        _seed(db, "titled", message, title="Isolated Git Worktree Setup")
        _seed(db, "untitled", message)
        _seed(db, "plain", "fix the title leak", title="Fixing The Title Leak")

        rows = db.list_skill_scaffolded_sessions()
        assert [row["id"] for row in rows] == ["titled"]
        assert rows[0]["title"] == "Isolated Git Worktree Setup"
        # The full first turn comes back so the caller can re-derive the ask.
        assert "fix the title leak" in rows[0]["content"]

    def test_limit_is_honored(self, db, tmp_path, monkeypatch):
        _install_skill(tmp_path, monkeypatch)
        message = skill_commands.build_skill_invocation_message("/work")
        for i in range(3):
            _seed(db, f"s{i}", message, title=f"Title {i}")
        assert len(db.list_skill_scaffolded_sessions(limit=2)) == 2


