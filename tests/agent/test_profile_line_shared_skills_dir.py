"""The profile line must not forbid writes to the directory the config hands over.

``skills.create_dir`` may point at the canonical root, which makes the root's
``skills/`` this session's own write target. The profile line names the same
path as another session's data and says "Do NOT modify", so the agent refuses
work it is configured to own.
"""

from pathlib import Path
from unittest.mock import patch


def _profile_line(root: Path, profile: str, create_dir):
    """Render the line for a session bound to ``root/profiles/<profile>``."""
    import agent.system_prompt as system_prompt

    profile_home = root / "profiles" / profile
    profile_home.mkdir(parents=True, exist_ok=True)

    class _Agent:
        hermes_home = str(profile_home)

    create_dir_path = None
    if create_dir is not None:
        create_dir_path = (
            root / create_dir if not create_dir.startswith("/") else Path(create_dir)
        )
        create_dir_path.mkdir(parents=True, exist_ok=True)

    with patch("agent.system_prompt.get_hermes_home", return_value=profile_home), \
         patch("agent.system_prompt.get_default_hermes_root", return_value=root), \
         patch("agent.system_prompt._active_profile_name", return_value=profile), \
         patch("agent.system_prompt._agent_home", return_value=profile_home), \
         patch("agent.skill_utils.get_skill_create_dir", return_value=create_dir_path):
        return system_prompt._active_profile_line(_Agent())


def _guarded_paths(line: str) -> str:
    """The stretch of the line that enumerates another session's directories."""
    assert "The default profile's data lives at" in line, line
    return line.split("The default profile's data lives at", 1)[1].split("—", 1)[0]


def test_unset_create_dir_keeps_today_wording(tmp_path):
    """No create_dir configured: the line must stay byte-for-byte what it was."""
    root = tmp_path / ".hermes"
    root.mkdir()

    line = _profile_line(root, "coder", None)

    assert line == (
        f"Active Hermes profile: coder. This session reads and writes "
        f"{root}/profiles/coder/. The default profile's data lives at "
        f"{root}/skills/, {root}/plugins/, {root}/cron/, {root}/memories/ — "
        f"those belong to a different session run from a different shell. "
        f"Do NOT modify another profile's skills/plugins/cron/memories unless "
        f"the user explicitly directs you to."
    )


def test_create_dir_at_root_is_named_writable(tmp_path):
    """create_dir at the canonical root hands the root's skills/ to this session."""
    root = tmp_path / ".hermes"
    root.mkdir()

    line = _profile_line(root, "coder", "skills")

    shared = f"{root}/skills"
    # The configured write root is named as this session's own.
    assert shared in line
    # ...and it no longer sits in the enumerated set of another session's dirs.
    guarded = _guarded_paths(line)
    assert shared not in guarded
    # The rest of the root stays another session's business.
    assert f"{root}/plugins/" in guarded
    assert f"{root}/cron/" in guarded
    assert f"{root}/memories/" in guarded
    assert "Do NOT modify" in line


def test_create_dir_elsewhere_leaves_the_root_guarded(tmp_path):
    """A create_dir that is not under the root must not unlock the root's skills/."""
    root = tmp_path / ".hermes"
    root.mkdir()
    shared = tmp_path / "team-skills"

    line = _profile_line(root, "coder", str(shared))

    guarded = _guarded_paths(line)
    assert f"{root}/skills/" in guarded
    assert f"{root}/plugins/" in guarded
    assert "Do NOT modify" in line
