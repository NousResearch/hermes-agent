"""Regression coverage for #12260.

``agent.skill_utils.resolve_skill_config_values`` is what surfaces skill
config values into the agent's prompt.  Subprocess tools (Bash, background
launchers, terminal) later act on those values — and those subprocesses
run with ``HOME={HERMES_HOME}/home`` when that directory exists (see
``hermes_constants.get_subprocess_home``), *not* the Python process's
own ``HOME``.

Before the fix, ``~/wiki`` in a skill default expanded via
``os.path.expanduser`` against the Python-process ``HOME``, producing a
path that subprocess tools couldn't find.  After the fix, the expansion
honours the subprocess HOME when active, and falls through to the
original behaviour when no subprocess HOME is set (``~/.hermes/home``
doesn't exist).
"""

import os
from unittest.mock import patch

import pytest

from agent.skill_utils import (
    SKILL_CONFIG_PREFIX,
    _expanduser_for_subprocess,
    resolve_skill_config_values,
)


# ---------------------------------------------------------------------------
# _expanduser_for_subprocess — unit coverage
# ---------------------------------------------------------------------------


class TestExpanduserForSubprocess:
    def test_tilde_slash_uses_subprocess_home_when_active(self):
        with patch(
            "agent.skill_utils.get_subprocess_home",
            return_value="/opt/data/home",
        ):
            assert _expanduser_for_subprocess("~/wiki") == "/opt/data/home/wiki"

    def test_bare_tilde_uses_subprocess_home_when_active(self):
        with patch(
            "agent.skill_utils.get_subprocess_home",
            return_value="/opt/data/home",
        ):
            assert _expanduser_for_subprocess("~") == "/opt/data/home"

    def test_nested_tilde_path_uses_subprocess_home(self):
        with patch(
            "agent.skill_utils.get_subprocess_home",
            return_value="/opt/data/home",
        ):
            assert (
                _expanduser_for_subprocess("~/cache/notes")
                == "/opt/data/home/cache/notes"
            )

    def test_absolute_path_unaffected(self):
        with patch(
            "agent.skill_utils.get_subprocess_home",
            return_value="/opt/data/home",
        ):
            assert _expanduser_for_subprocess("/var/data/wiki") == "/var/data/wiki"

    def test_tilde_user_falls_through_to_os_expanduser(self):
        """``~user`` addresses a specific OS user and must defer to the
        passwd database; subprocess HOME shouldn't hijack it."""
        with patch(
            "agent.skill_utils.get_subprocess_home",
            return_value="/opt/data/home",
        ):
            # Expand against a username that we know exists on the test host —
            # ``root`` is universal on POSIX runners.  We only assert the
            # result is *not* prefixed with our subprocess HOME, because the
            # actual expansion depends on /etc/passwd on the runner.
            result = _expanduser_for_subprocess("~root/conf")
            assert not result.startswith("/opt/data/home")

    def test_no_subprocess_home_falls_back_to_os_expanduser(self, monkeypatch):
        """When ``get_subprocess_home()`` returns ``None``, behaviour must be
        identical to ``os.path.expanduser`` — no change for users whose
        ``{HERMES_HOME}/home`` doesn't exist."""
        with patch("agent.skill_utils.get_subprocess_home", return_value=None):
            monkeypatch.setenv("HOME", "/home/alice")
            assert _expanduser_for_subprocess("~/wiki") == "/home/alice/wiki"
            assert _expanduser_for_subprocess("~") == "/home/alice"
            assert _expanduser_for_subprocess("/abs/path") == "/abs/path"


# ---------------------------------------------------------------------------
# resolve_skill_config_values — end-to-end through the skill config surface
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home_with_subprocess_home(tmp_path):
    """Create a HERMES_HOME with a ``home/`` subdirectory (activates
    ``get_subprocess_home``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "home").mkdir()  # <-- triggers get_subprocess_home activation
    (home / "config.yaml").write_text("skills:\n  config: {}\n")
    return home


@pytest.fixture
def hermes_home_without_subprocess_home(tmp_path):
    """Create a HERMES_HOME with NO ``home/`` subdirectory (subprocess HOME
    falls back to the Python process HOME)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  config: {}\n")
    return home


class TestResolveSkillConfigValuesHOME:
    def test_tilde_default_resolves_to_subprocess_home(
        self, hermes_home_with_subprocess_home
    ):
        """Reporter's scenario: ``default: ~/wiki`` must surface
        ``{HERMES_HOME}/home/wiki`` when the subprocess HOME is active."""
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home)},
        ):
            result = resolve_skill_config_values(
                [{"key": "wiki.path", "default": "~/wiki"}]
            )
        expected = str(hermes_home_with_subprocess_home / "home" / "wiki")
        assert result == {"wiki.path": expected}

    def test_tilde_default_falls_back_without_subprocess_home(
        self, hermes_home_without_subprocess_home, monkeypatch
    ):
        """Without ``{HERMES_HOME}/home`` on disk, behaviour matches the
        pre-fix ``os.path.expanduser`` output — nobody regresses."""
        monkeypatch.setenv("HOME", "/home/alice")
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_without_subprocess_home),
             "HOME": "/home/alice"},
        ):
            result = resolve_skill_config_values(
                [{"key": "wiki.path", "default": "~/wiki"}]
            )
        assert result == {"wiki.path": "/home/alice/wiki"}

    def test_explicit_config_value_also_expanded(
        self, hermes_home_with_subprocess_home
    ):
        """User-supplied ``skills.config.wiki.path: ~/notes`` in config.yaml
        must also flow through the subprocess-HOME expansion."""
        (hermes_home_with_subprocess_home / "config.yaml").write_text(
            "skills:\n  config:\n    wiki:\n      path: ~/notes\n"
        )
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home)},
        ):
            result = resolve_skill_config_values(
                [{"key": "wiki.path", "default": "~/wiki"}]
            )
        expected = str(hermes_home_with_subprocess_home / "home" / "notes")
        assert result == {"wiki.path": expected}

    def test_absolute_default_unchanged(self, hermes_home_with_subprocess_home):
        """Non-tilde defaults are passed through untouched — no false
        positives for absolute paths."""
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home)},
        ):
            result = resolve_skill_config_values(
                [{"key": "tool.dir", "default": "/opt/preconfigured/tool"}]
            )
        assert result == {"tool.dir": "/opt/preconfigured/tool"}

    def test_envvar_default_still_expanded(self, hermes_home_with_subprocess_home):
        """``${VAR}`` expansion is preserved alongside the new ``~``
        semantics.  Regression guard for the os.path.expandvars chain."""
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home),
             "MY_SKILL_DIR": "~/custom"},
        ):
            result = resolve_skill_config_values(
                [{"key": "tool.dir", "default": "${MY_SKILL_DIR}/bin"}]
            )
        expected = str(hermes_home_with_subprocess_home / "home" / "custom" / "bin")
        assert result == {"tool.dir": expected}

    def test_non_string_value_unaffected(self, hermes_home_with_subprocess_home):
        """Dict/list values stored under skills.config must not be mangled
        by the expansion step."""
        (hermes_home_with_subprocess_home / "config.yaml").write_text(
            "skills:\n  config:\n    tool:\n      flags: [a, b]\n"
        )
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home)},
        ):
            result = resolve_skill_config_values([{"key": "tool.flags"}])
        assert result == {"tool.flags": ["a", "b"]}

    def test_missing_default_uses_empty_string(
        self, hermes_home_with_subprocess_home
    ):
        """No default + no config value → empty string (unchanged)."""
        with patch.dict(
            os.environ,
            {"HERMES_HOME": str(hermes_home_with_subprocess_home)},
        ):
            result = resolve_skill_config_values([{"key": "unset.key"}])
        assert result == {"unset.key": ""}
