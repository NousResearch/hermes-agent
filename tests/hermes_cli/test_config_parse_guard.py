"""A read-modify-write of config.yaml must never clobber an unparsable file.

Class regression for #75431: every config reader (``read_raw_config``,
``load_config``) degrades a YAML syntax error to ``{}``/defaults, so a caller
that reads, mutates one key and saves writes a document containing only that
key — destroying every user setting. #75885/#75900 closed that for
``hermes config set/unset``; ``write_platform_config_field`` is the shared
read-modify-write helper behind the dashboard platform toggle
(``hermes_cli/web_server.py::_write_platform_enabled``), the gateway's
unauthorized-DM setter, and the photon CLI, and was still unguarded.

These drive the real functions against a real temp ``HERMES_HOME`` — no mocks
on the resolution chain, since the bug lives in how the readers degrade.
"""

from __future__ import annotations

import pytest

BROKEN_CONFIG = """\
model:
  default: claude-opus-4
terminal:
  backend: docker
  cwd: /srv/work
memory:
  provider: honcho
broken: [unclosed
"""

VALID_CONFIG = """\
model:
  default: claude-opus-4
terminal:
  backend: docker
  cwd: /srv/work
memory:
  provider: honcho
"""


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _config_path():
    from hermes_cli.config import get_config_path

    return get_config_path()


class TestRequireParsableConfigBeforeWrite:
    def test_absent_file_is_allowed(self, config_home):
        """First-run creation must not be blocked — there is nothing to lose."""
        from hermes_cli.config import require_parsable_config_before_write

        assert not _config_path().exists()
        require_parsable_config_before_write()  # must not raise

    def test_valid_file_is_allowed(self, config_home):
        from hermes_cli.config import require_parsable_config_before_write

        _config_path().write_text(VALID_CONFIG, encoding="utf-8")
        require_parsable_config_before_write()  # must not raise

    def test_syntax_error_refuses(self, config_home):
        from hermes_cli.config import require_parsable_config_before_write

        _config_path().write_text(BROKEN_CONFIG, encoding="utf-8")
        with pytest.raises(RuntimeError, match="YAML syntax error"):
            require_parsable_config_before_write()


class TestWritePlatformConfigFieldPreservesBrokenConfig:
    """The defect: one toggle replaced the whole file with just that toggle."""

    @pytest.mark.parametrize("raw", [False, True])
    def test_refuses_and_leaves_file_byte_identical(self, config_home, raw):
        from hermes_cli.config import write_platform_config_field

        path = _config_path()
        path.write_text(BROKEN_CONFIG, encoding="utf-8")
        before = path.read_bytes()

        with pytest.raises(RuntimeError, match="YAML syntax error"):
            write_platform_config_field("telegram", "enabled", True, raw=raw)

        assert path.read_bytes() == before

    def test_user_sections_survive(self, config_home):
        """Without the guard every one of these disappeared."""
        from hermes_cli.config import write_platform_config_field

        path = _config_path()
        path.write_text(BROKEN_CONFIG, encoding="utf-8")

        with pytest.raises(RuntimeError):
            write_platform_config_field("telegram", "enabled", True)

        after = path.read_text(encoding="utf-8")
        for section in ("terminal:", "backend: docker", "cwd: /srv/work",
                        "memory:", "provider: honcho"):
            assert section in after

    def test_dashboard_toggle_helper_is_guarded(self, config_home):
        """The real dashboard entry point, not just the shared helper."""
        from hermes_cli.web_server import _write_platform_enabled

        path = _config_path()
        path.write_text(BROKEN_CONFIG, encoding="utf-8")
        before = path.read_bytes()

        with pytest.raises(RuntimeError, match="YAML syntax error"):
            _write_platform_enabled("telegram", True)

        assert path.read_bytes() == before


class TestWritePlatformConfigFieldStillWorks:
    """The guard must not cost the happy path."""

    @pytest.mark.parametrize("raw", [False, True])
    def test_writes_field_and_keeps_other_sections(self, config_home, raw):
        from hermes_cli.config import read_user_config_raw, write_platform_config_field

        path = _config_path()
        path.write_text(VALID_CONFIG, encoding="utf-8")

        write_platform_config_field("telegram", "enabled", True, raw=raw)

        saved = read_user_config_raw(path)
        assert saved["platforms"]["telegram"]["enabled"] is True
        assert saved["terminal"]["backend"] == "docker"
        assert saved["terminal"]["cwd"] == "/srv/work"
        assert saved["memory"]["provider"] == "honcho"

    def test_creates_config_when_absent(self, config_home):
        from hermes_cli.config import read_user_config_raw, write_platform_config_field

        path = _config_path()
        assert not path.exists()

        write_platform_config_field("discord", "enabled", False)

        assert read_user_config_raw(path)["platforms"]["discord"]["enabled"] is False
