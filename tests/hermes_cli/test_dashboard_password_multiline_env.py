"""``hermes dashboard password`` removing the overriding ``HERMES_DASHBOARD_BASIC_AUTH_*`` lines from a ``.env``
that holds a quoted value over several lines (a PEM key). The loader (python-dotenv) reads such a value whole,
so only whole assignments, as it reads them, are removed: a look-alike line inside another value stays, a
multi-line password goes with all its lines, and every other byte and the file's mode stay as they were. A
file the rewrite would re-encode (not UTF-8) is refused before anything is written when it has an assignment to
remove; with none, it is left alone, as on main. Follow-up to #157393.
"""
import os
import stat

import pytest
from dotenv import dotenv_values

from tests.hermes_cli.live_gateway import home_env, run_cli, write_config

pytestmark = pytest.mark.platforms("linux")

USER, PASSWORD = "spike-admin", "throwaway-Pw-51c2"  # invented for this test

_PEM = ('TLS_KEY="-----BEGIN SCRATCH PEM BLOCK-----\n'
        'c2NyYXRjaCBibG9jayBib2R5\n'
        'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=look-alike-line-inside-the-pem\n'
        '-----END SCRATCH PEM BLOCK-----"\n')


def _home(tmp_path):
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    write_config(home)
    return home, home_env(home, user)


def _set_password(env):
    return run_cli(env, "dashboard", "password", "--username", USER, "--password-stdin", stdin=PASSWORD + "\n")


def _write_env(home, data: bytes, mode=0o640):
    path = home / ".env"
    path.write_bytes(data)
    path.chmod(mode)
    return path


def test_whole_assignments_go_and_a_look_alike_line_inside_a_pem_stays(tmp_path):
    home, env = _home(tmp_path)
    dotenv = _write_env(home, (
        "PLAIN_SETTING=first\n" + _PEM
        + 'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD="old-pass-first-line\nold-pass-second-line"\n'
        + "OTHER_SETTING=kept\n").encode())

    result = _set_password(env)

    assert result.returncode == 0, result.stdout
    assert "Removed HERMES_DASHBOARD_BASIC_AUTH_PASSWORD from .env" in result.stdout
    assert "old-pass" not in result.stdout
    assert dotenv.read_bytes() == ("PLAIN_SETTING=first\n" + _PEM + "OTHER_SETTING=kept\n").encode()
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640
    loaded = dotenv_values(dotenv)
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" not in loaded
    assert loaded["TLS_KEY"].splitlines()[2] == "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=look-alike-line-inside-the-pem"


def test_every_other_byte_stays_with_a_bom_crlf_comments_and_blank_lines(tmp_path):
    home, env = _home(tmp_path)
    keep_head = "\ufeff# local secrets\r\nPLAIN_SETTING=first\r\n"
    keep_tail = "\r\n# trailing comment\r\nOTHER_SETTING='a\r\nb'\r\n"
    dotenv = _write_env(home, (keep_head + "\r\nexport HERMES_DASHBOARD_BASIC_AUTH_USERNAME=old-user\r\n"
                               + 'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH="x\r\ny" # old\r\n' + keep_tail)
                        .encode(), mode=0o600)

    result = _set_password(env)

    assert result.returncode == 0, result.stdout
    assert dotenv.read_bytes() == (keep_head + "\r\n" + keep_tail).encode()
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600


def test_a_multiline_env_that_is_not_utf8_is_refused_before_anything_is_written(tmp_path):
    home, env = _home(tmp_path)
    dotenv = _write_env(home, b'NOTE="caf\xe9\nsecond line"\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=old-pass\n')
    before = {name: (home / name).read_bytes() for name in ("config.yaml", ".env")}

    result = _set_password(env)

    assert result.returncode == 1, result.stdout
    assert "not UTF-8 text" in result.stdout and "nothing was changed" in result.stdout
    assert {name: (home / name).read_bytes() for name in ("config.yaml", ".env")} == before
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640


@pytest.mark.parametrize("data", [
    b'NOTE="caf\xe9 line one\nline two"\nPLAIN=1\n',  # nothing to remove: exactly as on main
    b'NOTE="caf\xe9\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=look-alike"\nPLAIN=1\n',  # only inside another value
])
def test_a_multiline_env_that_is_not_utf8_with_nothing_to_remove_is_left_alone(tmp_path, data):
    home, env = _home(tmp_path)
    dotenv = _write_env(home, data)

    result = _set_password(env)

    assert result.returncode == 0, result.stdout
    assert "not UTF-8" not in result.stdout and "Removed" not in result.stdout
    assert dotenv.read_bytes() == data
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640


@pytest.mark.parametrize("multiline", [True, False])
def test_a_quoted_password_name_the_loader_reads_is_removed(tmp_path, multiline):
    """'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD'=old is the password to python-dotenv (the gateway's loader), so the
    command removes it instead of reporting success while it still overrides the new hash."""
    home, env = _home(tmp_path)
    # A value over several lines with no look-alike line in it (which the line reader would also see).
    kept = ("PLAIN_SETTING=first\n" + ('NOTE="line one\nline two"\n' if multiline else "") + "OTHER_SETTING=kept\n").encode()
    dotenv = _write_env(home, kept + b"'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD'=old-pass\n")

    result = _set_password(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Removed HERMES_DASHBOARD_BASIC_AUTH_PASSWORD from .env" in result.stdout
    assert dotenv.read_bytes() == kept
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" not in dotenv_values(dotenv)


def test_a_plain_env_is_cleaned_line_by_line_as_before(tmp_path):
    """Control: a .env with no value over several lines takes the unchanged line-by-line path."""
    home, env = _home(tmp_path)
    dotenv = _write_env(home, b"PLAIN_SETTING=first\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=old-pass\nOTHER_SETTING=kept\n")

    result = _set_password(env)

    assert result.returncode == 0, result.stdout
    assert dotenv.read_bytes() == b"PLAIN_SETTING=first\nOTHER_SETTING=kept\n"
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640


def _windows_newlines(monkeypatch):
    """Simulate Windows text mode here: `newline=None` writes "\\n" as "\\r\\n" there."""
    from hermes_cli import config
    real = os.fdopen

    def fdopen(fd, *args, **kwargs):
        if kwargs.get("newline") is None:
            kwargs["newline"] = "\r\n"
        return real(fd, *args, **kwargs)

    monkeypatch.setattr(config.os, "fdopen", fdopen)


def test_the_multiline_cleanup_writes_crlf_once_with_windows_newline_translation(tmp_path, monkeypatch):
    from hermes_cli.config import remove_env_value
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", raising=False)
    _windows_newlines(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_bytes(b'KEEP="a\r\nb"\r\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=old\r\nLF_LINE=c\n')

    assert remove_env_value("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD") is True
    assert dotenv.read_bytes() == b'KEEP="a\r\nb"\r\nLF_LINE=c\n'


def test_the_plain_line_by_line_cleanup_is_unchanged_with_windows_newline_translation(tmp_path, monkeypatch):
    """Control: main's writer re-reads lines with universal newlines, so it writes CRLF once (it does turn an
    LF file into CRLF on Windows, as main always did)."""
    from hermes_cli.config import remove_env_value
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", raising=False)
    _windows_newlines(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_bytes(b"A=1\r\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=old\r\nB=2\n")

    assert remove_env_value("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD") is True
    assert dotenv.read_bytes() == b"A=1\r\nB=2\r\n"
