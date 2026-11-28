"""``remove_env_value`` on a ``.env`` that holds a quoted value over several lines (a PEM key): python-dotenv,
the loader's parser, reads such a value whole, so the key's whole assignments, as it reads them, are removed,
and every other byte stays (line endings and the file's mode included). A look-alike line inside another value
is part of that value. A file the rewrite would re-encode (not UTF-8) is refused when it has an assignment to
remove. Any other file takes the line-by-line path, unchanged. ``hermes dashboard password`` removes its
overrides through this one API. Follow-up to #157393.
"""
import os
import stat

import pytest

from hermes_cli import config

_PEM = ('TLS_KEY="-----BEGIN SCRATCH PEM BLOCK-----\n'
        'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=look-alike-line-inside-the-pem\n'
        '-----END SCRATCH PEM BLOCK-----"\n')
_KEY = "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(_KEY, raising=False)
    return tmp_path / ".env"


def _write(path, data: bytes, mode=0o640):
    path.write_bytes(data)
    path.chmod(mode)


def test_a_multiline_assignment_goes_whole_and_a_look_alike_line_stays(dotenv, monkeypatch):
    _write(dotenv, ("A=1\n" + _PEM + f'{_KEY}="first\nsecond"\n' + "B=2\n").encode())
    monkeypatch.setenv(_KEY, "published")

    assert config.remove_env_value(_KEY) is True

    assert dotenv.read_bytes() == ("A=1\n" + _PEM + "B=2\n").encode()
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640
    assert _KEY not in os.environ  # published as removed, as on the line-by-line path


def test_only_a_look_alike_line_inside_a_multiline_value_is_not_an_assignment(dotenv):
    data = ("A=1\n" + _PEM).encode()
    _write(dotenv, data)

    assert config.remove_env_value(_KEY) is False
    assert dotenv.read_bytes() == data


def test_a_multiline_env_that_is_not_utf8_is_refused_only_with_an_assignment_to_remove(dotenv):
    data = b'NOTE="caf\xe9\nline two"\n' + _KEY.encode() + b"=old\n"
    _write(dotenv, data)

    with pytest.raises(OSError, match="not UTF-8"):
        config.remove_env_value(_KEY)
    assert dotenv.read_bytes() == data

    _write(dotenv, b'NOTE="caf\xe9\nline two"\nPLAIN=1\n')
    assert config.remove_env_value(_KEY) is False
    assert dotenv.read_bytes() == b'NOTE="caf\xe9\nline two"\nPLAIN=1\n'


@pytest.mark.parametrize("data", [
    b"A=1\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=old\nB=2\n",
    b"  A = 1 \r\nexport HERMES_DASHBOARD_BASIC_AUTH_PASSWORD='old'\r\n# c\r\n\r\nB=2",
    b'Q="one line"\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD="old"\n',
])
def test_a_plain_env_takes_the_line_by_line_path(dotenv, monkeypatch, data):
    """Control: no value over several lines, so the line reader writes exactly what it wrote before."""
    _write(dotenv, data)
    calls = []
    real = config._write_env_lines
    monkeypatch.setattr(config, "_write_env_lines", lambda *a, **kw: (calls.append(kw), real(*a, **kw))[1])

    assert config.remove_env_value(_KEY) is True

    expected = [line for line in config._sanitize_env_lines(data.decode().splitlines(keepends=True))
                if not config._env_line_defines_key(line, _KEY)]
    assert dotenv.read_text(encoding="utf-8") == "".join(expected)
    assert calls == [{"preserve_mode": True}]


@pytest.mark.parametrize("data", [
    ("A=1\nNOTE=\"line one\nline two\"\n" + f"'{_KEY}'=old\n" + "B=2\n").encode(),  # beside a value over several lines
    f"A=1\n'{_KEY}'=old\r\nB=2\n".encode(),  # a plain file: the line reader does not see the quoted name
])
def test_a_quoted_name_the_loader_reads_is_removed_whole(dotenv, data):
    """python-dotenv reads 'KEY'=value as KEY, so removing KEY removes it; every other byte stays."""
    from dotenv import dotenv_values
    _write(dotenv, data)
    assert dotenv_values(dotenv)[_KEY] == "old"

    assert config.remove_env_value(_KEY) is True

    assert dotenv.read_bytes() == data.replace(f"'{_KEY}'=old\r\n".encode(), b"").replace(
        f"'{_KEY}'=old\n".encode(), b"")
    assert _KEY not in dotenv_values(dotenv)
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o640


def test_dashboard_password_removes_its_overrides_through_remove_env_value(dotenv, monkeypatch):
    from hermes_cli import dashboard_password
    _write(dotenv, ("A=1\n" + _PEM + f'{_KEY}="first\nsecond"\n').encode())
    seen = []
    real = config.remove_env_value
    monkeypatch.setattr(config, "remove_env_value", lambda key: (seen.append(key), real(key))[1])

    assert dashboard_password._clear_dotenv_shadows() == ([_KEY], [])

    assert seen == [_KEY]
    assert dotenv.read_bytes() == ("A=1\n" + _PEM).encode()
