"""Tests for the env-write denylist on the memory-setup ``.env`` writer.

``hermes_cli.memory_setup._write_env_vars`` persists provider plugin
credentials to ``~/.hermes/.env``. It previously called ``Path.write_text``
directly, bypassing the ``_ENV_VAR_NAME_DENYLIST`` / ``_ENV_VAR_NAME_RE`` /
CR-LF-stripping gates that ``save_env_value`` enforces for every other
``.env`` writer in the codebase, and left the file at the default umask
between the write and a later ``chmod`` (a TOCTOU permission window).

A memory provider plugin schema declaring ``env_var: "LD_PRELOAD"`` (or any
other subprocess-influencing or Hermes-runtime-location name) could
otherwise plant a value into ``.env`` via the interactive memory-setup
wizard. The next Hermes process would load it through the
``env_loader.py`` ``.env -> os.environ`` chain and execute attacker code
before ``main()``.

The fix routes through ``save_env_value`` so the same gates fire.
"""

import pytest

from hermes_cli.config import ensure_hermes_home, get_env_path, load_env
from hermes_cli.memory_setup import _write_env_vars


def _env_file_keys() -> set[str]:
    """Parse ``~/.hermes/.env`` directly and return the set of keys present.

    Used by tests that want to verify a key was NOT written to disk without
    going through ``load_env()`` (whose sanitization/caching could mask the
    underlying file state).
    """
    env_path = get_env_path()
    if not env_path.exists():
        return set()
    keys: set[str] = set()
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, _ = line.partition("=")
            keys.add(key.strip())
    return keys


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ensure_hermes_home()
    return tmp_path


@pytest.mark.parametrize(
    "denied_key",
    [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "PATH",
        "EDITOR",
        "GIT_SSH_COMMAND",
        "HERMES_HOME",
        "HERMES_PROFILE",
        "HERMES_CONFIG",
        "HERMES_ENV",
    ],
)
def test_denylisted_key_is_skipped(denied_key, capsys):
    """Each denylisted name must not land in .env even though the
    memory-setup wizard accepted it from a (hypothetically malicious)
    provider schema. The wizard prints a warning and continues."""
    _write_env_vars({denied_key: "/tmp/evil.so"})

    # Assert directly against ``.env`` file contents so the test isn't
    # coupled to ``load_env()``'s sanitization or any future merge with
    # ``os.environ`` (where common names like ``PATH``/``EDITOR`` would
    # always appear and trivially satisfy a ``not in env`` check).
    assert denied_key not in _env_file_keys()

    captured = capsys.readouterr()
    assert denied_key in captured.out
    assert "denylist" in captured.out.lower() or "Skipping" in captured.out


def test_denylisted_key_does_not_block_other_writes(capsys):
    """If a single batch contains one denylisted key plus legitimate
    integration credentials, the denylisted one is skipped but the
    legitimate ones still land. The wizard must not abort mid-batch."""
    _write_env_vars({
        "LD_PRELOAD": "/tmp/evil.so",
        "HERMES_LANGFUSE_PUBLIC_KEY": "pk-test-123",
        "OPENROUTER_API_KEY": "sk-or-test-456",
    })

    assert "LD_PRELOAD" not in _env_file_keys()
    env = load_env()
    assert env["HERMES_LANGFUSE_PUBLIC_KEY"] == "pk-test-123"
    assert env["OPENROUTER_API_KEY"] == "sk-or-test-456"


def test_legitimate_hermes_integration_key_still_writable():
    """``HERMES_*`` overall is NOT blocked — only the four runtime
    location names (HOME/PROFILE/CONFIG/ENV). Integration credentials
    following the ``HERMES_*`` convention (HERMES_LANGFUSE_*,
    HERMES_SPOTIFY_*, HERMES_QWEN_BASE_URL, ...) must keep working or
    the memory-setup wizard regresses for every plugin that follows
    the convention."""
    _write_env_vars({
        "HERMES_LANGFUSE_PUBLIC_KEY": "pk-lf-789",
        "HERMES_QWEN_BASE_URL": "https://example.com/v1",
    })

    env = load_env()
    assert env["HERMES_LANGFUSE_PUBLIC_KEY"] == "pk-lf-789"
    assert env["HERMES_QWEN_BASE_URL"] == "https://example.com/v1"


def test_malformed_key_name_is_skipped(capsys):
    """The canonical writer also enforces ``_ENV_VAR_NAME_RE`` —
    identifiers must match ``[A-Za-z_][A-Za-z0-9_]*``. A plugin schema
    declaring ``env_var: "FOO BAR"`` (space) was previously persisted
    verbatim, producing a malformed ``.env`` line."""
    _write_env_vars({"FOO BAR": "value"})

    keys = _env_file_keys()
    assert "FOO BAR" not in keys
    assert "FOO" not in keys  # not silently truncated either

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "FOO BAR" in captured.out


def test_legitimate_value_writes_round_trip():
    """Negative control — the gate must not regress on a normal write."""
    _write_env_vars({"MEM0_API_KEY": "m0-test-key-abc"})

    env = load_env()
    assert env["MEM0_API_KEY"] == "m0-test-key-abc"


def test_explicit_hermes_home_writes_to_that_env_file(tmp_path):
    """Plugin ``post_setup`` hooks (e.g. Supermemory) pass an explicit
    Hermes home; keep that target while still routing through
    ``save_env_value`` validation instead of writing directly."""
    home = tmp_path / "plugin-home"

    _write_env_vars({"MEM0_API_KEY": "m0-test-key-abc"}, hermes_home=home)

    assert "MEM0_API_KEY=m0-test-key-abc\n" in (home / ".env").read_text(
        encoding="utf-8"
    )


def test_value_with_embedded_newline_is_stripped():
    """``save_env_value`` strips CR/LF from the value to prevent
    .env-file structure injection (a value containing ``\\n`` would
    otherwise split the line and inject an arbitrary follow-on key).
    Routing through it gives the memory-setup wizard the same
    protection."""
    _write_env_vars({"MEM0_API_KEY": "key1\nEVIL=injected\n"})

    env = load_env()
    # CR/LF stripped, value still lands intact (minus the newlines)
    assert env["MEM0_API_KEY"] == "key1EVIL=injected"
    # And no smuggled key landed — assert against the file too so the test
    # holds even if ``load_env()`` ever starts merging ``os.environ``.
    assert "EVIL" not in _env_file_keys()


def test_env_file_created_with_secure_permissions(tmp_path):
    """Regression guard for the TOCTOU window the direct ``Path.write_text``
    + post-hoc ``chmod`` implementation had: ``save_env_value`` creates the
    temp file with 0o600 before any content is written and atomically
    replaces the target, so the file is never briefly world/group-readable
    at the process umask."""
    import stat

    _write_env_vars({"MEM0_API_KEY": "m0-test-key-abc"})

    env_path = get_env_path()
    mode = stat.S_IMODE(env_path.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR
