"""KeePass / KeePassXC secret source + vault backend (#108338).

The library's own suite covers 1Password and Bitwarden by running a fake CLI as a shebang
script, which is skipped on Windows. These pin the same class of contract with the CLI faked at
the ``subprocess.run`` boundary instead, so they exercise the real code path on every host:

1. The database key reaches keepassxc-cli on the CHILD's stdin — never argv, never the env.
2. A key-file-only call gets /dev/null on stdin, so the CLI can never block waiting for a prompt.
3. The export CSV's Password column never reaches the returned entries.
4. A missing, or pinned-but-absent, binary is a clear error rather than a crash or a PATH fallback.
5. The backend turns entries into CLI-visible paths and drops logins with no URL to bind a fill to.
"""

from __future__ import annotations

import dataclasses
import subprocess

import pytest

from agent.secret_sources import keepass as keepass_mod
from agent.vault_backends.keepass import KeePassLoginBackend

MASTER = "correct horse battery staple"

# KeePassXC's own export shape (the Password column is the one that must never be kept).
_EXPORT_CSV = """\
Group,Title,Username,Password,URL,Notes
Root/Work,Example,jane@example.com,{master},https://example.com/login,note
Root/Work/Prod,GitHub,octocat,{master},https://github.com,,,
Root/Personal,No URL entry,someone,{master},,
Root/Work,No password,yet-another,{master},https://example.org,
""".format(master=MASTER)


class _Recorder:
    """Stands in for ``subprocess.run``: records the call, answers with canned output."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.calls: list = []
        self._stdout = stdout
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, self._returncode, stdout=self._stdout, stderr=self._stderr)

    @property
    def argv(self):
        return self.calls[-1][0]

    @property
    def kwargs(self):
        return self.calls[-1][1]


@pytest.fixture()
def fake_run(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(keepass_mod.subprocess, "run", rec)
    monkeypatch.setattr(keepass_mod, "find_keepassxc", lambda binary_path="": keepass_mod.Path("/usr/bin/keepassxc-cli"))
    return rec


def _cfg_with_db(tmp_path, **extra):
    db = tmp_path / "vault.kdbx"
    db.write_bytes(b"kdbx")
    return {"db": str(db), **extra}


# ── the key's travel path ────────────────────────────────────────────────────

def test_key_goes_on_stdin_and_never_in_argv_or_env(fake_run):
    keepass_mod.verify_database(db="C:/db.kdbx", key=MASTER)

    assert MASTER not in " ".join(fake_run.argv), "the database key must never be an argument"
    assert fake_run.kwargs["input"] == MASTER + "\n", "keepassxc-cli reads the key from stdin"
    env_values = " ".join(str(v) for v in (fake_run.kwargs.get("env") or {}).values())
    assert MASTER not in env_values, "the key must not travel through the child environment"


def test_keyfile_only_call_never_waits_for_a_prompt(fake_run):
    keepass_mod.verify_database(db="C:/db.kdbx", key=None, keyfile="C:/keys/vault.key")

    assert fake_run.kwargs.get("stdin") is subprocess.DEVNULL
    assert "input" not in fake_run.kwargs, "no key configured means no stdin payload at all"
    from pathlib import Path

    assert "-k" in fake_run.argv
    assert str(Path("C:/keys/vault.key")) in fake_run.argv, "key files are passed expanded"


def test_positional_paths_are_separated_from_options(fake_run):
    """``--`` keeps a leading-dash entry path from parsing as an option."""
    fake_run._stdout = "hunter2-ish\n"
    keepass_mod.show_attribute(db="C:/db.kdbx", entry="-user/.config", key=MASTER)

    argv = fake_run.argv
    assert argv[argv.index("--") + 1:] == ["C:/db.kdbx", "-user/.config"]
    assert argv[:2] == [argv[0], "show"] and "-a" in argv


def test_show_attribute_refuses_an_empty_value(fake_run):
    """An exit-0 empty attribute must not be applied: that would clobber a good credential."""
    fake_run._stdout = "\n"
    with pytest.raises(RuntimeError, match="empty Password"):
        keepass_mod.show_attribute(db="C:/db.kdbx", entry="Root/Example", key=MASTER)


# ── missing binary ───────────────────────────────────────────────────────────

def test_missing_binary_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(keepass_mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="keepassxc-cli was not found on PATH"):
        keepass_mod.export_entries(db="C:/db.kdbx", key=MASTER)


def test_pinned_binary_does_not_fall_back_to_path(monkeypatch):
    monkeypatch.setattr(keepass_mod.shutil, "which", lambda name: "/usr/bin/keepassxc-cli")

    with pytest.raises(RuntimeError, match="is not an executable keepassxc-cli"):
        keepass_mod.export_entries(db="C:/db.kdbx", key=MASTER, binary_path="C:/nope/keepassxc-cli")


# ── the export CSV ───────────────────────────────────────────────────────────

def test_export_keeps_metadata_and_drops_the_password_column(fake_run):
    fake_run._stdout = _EXPORT_CSV
    entries = keepass_mod.export_entries(db="C:/db.kdbx", key=MASTER)

    assert [e.title for e in entries] == ["Example", "GitHub", "No URL entry", "No password"]
    example = entries[0]
    assert example.username == "jane@example.com"
    assert example.url == "https://example.com/login"
    assert example.path == "/Work/Example", "the export's root group name is not part of a CLI path"
    assert not hasattr(example, "password"), "the entry shape carries no password field"

    dumped = " ".join(str(v) for e in entries for v in dataclasses.asdict(e).values())
    assert MASTER not in dumped, "the export's Password column must not survive parsing"


def test_nested_group_names_become_cli_paths(fake_run):
    fake_run._stdout = _EXPORT_CSV
    entries = keepass_mod.export_entries(db="C:/db.kdbx", key=MASTER)

    by_title = {e.title: e.path for e in entries}
    assert by_title["GitHub"] == "/Work/Prod/GitHub"
    assert by_title["No URL entry"] == "/Personal/No URL entry"


def test_unrecognized_csv_header_is_reported_not_swallowed(fake_run):
    fake_run._stdout = "Name,Value\nsomething,else\n"
    with pytest.raises(RuntimeError, match="unrecognized CSV header"):
        keepass_mod.export_entries(db="C:/db.kdbx", key=MASTER)


# ── the backend contract ─────────────────────────────────────────────────────

def test_backend_lists_only_logins_a_fill_can_be_bound_to(tmp_path, monkeypatch):
    backend = KeePassLoginBackend(_cfg_with_db(tmp_path))
    monkeypatch.setattr(type(backend), "is_unlocked", lambda self: True)
    monkeypatch.setattr(type(backend), "_key", lambda self: "db-key")
    monkeypatch.setattr(
        "agent.vault_backends.keepass.export_entries",
        lambda **_: [
            keepass_mod.KeepassEntry(path="/Work/Example", title="Example",
                                     username="jane@example.com", url="https://example.com/login"),
            keepass_mod.KeepassEntry(path="/Personal/No URL", title="No URL", username="someone", url=""),
        ],
    )

    items = backend.list_items()

    assert [i.label for i in items] == ["Example"], "a login with no URL has nothing to fill"
    assert items[0].id == "keepass:/Work/Example"
    assert items[0].kind == "login"
    assert items[0].identifier == "jane@example.com"


def test_backend_reports_a_missing_database_on_the_unlock_path(tmp_path):
    """An unconfigured or absent database is not an unlocked one: the unlock call names the config key."""
    with pytest.raises(RuntimeError, match="vault.keepass.db"):
        KeePassLoginBackend({}).unlock(MASTER)

    missing = KeePassLoginBackend({"db": str(tmp_path / "absent.kdbx")})
    with pytest.raises(RuntimeError, match="was not found"):
        missing.unlock(MASTER)


def test_backend_without_a_configured_database_lists_nothing(tmp_path):
    """Nothing to read and nobody asked for a password: an empty list, not an error."""
    assert KeePassLoginBackend({}).list_items() == []
