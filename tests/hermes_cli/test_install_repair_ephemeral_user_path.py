"""The hermes update tail must not persist test/ephemeral home launchers into the user PATH.

Sibling of tests/scripts/install/test_install_ps1_ephemeral_user_path.py: install.ps1's
``Test-EphemeralLauncherHome`` guards fresh installs (#125614), but ``migrate_windows_bin_path``
runs unconditionally from the update self-heal list and writes the SAME persistent user PATH.
``get_default_hermes_root()`` honors ``HERMES_HOME``, so a harness home under %TEMP% equals
``home``, passes the managed-clone gate, and would prepend ``home\\bin`` to the persistent PATH —
the stale-shim shadow the installer guard exists to prevent, reached through a second call site.
Both writers must share one classifier: ``is_ephemeral_launcher_home``.
"""
from hermes_cli._install_repair import migrate_windows_bin_path


def _fake_user_path(entries):
    calls = {"read": 0, "written": None}

    def read_user_path():
        calls["read"] += 1
        return list(entries), 2  # REG_EXPAND_SZ

    def write_user_path(new_entries, kind):
        calls["written"] = (list(new_entries), kind)

    return read_user_path, write_user_path, calls


def test_ephemeral_home_update_tail_never_touches_user_path(tmp_path, monkeypatch):
    """A TEMP-rooted HERMES_HOME (the smoke-harness shape) returns False before any
    PATH read or write — the registry value stays byte-identical. Launchers are
    pre-staged so the old code's staging-verify passes and the write is genuinely
    reachable: only the ephemeral-home guard stops it."""
    home = tmp_path / "hermes_test_home_125614"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / "managed-clone"
    home_bin = home / "bin"
    home_bin.mkdir(parents=True)
    for name in ("hermes.exe", "hermes.cmd", "hermes-acp.exe", "hermes-acp.cmd"):
        (home_bin / name).write_bytes(b"stub")
    read_user_path, write_user_path, calls = _fake_user_path([r"C:\Windows\System32"])

    result = migrate_windows_bin_path(
        root, windows=True, read_user_path=read_user_path, write_user_path=write_user_path)

    assert result is False
    assert calls["written"] is None, "persistent user PATH must not be written for an ephemeral home"
    assert calls["read"] == 0, "the guard must fire before the registry is even read"


def test_production_home_update_tail_still_migrates(tmp_path, monkeypatch):
    """A production home keeps today's behavior: prepend home\\bin once, strip legacy entries.
    (Classifier patched False to simulate a non-temp install; launchers pre-staged so the
    verify step passes.)"""
    home = tmp_path / "prod-like-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / "managed-clone"
    home_bin = home / "bin"
    home_bin.mkdir(parents=True)
    for name in ("hermes.exe", "hermes.cmd", "hermes-acp.exe", "hermes-acp.cmd"):
        (home_bin / name).write_bytes(b"stub")
    legacy_entry = str(root / "bin")
    read_user_path, write_user_path, calls = _fake_user_path([legacy_entry])

    monkeypatch.setattr(
        "hermes_cli._install_repair.is_ephemeral_launcher_home", lambda path: False)
    result = migrate_windows_bin_path(
        root, windows=True, read_user_path=read_user_path, write_user_path=write_user_path)

    assert result is True
    written, kind = calls["written"]
    assert written[0] == str(home_bin)
    assert legacy_entry not in written
    assert kind == 2


def test_classifier_matches_the_installer_twin(tmp_path, monkeypatch):
    """Same verdicts install.ps1's Test-EphemeralLauncherHome gives (its C0/C2/C3 shape):
    under the temp dir -> ephemeral; hermes_test_home marker -> ephemeral; a production-like
    dir outside every temp root -> production."""
    from hermes_cli._install_repair import is_ephemeral_launcher_home
    assert is_ephemeral_launcher_home(tmp_path / "hermes_test_home_abc" / "bin") is True
    assert is_ephemeral_launcher_home(tmp_path / "pytest-of-x" / "home" / "bin") is True
    production = tmp_path / "prod-like" / "bin"
    # Simulate a non-temp production root by pointing the temp roots at a fresh dir.
    monkeypatch.setenv("TMP", str(tmp_path / "othertmp"))
    monkeypatch.setenv("TEMP", str(tmp_path / "othertmp"))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "othertmp"))
    (tmp_path / "othertmp").mkdir()
    assert is_ephemeral_launcher_home(production) is False
    assert is_ephemeral_launcher_home(tmp_path / "othertmp" / "inside" / "bin") is True
    # The hermes_test_home marker catches marker-named homes even outside the temp roots.
    assert is_ephemeral_launcher_home(tmp_path / "plain" / "hermes_test_home_marker" / "bin") is True
