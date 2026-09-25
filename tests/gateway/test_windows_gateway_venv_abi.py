"""Windows gateway venv selection: prefer an ABI-matching venv.

Regression test for https://github.com/NousResearch/hermes-agent/issues/122325:
on PM-managed installs the gateway runs under a newer Python while
``_ensure_windows_gateway_venv_imports`` forced the legacy in-tree venv first,
so binary extensions (pydantic_core) failed to import.
"""

import sys

import gateway.run as run


def _write_cfg(path, version, key="version_info"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyvenv.cfg").write_text(
        "home = /nonexistent\nimplementation = CPython\n%s = %s\n" % (key, version),
        encoding="utf-8",
    )
    (path / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)


def test_abi_matches_current_interpreter(tmp_path):
    current = "%d.%d.0" % sys.version_info[:2]
    match = tmp_path / "match"
    _write_cfg(match, current)
    assert run._venv_abi_matches(match) is True


def test_abi_rejects_other_interpreter(tmp_path):
    other = tmp_path / "other"
    _write_cfg(other, "2.7.18" if sys.version_info[:2] != (2, 7) else "3.99.0")
    assert run._venv_abi_matches(other) is False


def test_abi_stdlib_version_key(tmp_path):
    """CPython's `python -m venv` writes `version`, not `version_info`."""
    current = "%d.%d.%d" % sys.version_info[:3]
    match = tmp_path / "stdlib-match"
    _write_cfg(match, current, key="version")
    assert run._venv_abi_matches(match) is True
    mismatch = tmp_path / "stdlib-mismatch"
    _write_cfg(mismatch, "2.7.18" if sys.version_info[:2] != (2, 7) else "3.99.0",
               key="version")
    assert run._venv_abi_matches(mismatch) is False


def test_abi_missing_or_broken_cfg_keeps_legacy_behavior(tmp_path):
    assert run._venv_abi_matches(tmp_path / "absent") is True
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
    assert run._venv_abi_matches(broken) is True
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "pyvenv.cfg").write_text("version_info = nonsense\n", encoding="utf-8")
    assert run._venv_abi_matches(garbage) is True


def test_mismatched_committed_venv_falls_behind_abi_match(tmp_path, monkeypatch):
    """The committed pick sorts behind an ABI-matching candidate.

    A committed venv built for another Python must not shadow the working
    one: without the sort it would land first on sys.path.
    """
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    _write_cfg(bad, "2.7.18" if sys.version_info[:2] != (2, 7) else "3.99.0")
    _write_cfg(good, "%d.%d.0" % sys.version_info[:2])
    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setenv("VIRTUAL_ENV", str(good))
    old_path = list(sys.path)
    old_pythonpath = run.os.environ.get("PYTHONPATH")
    try:
        run.os.environ.pop("PYTHONPATH", None)
        import pm.environments as env

        monkeypatch.setattr(env, "committed_venv", lambda root: bad)
        run._ensure_windows_gateway_venv_imports()
        entries = [entry.lower() for entry in sys.path]
        # The mismatched committed pick must not shadow anything: the
        # ABI-matching candidate wins.
        assert not any(str(bad).lower() in entry for entry in entries)
        assert any(str(good / "Lib").lower() in entry for entry in entries)
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            run.os.environ.pop("PYTHONPATH", None)
        else:
            run.os.environ["PYTHONPATH"] = old_pythonpath
