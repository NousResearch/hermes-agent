"""A Windows test run must never write the operator's real ``HKCU\\Environment\\Path``.

Test isolation redirects ``HERMES_HOME`` to a per-session sandbox, so anything
that registers ``$HERMES_HOME\\bin`` on the User PATH while pytest is running is
prepending a throwaway ``%TEMP%`` directory to the machine's *persisted* PATH.
Nothing ever removes it, so the entry count grows with every test that reaches
the writer until the 32,767-character environment block is full and the
operator's own PATH stops loading (measured 2026-10-10 08:35-08:44: 17 -> 235
entries, 723 -> 29,601 chars, ~310 chars/minute).

Two writers persist that entry, so there are two red lights:

- ``test_register_windows_user_path_is_inert_under_test_isolation`` covers
  ``hermes_cli/_launchers._register_windows_user_path`` (the ``hermes update``
  tail's ``expose_cli`` route);
- ``test_migrate_windows_bin_path_is_inert_under_test_isolation`` covers
  ``hermes_cli/_install_repair._write_user_path_raw`` (the same tail's
  ``migrate_windows_bin_path`` route).

Both drive the real writer and assert the stored value *and its registry type*
are byte-identical. Run either against a tree without its guard and it fails --
because the entry really landed in the operator's PATH.
"""
from __future__ import annotations

import pytest

# ``platforms("windows")`` rather than ``skipif(sys.platform != "win32")``: the
# OS lanes import the files ``scripts/ci/list_os_marked_tests.py`` lists for
# their marker, and the Linux full-suite lane skips anything gated off it — so a
# bare skipif leaves these tests running on NO host. It also kept ``winreg`` in
# the module header, which made the file a collection ERROR on Linux.
pytestmark = pytest.mark.platforms("windows")


def _raw_user_path() -> tuple[str, int] | None:
    """The stored value AND its type -- never the ``%VAR%``-expanded form.

    Expanding would hide a type change (REG_EXPAND_SZ -> REG_SZ freezes
    ``%LOCALAPPDATA%\\...`` entries), so the comparison is on raw bytes.
    """
    import winreg  # lazy: the module must stay importable off Windows

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ
    ) as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return None
        return str(value), int(kind)


def test_register_windows_user_path_is_inert_under_test_isolation(tmp_path):
    """The real writer must not touch the operator's PATH during a test run."""
    from hermes_cli import _launchers

    sandbox_bin = tmp_path / "hermes_test" / "bin"
    before = _raw_user_path()

    _launchers._register_windows_user_path(sandbox_bin)

    after = _raw_user_path()
    assert after == before, (
        f"{sandbox_bin} leaked into the operator's persisted User PATH "
        f"(HKCU\\Environment\\Path): {before[0][:80]!r}... -> {after[0][:80]!r}..."
        if after and before
        else "HKCU\\Environment\\Path changed during a test run"
    )


def test_migrate_windows_bin_path_is_inert_under_test_isolation(tmp_path, monkeypatch):
    """The *second* prepender writes the same value and needs the same guard.

    ``migrate_windows_bin_path`` prepends ``$HERMES_HOME\\bin`` through
    ``_write_user_path_raw``. Its only gate is
    ``root.parent == get_default_hermes_root()`` -- and test isolation points
    ``HERMES_HOME`` *at* the sandbox, so a root living inside the sandbox
    satisfies that gate and the throwaway ``<tmp>\\...\\bin`` is persisted into
    the operator's real PATH, one entry per test that reaches it.

    Drives the real function -- no injected ``read_user_path``/``write_user_path``
    -- so the default ``_read_user_path_raw``/``_write_user_path_raw`` pair runs
    against ``HKCU\\Environment``, then compares raw value and registry type.
    """
    from hermes_cli import _install_repair

    sandbox_home = tmp_path / "hermes_home"
    root = sandbox_home / "hermes-agent"
    home_bin = sandbox_home / "bin"
    home_bin.mkdir(parents=True)
    for name in ("hermes", "hermes-acp"):
        (home_bin / f"{name}.exe").write_bytes(b"stub launcher")

    # Keep ``ensure_windows_bin_launchers``' staging branch out of the picture:
    # it re-stages only while ``<root>\\.hermes\\bin`` is missing a launcher, and
    # minting one needs a store python this sandbox has not got.
    local_bin = root / ".hermes" / "bin"
    local_bin.mkdir(parents=True)
    for name in ("hermes", "hermes-acp"):
        (local_bin / f"{name}.exe").write_bytes(b"stub launcher")

    monkeypatch.setenv("HERMES_HOME", str(sandbox_home))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(sandbox_home))

    before = _raw_user_path()

    migrated = _install_repair.migrate_windows_bin_path(root, windows=True)

    after = _raw_user_path()
    # The migration must have got past its gates and reported the canonical
    # layout in place -- a False here means this case bailed early and the
    # comparison below would prove nothing.
    assert migrated is True
    assert after == before, (
        f"{home_bin} leaked into the operator's persisted User PATH "
        f"(HKCU\\Environment\\Path): {before[0][:80]!r}... -> {after[0][:80]!r}..."
        if after and before
        else "HKCU\\Environment\\Path changed during a test run"
    )
