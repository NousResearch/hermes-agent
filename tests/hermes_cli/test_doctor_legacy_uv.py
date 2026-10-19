"""A pre-PM ``$HERMES_HOME/bin/uv*`` shadows the user's own uv and must be reported and cleared.

PM keeps its uv in the store and deliberately off PATH, so the only ``uv`` a
Hermes install should ever contribute to a shell is none. On Windows the
launcher prepends ``$HERMES_HOME/bin`` to the User PATH and the agent terminal
appends it everywhere else (#101269), so a leftover ``uv`` there is the one
every invocation resolves — and it is a version nothing maintains.

These tests pin what ``hermes doctor`` reports and which issue bucket (fixable vs
manual-only) each outcome lands in. The deletion mechanics themselves are covered
by ``tests/hermes_cli/test_legacy_uv_cleanup.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import doctor, doctor_state, legacy_uv
from tests.hermes_cli.legacy_uv_support import _pre_pm_binary, _uv, _uv_binary, _uvx_binary

pytest_plugins = ["tests.hermes_cli.legacy_uv_support"]


@pytest.mark.platforms("posix")
def test_doctor_reports_a_pre_pm_uv_shadow(home, capsys):
    _uvx_binary(home / "bin" / "uvx")
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert any("hermes doctor --fix" in issue for issue in finding.issues)
    assert "shadow your own uv" in capsys.readouterr().out
    assert finding.fixed == 0
    # A report must not delete: the user gets to choose how to clear it.
    assert (home / "bin" / "uvx").is_file()


@pytest.mark.platforms("posix")
def test_doctor_names_the_update_prerequisite_before_offering_fix(home, monkeypatch):
    """The no-``--fix`` report states the ``hermes update`` prerequisite up front while PM's
    store has no uv, instead of sending the user into a ``--fix`` that can only refuse."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _uv_binary(home / "bin" / "uv")

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    assert finding.issues
    assert "hermes update" in finding.issues[0]
    assert "hermes doctor --fix" in finding.issues[0]
    assert (home / "bin" / "uv").is_file(), "a report must never delete"


@pytest.mark.platforms("posix")
def test_doctor_keeps_an_unprobeable_uv_unresolved(home, monkeypatch):
    """A file disappearing between discovery and sizing must not abort or look clean."""
    _uv_binary(home / "bin" / "uv")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv":
            raise PermissionError("raced with cleanup")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    # ``_fail_and_issue`` records the FIX instruction, not the headline text.
    assert finding.issues
    assert "inaccessible" in finding.issues[0]


@pytest.mark.platforms("windows")
def test_doctor_keeps_an_unprobeable_uv_manual_on_windows(home, monkeypatch):
    """The same race on a fail-closed platform: automatic_cleanup() is POSIX-only,
    so a retry-``--fix`` hint would be advice no run can satisfy — the finding
    lands in the manual bucket that survives --fix instead."""
    _uv_binary(home / "bin" / "uv")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv":
            raise PermissionError("raced with cleanup")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "not performed on this platform" in finding.manual_issues[0]
    assert "manually" in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable on Windows"


@pytest.mark.platforms("posix")
def test_doctor_fix_removes_only_the_uv_family(home, monkeypatch):
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        _pre_pm_binary(home / "bin" / name)
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    (home / "bin" / "mytool").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 4
    assert not finding.issues
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        assert not (home / "bin" / name).exists()
    # ``bin/`` holds the launchers and the user's own scripts — never those.
    assert (home / "bin" / "hermes").is_file()
    assert (home / "bin" / "mytool").is_file()


@pytest.mark.platforms("posix")
def test_doctor_fix_keeps_the_family_until_the_store_has_uv(home, monkeypatch, capsys):
    """``--fix`` must not trade the install's only uv for none: while PM's store has no uv the
    legacy binary is still what this install runs on (same guard as
    ``update_cmd_maint._notice_legacy_managed_uv``: act only once a private
    target exists). The shadowing is still reported, so the run exits non-zero with a next step."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _uv_binary(home / "bin" / "uv")
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.issues
    assert (home / "bin" / "uv").is_file()
    out = capsys.readouterr().out
    assert "shadow your own uv" in out
    assert "hermes update" in finding.issues[0]


def test_doctor_is_quiet_once_the_family_is_gone(home):
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues
    assert finding.fixed == 0


@pytest.mark.platforms("posix")
def test_doctor_does_not_name_a_linked_bin_as_removable(tmp_path, monkeypatch, capsys):
    """``$HERMES_HOME/bin`` symlinked into an ordinary user bin: doctor must not report the
    user's own uv behind that link as a pre-PM leftover to remove, and ``--fix`` must not
    touch it. The deletion side already refuses (the uninstall twin above); this asserts the
    REPORT side no longer contradicts it by telling the user to delete their own files."""
    from hermes_cli.legacy_uv import LEGACY_MANAGED_UV_NAMES

    home = tmp_path / "hermes"
    outside = tmp_path / "user-bin"
    home.mkdir()
    outside.mkdir()
    for name in LEGACY_MANAGED_UV_NAMES:
        _uv(outside / name, body="users-own-uv")
    (home / "bin").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues, "the user's own uv is not a removable finding"
    assert finding.fixed == 0
    for name in LEGACY_MANAGED_UV_NAMES:
        assert "users-own-uv" in (outside / name).read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "outside the Hermes home" in out
    assert "Remove" not in out


@pytest.mark.platforms("windows")
def test_doctor_names_the_manual_step_on_the_fail_closed_platform(home, capsys):
    """Windows gets the manual-removal instruction, not a retry: on a platform the
    cleanup fails closed on, a "retry --fix" hint would be a finding no fix run
    could ever clear — so it lands in the bucket that survives --fix, and the
    summary still shows it (``_print_summary`` reads issues + manual_issues)."""
    _uv_binary(home / "bin" / "uv.exe")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "manual" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable, so no auto-fix issue"
    assert (home / "bin" / "uv.exe").is_file()
    assert "shadow your own uv" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_manual_step_lands_in_the_bucket_no_fix_run_can_clear(home, capsys, monkeypatch):
    """The fail-closed instruction is a manual issue on any host reaching that branch.

        No ``--fix`` run can ever clear it, so filing it under auto-fixable issues
        would keep it out of the bucket the summary labels "require manual
        intervention". The branch is platform policy, so flipping the flag pins the
        bucket split on every lane; the marked Windows test covers the real gate."""

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    _uv_binary(home / "bin" / "uv.exe")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "manual" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()
    assert "shadow your own uv" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_manual_bucket_when_the_flag_is_rejected_at_runtime(home, monkeypatch):
    """A filesystem that rejects the no-clobber flag at runtime must stop doctor
    advertising a ``--fix`` it cannot run.

    ``automatic_cleanup()`` re-reads the rename binding; the import-time platform
    gate alone would keep saying "retry --fix" forever after the first EINVAL.
    """

    assert legacy_uv.automatic_cleanup() is (legacy_uv._bind_noreplace_rename() is not False)
    monkeypatch.setattr(legacy_uv, "_noreplace_binding", False)
    assert legacy_uv.automatic_cleanup() is False

    _uv_binary(home / "bin" / "uv.exe")
    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()


@pytest.mark.platforms("posix")
def test_fix_routes_to_manual_when_the_rename_rejects_the_flag(home, monkeypatch):
    """A rename rejecting the no-clobber flag mid-run must land this --fix run in
    manual_issues, not a "retry --fix" that can never succeed here."""
    import errno

    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    _uv_binary(home / "bin" / "uv")

    def reject(*_args, **_kwargs):
        legacy_uv._noreplace_binding = False
        raise legacy_uv._NoReplaceUnsupported(errno.ENOTSUP, "ENOTSUP", "uv")

    monkeypatch.setattr(legacy_uv, "_noreplace_binding", None)
    monkeypatch.setattr(legacy_uv, "_rename_noreplace", reject)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "flag rejection must route to the manual step"
    assert not finding.issues, "a retry --fix can never succeed here"
    assert (home / "bin" / "uv").is_file()


@pytest.mark.platforms("posix")
def test_unprobeable_uv_is_manual_on_the_fail_closed_platform(home, capsys, monkeypatch):
    """An unprobeable file obeys the same platform gate as a removable one.

        On a fail-closed platform no ``--fix`` run deletes anything, so the POSIX
        "retry ``hermes doctor --fix``" wording asks for something that can never
        happen. Ungated, this path landed in the auto-fixable bucket and told
        Windows users to retry a repair that cannot run."""

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    _uv_binary(home / "bin" / "uv.exe")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv.exe":
            raise PermissionError("locked by another process")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "no --fix run can clear this, so it is a manual issue"
    assert "manually" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()
    assert "could not be inspected" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_doctor_reports_the_unproven_names_beside_the_removable_ones(home, capsys):
    """One home holding both a provable leftover and a too-small/symlinked name.

    The report-only run never runs the cleanup whose warning would name the
    unproven files, so the headline alone would hide them: doctor must print
    both buckets."""
    _uv_binary(home / "bin" / "uv.exe")
    _uv(home / "bin" / "uvx", body="tiny, not the astral binary")

    finding = doctor_state._check_legacy_uv_shadow(False)

    out = capsys.readouterr().out
    assert "not shaped like a pre-PM uv binary" in out
    assert "uvx" in out, "the unproven name must appear in the review line"
    assert finding.issues, "the removable family stays an unresolved finding"
    assert not any("uvx" in issue for issue in finding.issues), (
        "an unproven file is never a removable finding"
    )
    assert (home / "bin" / "uv.exe").is_file(), "report-only must not delete"
    assert (home / "bin" / "uvx").is_file()


@pytest.mark.platforms("posix")
def test_doctor_reports_only_the_unproven_bucket_when_nothing_is_provable(home, capsys):
    """The unproven-only path stays a clean report (no issue to fix), and the shared
    review line is the same wording the both-buckets path prints."""
    _uv(home / "bin" / "uv", body="the user's own tiny uv")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues
    assert not finding.manual_issues
    out = capsys.readouterr().out
    assert "No legacy-shaped uv binaries" in out
    assert "not shaped like a pre-PM uv binary" in out
    assert (home / "bin" / "uv").is_file()


@pytest.mark.platforms("posix")
def test_doctor_reports_partial_cleanup_as_unresolved(home, monkeypatch):
    """One name failing mid-family: the three that went count as fixes, the one that stayed
    remains an unresolved finding — a clean result over a live leftover is the bug."""
    import pm

    real_unlink = os.unlink

    def _fail_one(name, *, dir_fd=None):
        # Deletion uses the staging name ``.<name>.hermes-cleanup-<pid>``.
        if "uvx.exe" in os.path.basename(str(name)):
            raise PermissionError("simulated unlink failure")
        return real_unlink(name, dir_fd=dir_fd) if dir_fd is not None else real_unlink(name)

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    monkeypatch.setattr("hermes_cli.legacy_uv.os.unlink", _fail_one)
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        _pre_pm_binary(home / "bin" / name)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 3
    assert finding.issues, "the name that stayed is an unresolved finding"
    assert (home / "bin" / "uvx.exe").is_file()
    for name in ("uv", "uvx", "uv.exe"):
        assert not (home / "bin" / name).exists()


# --- Ownership policy: identity says what, never who installed it -----------
# Two copies byte-indistinguishable from a genuine leftover: one the pre-PM
# installer migrated (it wrote no receipt), one an independent install of the
# same tool. Neither a report-only run nor an identity test may remove them, only
# the explicit `hermes doctor --fix` opt-in once PM's own uv exists.

@pytest.mark.platforms("posix")
def test_policy_known_migrated_copy_survives_report_only(home, monkeypatch, capsys):
    """The 'known migration' case: a pre-PM uv exactly where the old installer put
    it. It matches every identity marker, so the report-only run PRESERVES it —
    identity proves what the file is, not that Hermes owns it. The finding names
    the missing receipt instead of treating the shape as provenance."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    legacy = home / "bin" / "uv"
    _uv_binary(legacy)  # exact pre-PM shape: native + uv marker, over the floor

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert legacy.is_file(), "a migrated copy survives the report-only run"
    assert finding.fixed == 0
    assert finding.issues, "the migrated copy is an unresolved finding while present"
    out = capsys.readouterr().out
    assert "shadow your own uv" in out


@pytest.mark.platforms("posix")
def test_policy_migrated_copy_removed_only_by_explicit_fix(home, monkeypatch):
    """The other half of the migrated-copy policy: removal is the explicit
    `--fix` opt-in, gated on PM's own uv existing first — never automatic."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    legacy = home / "bin" / "uv"
    _uv_binary(legacy)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 1, "only the explicit --fix decision removes the copy"
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_policy_independent_install_is_indistinguishable_and_reported_as_own_call(home, monkeypatch, capsys):
    """The 'independent install of the same tool' case: a user's own astral uvx
    in the managed bin classifies REMOVABLE — identity CANNOT separate it from a
    genuine leftover. The policy therefore lives in the wording, not the gate:
    the report must tell the user shape alone is not provenance and deletion is
    their call, and the report-only run must never touch the file."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    foreign = home / "bin" / "uvx"
    _uvx_binary(foreign)  # thin astral launcher: byte-shape of the pre-PM leftover

    assert legacy_uv.classify_leftover(foreign) == legacy_uv.REMOVABLE, (
        "identity cannot prove ownership — the policy must not pretend it can"
    )

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert foreign.is_file(), "an independent install survives the report-only run"
    assert foreign.read_bytes().startswith(b"\x7fELF")
    assert finding.fixed == 0 and finding.issues
    assert any("cannot prove" in m for m in finding.issues + finding.manual_issues), (
        "the report must hand the ownership call to the user"
    )


@pytest.mark.platforms("posix")
def test_a_quarantined_leaf_is_not_reported_as_removed(home, monkeypatch, capsys):
    """A leaf whose dispose and restore both fail must surface as unresolved.

    It sits under its staging name with the public name free, so rechecking the
    public path says "gone" and would report a failed repair as healthy.
    """
    import pm

    from hermes_cli.doctor_state import _check_legacy_uv_shadow

    _pre_pm_binary(home / "bin" / "uv")
    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    staging = f".uv.hermes-cleanup-{os.getpid()}"

    real_unlink, real_rename = os.unlink, legacy_uv._rename_noreplace

    def fail_disposal(name, *, dir_fd=None):
        if os.path.basename(os.fsdecode(name)) == staging:
            raise OSError("disposal refused")
        return real_unlink(name, dir_fd=dir_fd)

    def fail_restore(fd, src, dst):
        if src == staging:
            raise OSError("restore refused")
        return real_rename(fd, src, dst)

    monkeypatch.setattr(os, "unlink", fail_disposal)
    monkeypatch.setattr(legacy_uv, "_rename_noreplace", fail_restore)

    outcome = legacy_uv.remove_legacy_managed_uv(home)
    assert outcome.removed == []
    # ``retained`` names the file that actually exists: the public name is gone, so
    # naming it would send the user to nothing. The quarantine name is the recovery path.
    assert [p.name for p in outcome.retained] == [staging], outcome
    assert outcome.retained[0].is_file(), "the copy stays for manual recovery"
    assert (home / "bin" / staging).is_file()

    finding = _check_legacy_uv_shadow(True)
    assert finding.fixed == 0
    assert finding.issues or finding.manual_issues, (
        "a quarantined copy is an unresolved finding, not a successful delete"
    )
    assert "left 1 uv copy aside" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_a_quarantined_leaf_is_reported_beside_a_removable_one(home, monkeypatch, capsys):
    """Both findings surface in one round, so ``--fix`` cannot leave a clean section.

    A quarantined copy keeps no public name while a by-name sibling still reads
    removable. Reporting only the sibling would let ``--fix`` clear it and print a
    healthy section with the quarantined copy still sitting in ``bin``.
    """
    import pm

    from hermes_cli.doctor_state import _check_legacy_uv_shadow

    _pre_pm_binary(home / "bin" / "uv")
    _pre_pm_binary(home / "bin" / "uvx")
    staging = home / "bin" / f".uv.hermes-cleanup-{os.getpid()}"
    staging.write_bytes(b"\x7fELF" + b"\0" * 4096)
    monkeypatch.setattr(pm, "is_installed", lambda name: True)

    finding = _check_legacy_uv_shadow(False)
    out = capsys.readouterr().out

    assert "left 1 uv copy aside" in out, out
    assert "shadow your own uv" in out, "the removable sibling must still be named"
    assert finding.manual_issues and finding.issues, (finding.manual_issues, finding.issues)


@pytest.mark.platforms("posix")
def test_report_then_fix_then_report_is_a_clean_round_trip(home, monkeypatch, capsys):
    r"""The full loop a user actually walks: see the finding, fix it, see it gone.

    POSIX-only: the loop CLOSES there because ``--fix`` may delete. On a fail-closed
    platform (Windows) the finding lands in ``manual_issues`` with ``fixed == 0`` and no
    delete ever happens, which is a different contract —
    ``test_doctor_names_the_manual_step_on_the_fail_closed_platform`` covers that side.

    Each half is covered separately elsewhere (report-only, and ``--fix`` deletes);
    what this pins is that the loop CLOSES. A ``--fix`` that deletes the file but
    leaves the finding, or one that reports clean while the file survives, would
    pass both halves and still leave the user stuck in it.
    """
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    for name in ("uv", "uvx"):
        _pre_pm_binary(home / "bin" / name)
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    # 1. Report only: the leftover is named and nothing is touched.
    first = doctor_state._check_legacy_uv_shadow(False)
    assert first.issues, "a pre-PM leftover must be an actionable finding"
    assert first.fixed == 0
    assert (home / "bin" / "uv").is_file(), "report-only must not delete"
    assert "shadow your own uv" in capsys.readouterr().out

    # 2. Fix: exactly the family goes.
    second = doctor_state._check_legacy_uv_shadow(True)
    assert second.fixed == 2, second
    assert not (home / "bin" / "uv").exists()
    assert not (home / "bin" / "uvx").exists()
    assert (home / "bin" / "hermes").is_file(), "the launcher is not the leftover"
    capsys.readouterr()

    # 3. Report again: the loop is closed, and a second ``--fix`` is a no-op.
    third = doctor_state._check_legacy_uv_shadow(True)
    assert not third.issues and not third.manual_issues, (
        "the finding survived its own repair: the user has no way to reach a clean state"
    )
    assert third.fixed == 0
    out = capsys.readouterr().out
    assert "shadow your own uv" not in out
