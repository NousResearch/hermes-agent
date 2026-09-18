"""Bug class: ``hermes update`` must fail closed when the pre-update backup
actually FAILS, not when it merely has nothing to back up. Companion to
#114592.

Three contract pins live here (each ties to the same root-cause pair the
issue filed against):

A. ``_run_pre_update_backup`` propagates failure rather than swallowing it.
   - When ``mode="quick"`` and ``_run_quick_snapshots`` raises, the caller
     sees a structured failure signal (not a silent ``None``).
   - When ``mode="full"`` and the full zip backup raises, the caller sees
     a structured failure signal even though the quick snapshot succeeded.
   - When ``mode="off"`` (explicit user opt-out), the caller sees
     ``(snapshot_id=None, full_backup_ok=None)`` — opt-out is still honored.

B. ``_cmd_update_impl`` gates the destructive apply on the failure signal.
   - Pre-swap destructors (Windows venv-holder sweep, ZIP fallback, git
     fetch/merge) are NOT entered when backup failure was signalled.
   - The update receipt's ``pre_update_backup`` step records the failure
     and the run exits non-zero before any code swap is attempted.
   - When backup succeeds, the existing happy path is unchanged (no
     regression on the normal ``mode="quick"`` flow).

C. ``_ensure_default_soul_md`` only replaces a *cyclic* SOUL.md symlink.
   Resolving and dangling links are left untouched so operator wiring
   survives the first-boot seed (#114601 review feedback — alt-glitch
   triage + KeyArgo's three-row Linux verification). A cyclic link is
   replaced atomically via tempfile + os.replace so concurrent boot
   attempts cannot race on one shared path.

Only paths/env are monkeypatched; every behavior under test uses the real
``hermes_cli.update_cmd_maint`` / ``hermes_cli.config`` modules against a
temp HERMES_HOME so the contract survives a refactor of those modules.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# A. _run_pre_update_backup propagation contract
# ---------------------------------------------------------------------------


def test_run_pre_update_backup_quick_failure_propagates(monkeypatch):
    """``mode=quick`` with the quick snapshot raising must surface a
    structured failure rather than silently returning ``None`` (the prior
    contract made None mean both "nothing to snapshot" and "everything
    blew up"; the receipt could not distinguish)."""
    import hermes_cli.update_cmd_maint as ucm

    def boom_quick():
        raise RuntimeError("simulated quick snapshot failure")

    monkeypatch.setattr(ucm, "_run_quick_snapshots", boom_quick)
    monkeypatch.setattr(
        ucm, "_resolve_pre_update_backup_mode",
        lambda _args: "quick",
    )

    args = Namespace(no_backup=False)
    with pytest.raises(RuntimeError, match="simulated quick snapshot failure"):
        ucm._run_pre_update_backup(args)


def test_run_pre_update_backup_full_zip_failure_propagates(monkeypatch):
    """``mode=full`` with the zip backup raising must surface failure even
    when the quick snapshot succeeded — a partial backup is exactly the
    failure class #114592 reports."""
    import hermes_cli.update_cmd_maint as ucm

    monkeypatch.setattr(
        ucm, "_resolve_pre_update_backup_mode",
        lambda _args: "full",
    )
    monkeypatch.setattr(ucm, "_run_quick_snapshots", lambda: "snap-abc")

    def boom_zip():
        raise RuntimeError("simulated zip backup failure")

    monkeypatch.setattr(ucm, "_run_full_backup", boom_zip)

    args = Namespace(no_backup=False)
    with pytest.raises(RuntimeError, match="simulated zip backup failure"):
        ucm._run_pre_update_backup(args)


def test_run_pre_update_backup_off_mode_returns_none_silently(monkeypatch):
    """``mode=off`` is the explicit user opt-out path; that contract is
    preserved. The caller treats ``None`` as "user opted out", which is
    distinct from "backup failed"."""
    import hermes_cli.update_cmd_maint as ucm

    monkeypatch.setattr(
        ucm, "_resolve_pre_update_backup_mode",
        lambda _args: "off",
    )
    # Even if these would raise, the off mode short-circuits before them.
    monkeypatch.setattr(
        ucm, "_run_quick_snapshots",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called in off mode")),
    )

    args = Namespace(no_backup=False)
    assert ucm._run_pre_update_backup(args) is None


def test_run_pre_update_backup_quick_success_returns_snapshot_id(monkeypatch):
    """Happy path: ``mode=quick`` with a successful snapshot returns the
    snapshot id unchanged. No regression on the existing contract."""
    import hermes_cli.update_cmd_maint as ucm

    monkeypatch.setattr(
        ucm, "_resolve_pre_update_backup_mode",
        lambda _args: "quick",
    )
    monkeypatch.setattr(ucm, "_run_quick_snapshots", lambda: "snap-ok")

    args = Namespace(no_backup=False)
    assert ucm._run_pre_update_backup(args) == "snap-ok"


# ---------------------------------------------------------------------------
# B. _cmd_update_impl fail-closed gate on the backup signal
# ---------------------------------------------------------------------------


def test_cmd_update_impl_refuses_to_apply_when_backup_raises(monkeypatch, tmp_path):
    """When ``_run_pre_update_backup`` raises (per contract A), the
    destructive apply path is NOT entered — the receipt records the
    failure and the run exits non-zero."""
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.main as cli_main

    project_root = tmp_path / "project"
    project_root.mkdir()

    # All gates set up so the only thing the test exercises is the
    # backup-failure branch.
    recorded_steps = []

    def record_step(step, ok, detail=""):
        recorded_steps.append((step, ok, detail))

    def boom_backup(_args):
        raise RuntimeError("backup failed")

    monkeypatch.setattr(update_cmd, "_run_pre_update_backup", boom_backup)
    monkeypatch.setattr(update_cmd, "_record_update_step", record_step)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan",
                        lambda _args: SimpleNamespace(in_place_update=True))
    monkeypatch.setattr(update_cmd, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(cli_main, "_is_windows", lambda: False)
    monkeypatch.setattr(update_cmd, "_apply_pulled_update",
                        lambda *_a, **_kw: pytest.fail("apply path reached despite backup failure"))
    monkeypatch.setattr(update_cmd, "_update_via_zip",
                        lambda *_a, **_kw: pytest.fail("zip fallback reached despite backup failure"))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", project_root)

    args = Namespace(post_swap=None, yes=False, no_backup=False)

    with pytest.raises(RuntimeError, match="backup failed"):
        update_cmd._cmd_update_impl(args, gateway_mode=False)

    # The failure step MUST be recorded (the receipt is the only signal a
    # user has that something went wrong before any code swap).
    assert any(step == "pre_update_backup" and ok is False for step, ok, _ in recorded_steps), (
        "expected a pre_update_backup step to be recorded as failed, "
        f"got steps={recorded_steps!r}"
    )


# ---------------------------------------------------------------------------
# C. _ensure_default_soul_md only replaces a CYCLIC SOUL.md symlink
# ---------------------------------------------------------------------------


def test_ensure_default_soul_md_replaces_cyclic_symlink_with_regular_file(tmp_path):
    """If SOUL.md is a self-referential symlink (``SOUL.md -> SOUL.md``),
    ``_ensure_default_soul_md`` must replace it with a regular text file
    the gateway can read on next boot. The prior pre-narrow behavior
    wrote through the loop and raised ``OSError(ELOOP)`` mid-call, which
    propagated as ``HomeInitializationError`` and killed every gateway
    spawn with exit 75 (#114592). The fix is scoped to ELOOP: stat() on
    a cyclic chain raises ``OSError(errno.ELOOP)`` and only that branch
    atomically seeds ``DEFAULT_SOUL_MD`` via tempfile + ``os.replace``.
    """
    import hermes_cli.config as cfg

    soul_path = tmp_path / "SOUL.md"
    os.symlink(str(soul_path), str(soul_path))

    assert soul_path.is_symlink()

    cfg._ensure_default_soul_md(tmp_path)

    assert not soul_path.is_symlink(), (
        "_ensure_default_soul_md left a cyclic link in place; gateway "
        "boot would re-trigger the ELOOP exit-75 from #114592"
    )
    assert soul_path.is_file(), "SOUL.md must be a regular file after _ensure_default_soul_md"
    assert soul_path.read_text(encoding="utf-8") == cfg.DEFAULT_SOUL_MD

    # The atomic seed must not leave a tempfile next to SOUL.md.
    stragglers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".SOUL.md.")]
    assert not stragglers, (
        f"atomic seed left tempfile(s) in {tmp_path}: {stragglers!r}"
    )


def test_ensure_default_soul_md_preserves_resolving_link_to_custom_content(tmp_path):
    """A resolving ``SOUL.md -> /elsewhere/real-soul.md`` link keeps its
    custom content intact — operator wiring must survive a first boot.
    This is the row KeyArgo verified on Linux against the prior head:
    the unconditional ``is_symlink() -> unlink()`` clobbered the link
    and overwrote the target's identity. The narrowed fix only replaces
    a link that ``stat()`` cannot resolve (ELOOP); a resolving link
    stat()s cleanly and falls through to the read/compare branch.
    """
    import hermes_cli.config as cfg

    target = tmp_path / "real-soul.md"
    custom = "# operator identity\ndo not touch\n"
    target.write_text(custom, encoding="utf-8")

    soul_path = tmp_path / "SOUL.md"
    soul_path.symlink_to(target)

    cfg._ensure_default_soul_md(tmp_path)

    assert soul_path.is_symlink(), (
        "resolving SOUL.md link must be preserved (operator wiring); "
        "KeyArgo's three-row verification (#114601 review) confirms main "
        "treats this case as untouched"
    )
    assert soul_path.resolve() == target.resolve()
    assert soul_path.read_text(encoding="utf-8") == custom
    assert target.read_text(encoding="utf-8") == custom


def test_ensure_default_soul_md_preserves_dangling_link_and_writes_through_it(tmp_path):
    """A dangling ``SOUL.md -> /elsewhere/missing.md`` link is left in
    place; the default seed is written THROUGH the link so the target
    becomes the identity file. This matches main's pre-#114592 behaviour
    (KeyArgo row 3): a dangling link is operator wiring that we must
    not delete, and the existing read/compare path handles the write
    transparently because ``read_text`` / ``write_text`` follow the link.
    """
    import hermes_cli.config as cfg

    target = tmp_path / "missing-target.md"
    assert not target.exists()

    soul_path = tmp_path / "SOUL.md"
    soul_path.symlink_to(target)

    cfg._ensure_default_soul_md(tmp_path)

    assert soul_path.is_symlink(), (
        "dangling SOUL.md link must be preserved; main behaviour writes "
        "through the link and lets the target become the identity file"
    )
    assert target.exists(), "the default seed must be written through the dangling link"
    assert target.read_text(encoding="utf-8") == cfg.DEFAULT_SOUL_MD


def test_ensure_default_soul_md_writes_regular_file_when_missing(tmp_path):
    """Happy path: fresh home with no SOUL.md gets a regular (non-symlink)
    file with the default content."""
    import hermes_cli.config as cfg

    cfg._ensure_default_soul_md(tmp_path)
    soul_path = tmp_path / "SOUL.md"
    assert soul_path.is_file()
    assert not soul_path.is_symlink()
    assert soul_path.read_text(encoding="utf-8") == cfg.DEFAULT_SOUL_MD


def test_ensure_default_soul_md_preserves_user_content(tmp_path):
    """A user-customized SOUL.md is not touched (existing contract)."""
    import hermes_cli.config as cfg

    soul_path = tmp_path / "SOUL.md"
    user_content = "# my custom soul — do not overwrite"
    soul_path.write_text(user_content, encoding="utf-8")

    cfg._ensure_default_soul_md(tmp_path)
    assert soul_path.read_text(encoding="utf-8") == user_content
    assert not soul_path.is_symlink()
