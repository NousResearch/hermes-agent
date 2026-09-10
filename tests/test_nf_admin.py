"""North Forge Full-tier in-session admin trigger (CHG-2026-09-10-001, NF-v0.10.0).

A Full-tier operator can type the drive's admin passcode as a bare message in a
running session to open the SAME Setup Run reconfiguration (`scripts/nf-setup.ps1`
-Force) that nf-setup already provides — tier / pin / edition — without a
re-provision-from-scratch cycle.

`hermes_cli.nf_admin.maybe_recognize_admin_phrase` is the classifier that gates
it. It must be **silent**: on a Basic-tier drive, an unprovisioned drive, plain
upstream Hermes, or a wrong/blank attempt it returns ``None`` and the input
routes exactly as normal chat — zero observable difference. Only an exact
passcode on an active Full-tier drive returns ``"open"``. Every recognized
attempt (a hit, or a plausible miss on a Full drive) is appended to
``<nf-root>/north-forge/admin-attempts.log`` for the owner; the passcode itself
is never written.

These are in-process unit tests; the classifier takes a ``root`` override exactly
like every ``nf_tier`` entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PASSCODE = "forge-master-9x"


def _nf():
    from hermes_cli import nf_tier
    nf_tier.clear_cache()
    return nf_tier


def _admin():
    from hermes_cli import nf_admin
    return nf_admin


@pytest.fixture
def drive(tmp_path):
    """A bare deployed-drive data dir (``<tmp>/data`` == HERMES_HOME)."""
    return tmp_path / "data"


def _provision(data: Path, *, tier: str, pin: str = "default") -> None:
    t = _nf()
    t.write_provisioning(tier=tier, pinned_edition=pin, root=data, overwrite=True)


def _log_lines(data: Path) -> list[str]:
    lp = _nf().admin_attempt_log_path(data)
    if not lp.is_file():
        return []
    return [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- silent paths


def test_unprovisioned_drive_returns_none_and_writes_nothing(drive):
    a = _admin()
    assert a.maybe_recognize_admin_phrase(PASSCODE, root=drive) is None
    assert _log_lines(drive) == []


def test_basic_tier_returns_none_even_with_the_right_passcode(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    _nf().set_admin_passcode(PASSCODE, drive)
    a = _admin()
    assert a.maybe_recognize_admin_phrase(PASSCODE, root=drive) is None
    # the tier gate is checked before verify/logging — nothing recorded
    assert _log_lines(drive) == []


def test_full_tier_without_a_passcode_set_returns_none(drive):
    _provision(drive, tier="full")
    a = _admin()
    assert a.maybe_recognize_admin_phrase(PASSCODE, root=drive) is None
    assert _log_lines(drive) == []


@pytest.mark.parametrize("text", ["", "     ", "abc", "x" * 200, "two words here", " has\tspace "])
def test_wrong_shape_is_never_hashed_or_logged(drive, text):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    a = _admin()
    assert a.maybe_recognize_admin_phrase(text, root=drive) is None
    assert _log_lines(drive) == []


def test_none_text_returns_none(drive):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    assert _admin().maybe_recognize_admin_phrase(None, root=drive) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- recognized


def test_full_tier_exact_passcode_opens_and_logs_the_hit(drive):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    a = _admin()

    assert a.maybe_recognize_admin_phrase(PASSCODE, root=drive) == "open"

    lines = _log_lines(drive)
    assert len(lines) == 1
    assert "reconfig-opened" in lines[0]
    # tab-separated: <iso-ts> <event> <source> <whoami>
    assert lines[0].split("\t")[1] == "reconfig-opened"
    assert lines[0].split("\t")[2] == "cli"
    # the passcode is never written to the log
    assert PASSCODE not in "\n".join(lines)


def test_full_tier_plausible_wrong_passcode_returns_none_but_logs_a_mismatch(drive):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    a = _admin()

    # right shape (bare token, in range), wrong value
    assert a.maybe_recognize_admin_phrase("forge-master-NOPE", root=drive) is None

    lines = _log_lines(drive)
    assert len(lines) == 1
    assert "passcode-mismatch" in lines[0]
    assert "forge-master-NOPE" not in "\n".join(lines)


def test_admin_attempt_log_lives_under_north_forge_and_is_out_of_the_git_tree(drive):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    _admin().maybe_recognize_admin_phrase(PASSCODE, root=drive)

    lp = _nf().admin_attempt_log_path(drive)
    assert lp.parent.name == "north-forge"
    assert lp.parent.parent == drive
    assert lp.name == "admin-attempts.log"


# --------------------------------------------------------------------------- robustness


def test_never_raises_on_a_nonexistent_root(tmp_path):
    a = _admin()
    missing = tmp_path / "no-such-drive"
    assert a.maybe_recognize_admin_phrase("anything-here", root=missing) is None


def test_never_raises_on_a_tampered_record(drive):
    _provision(drive, tier="full")
    _nf().set_admin_passcode(PASSCODE, drive)
    # corrupt the signed record: load() -> STATE_TAMPERED, classifier must still
    # just return None (state != ACTIVE), never propagate
    rec = _nf().record_path(drive)
    rec.write_text(rec.read_text(encoding="utf-8").replace('"tier"', '"tⁱer"'), encoding="utf-8")
    assert _admin().maybe_recognize_admin_phrase(PASSCODE, root=drive) is None


def test_log_admin_attempt_never_raises_when_the_path_is_unwritable(tmp_path):
    # point nf-root at a *file* so mkdir/append fail; log_admin_attempt swallows it
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    _nf().log_admin_attempt("reconfig-opened", source="cli", root=blocker)  # must not raise


# --------------------------------------------------------------------------- reconfig launch (Windows)


@pytest.mark.windows_only
def test_run_nf_reconfig_and_resume_invokes_nf_setup_with_force(monkeypatch, tmp_path):
    """The launch path must pass ``-Force``: the trigger only ever fires on an
    already-provisioned drive, and ``nf-setup.ps1`` without ``-Force`` refuses
    with "already exists — pass --force". Auth is unaffected (nf-setup.ps1 still
    prompts for + verifies the admin passcode)."""
    a = _admin()

    calls: list[list[str]] = []
    monkeypatch.setattr(a, "_nf_setup_script", lambda: REPO_ROOT / "scripts" / "nf-setup.ps1")
    monkeypatch.setattr(a, "_resume", lambda: None)
    monkeypatch.setattr(a, "_log", lambda event: None)
    monkeypatch.setattr(a.subprocess, "call", lambda argv, **kw: calls.append(list(argv)) or 0)

    a.run_nf_reconfig_and_resume()

    assert len(calls) == 1
    argv = calls[0]
    assert "-Force" in argv
    assert argv[0] == "powershell"
    assert argv[-2].endswith("nf-setup.ps1")
    assert argv[-1] == "-Force"
    assert "-File" in argv
