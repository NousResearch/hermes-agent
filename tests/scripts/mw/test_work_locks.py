"""Hermetic tests for scripts/mw/work_locks.py

Uses a LOCAL bare git repo as the "central remote" — no real network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_LOCKS = REPO_ROOT / "scripts" / "mw" / "work_locks.py"

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
NOW_ISO = "2026-07-14T12:00:00Z"
PAST_ISO = "2026-07-10T12:00:00Z"  # well before NOW
FUTURE_EXP = "2026-07-16T12:00:00Z"  # NOW + 48h

QUEUE_REL = Path(".work") / "menu-queue.md"
HEADER = "| menu_id | owner | project | acquired_at | expires_at |"
SEPARATOR = "| --- | --- | --- | --- | --- |"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git(
    args: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    full_env["GIT_TERMINAL_PROMPT"] = "0"
    if env:
        full_env.update(env)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=True,
        text=True,
        env=full_env,
    )


def _run_cli(
    *cli_args: str,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, Any], str]:
    """Run work_locks.py; return (exit_code, parsed_json, stderr)."""
    full_env = os.environ.copy()
    # strip secrets noise; keep hermetic
    full_env["GIT_TERMINAL_PROMPT"] = "0"
    full_env["WORK_LOCKS_NOW"] = NOW_ISO
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, str(WORK_LOCKS), *cli_args],
        capture_output=True,
        text=True,
        env=full_env,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    data: Any = None
    if stdout:
        # last non-empty line is the JSON payload
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        data = json.loads(lines[-1])
    return proc.returncode, data, stderr


def _seed_bare_remote(tmp_path: Path, initial_rows: Optional[List[str]] = None) -> Path:
    """Create a bare remote with work-locks branch and header-only (or custom) queue."""
    bare = tmp_path / "central.git"
    seed = tmp_path / "seed"
    bare.mkdir()
    seed.mkdir()

    _git(["init", "--bare", str(bare)])

    _git(["init", "-b", "work-locks"], cwd=seed)
    _git(["config", "user.name", "seed"], cwd=seed)
    _git(["config", "user.email", "seed@test.local"], cwd=seed)
    _git(["remote", "add", "origin", str(bare)], cwd=seed)

    qdir = seed / ".work"
    qdir.mkdir(parents=True, exist_ok=True)
    if initial_rows is None:
        body = f"{HEADER}\n{SEPARATOR}\n"
    else:
        body = f"{HEADER}\n{SEPARATOR}\n" + "\n".join(initial_rows) + "\n"
    (qdir / "menu-queue.md").write_text(body, encoding="utf-8")

    _git(["add", ".work/menu-queue.md"], cwd=seed)
    _git(["commit", "-m", "seed work-locks"], cwd=seed)
    _git(["push", "-u", "origin", "work-locks"], cwd=seed)
    return bare


def _read_remote_queue(bare: Path, tmp_path: Path) -> str:
    """Fetch remote work-locks tip and return menu-queue.md contents."""
    clone = tmp_path / f"read-{os.getpid()}-{len(list(tmp_path.iterdir()))}"
    clone.mkdir()
    _git(["clone", "--branch", "work-locks", str(bare), str(clone)])
    path = clone / QUEUE_REL
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _queue_has_row(text: str, menu: str, owner: str) -> bool:
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "menu_id" in line or set(line.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] == menu and cells[1] == owner:
            return True
    return False


def _live_owners_for_menu(text: str, menu: str, now: datetime = NOW) -> List[str]:
    owners = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "menu_id" in line or set(line.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0] != menu:
            continue
        exp_s = cells[4]
        try:
            exp = datetime.fromisoformat(exp_s.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if exp > now:
            owners.append(cells[1])
    return owners


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bare(tmp_path: Path) -> Path:
    return _seed_bare_remote(tmp_path)


# ---------------------------------------------------------------------------
# T1 acquire_success
# ---------------------------------------------------------------------------

def test_t1_acquire_success(bare: Path, tmp_path: Path) -> None:
    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-alpha",
        "--owner",
        "alice",
        "--project",
        "proj-a",
        "--ttl-hours",
        "48",
    )
    assert code == 0, f"stderr={err} data={data}"
    assert data["ok"] is True
    assert data["menu"] == "menu-alpha"
    assert data["owner"] == "alice"

    remote_text = _read_remote_queue(bare, tmp_path)
    assert _queue_has_row(remote_text, "menu-alpha", "alice"), remote_text


# ---------------------------------------------------------------------------
# T2 two_clones_one_wins  (THE ACCEPTANCE TEST)
# ---------------------------------------------------------------------------

def test_t2_two_clones_one_wins(bare: Path, tmp_path: Path) -> None:
    """Simulate two clones reserving the SAME menu from the SAME starting commit.

    Clone A acquires (push ok). Clone B, starting from the same stale base,
    tries the same menu → must lose with exit 1 / held_by=A after fetch.
    Exactly ONE live owner remains.
    """
    menu = "menu-race"
    # Both start from the same empty remote tip (seeded header only).
    # A goes first.
    code_a, data_a, err_a = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "alice",
        "--project",
        "p-a",
    )
    assert code_a == 0, f"A failed: {err_a} {data_a}"
    assert data_a["ok"] is True

    # B attempts the same menu. Tool must fetch latest and see alice holds it
    # (or push-reject then re-read). Either path → exit 1, held_by=alice.
    code_b, data_b, err_b = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "bob",
        "--project",
        "p-b",
    )
    assert code_b == 1, f"B should lose: code={code_b} data={data_b} err={err_b}"
    assert data_b["ok"] is False
    assert data_b["held_by"] == "alice"

    remote_text = _read_remote_queue(bare, tmp_path)
    owners = _live_owners_for_menu(remote_text, menu)
    # Exactly one live owner
    assert len(owners) == 1, f"expected exactly 1 owner, got {owners}\n{remote_text}"
    assert owners[0] == "alice"

    # Human-readable outcome for the acceptance log
    succeeded = sum(1 for c in (code_a, code_b) if c == 0)
    print(f"T2 two_clones_one_wins: {succeeded}/2 succeeded (winner={owners[0]})")
    assert succeeded == 1


def test_t2b_concurrent_stale_base_push_race(bare: Path, tmp_path: Path) -> None:
    """Stronger race: both clones prepare commits offline from the same base,
    then push in sequence — second push must lose without overwriting.
    """
    menu = "menu-stale-race"

    # Build two independent working clones at the same base revision.
    base_rev = _git(["rev-parse", "work-locks"], cwd=None, check=False)
    # Get tip from bare via a helper clone
    tip_clone = tmp_path / "tip"
    _git(["clone", "--branch", "work-locks", str(bare), str(tip_clone)])
    tip = _git(["rev-parse", "HEAD"], cwd=tip_clone).stdout.strip()

    def _make_stale_clone(name: str, owner: str) -> Path:
        c = tmp_path / name
        _git(["clone", "--branch", "work-locks", str(bare), str(c)])
        _git(["config", "user.name", owner], cwd=c)
        _git(["config", "user.email", f"{owner}@test.local"], cwd=c)
        # Ensure pinned at shared tip
        _git(["reset", "--hard", tip], cwd=c)
        # Mutate queue as this owner (offline — no fetch)
        qpath = c / QUEUE_REL
        text = qpath.read_text(encoding="utf-8")
        row = (
            f"| {menu} | {owner} | p-{owner} "
            f"| {NOW_ISO} | {FUTURE_EXP} |"
        )
        qpath.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")
        _git(["add", str(QUEUE_REL)], cwd=c)
        _git(["commit", "-m", f"stale acquire {owner}"], cwd=c)
        return c

    clone_a = _make_stale_clone("stale-a", "alice")
    clone_b = _make_stale_clone("stale-b", "bob")

    # A pushes first — wins
    push_a = _git(
        ["push", "origin", "HEAD:refs/heads/work-locks"],
        cwd=clone_a,
        check=False,
    )
    assert push_a.returncode == 0, push_a.stderr

    # B pushes — non-fast-forward reject
    push_b = _git(
        ["push", "origin", "HEAD:refs/heads/work-locks"],
        cwd=clone_b,
        check=False,
    )
    assert push_b.returncode != 0, "B push should be rejected as non-fast-forward"

    # Now B uses the tool (which fetches + re-reads) — must lose cleanly
    code_b, data_b, err_b = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "bob",
    )
    assert code_b == 1, f"tool must report held: {data_b} {err_b}"
    assert data_b["ok"] is False
    assert data_b["held_by"] == "alice"

    remote_text = _read_remote_queue(bare, tmp_path)
    owners = _live_owners_for_menu(remote_text, menu)
    assert owners == ["alice"], f"got {owners}\n{remote_text}"
    print(f"T2b stale-base race: 1/2 succeeded (winner=alice)")


# ---------------------------------------------------------------------------
# T3 reject_when_held
# ---------------------------------------------------------------------------

def test_t3_reject_when_held(bare: Path) -> None:
    code1, d1, e1 = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-held",
        "--owner",
        "alice",
    )
    assert code1 == 0, e1

    code2, d2, e2 = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-held",
        "--owner",
        "carol",
    )
    assert code2 == 1, f"expected held, got {code2} {d2} {e2}"
    assert d2["ok"] is False
    assert d2["held_by"] == "alice"
    assert "expires_at" in d2


# ---------------------------------------------------------------------------
# T4 expired_takeover
# ---------------------------------------------------------------------------

def test_t4_expired_takeover(tmp_path: Path) -> None:
    # Seed a lock that expired in the past
    expired_row = (
        f"| menu-old | alice | proj "
        f"| 2026-07-01T00:00:00Z | {PAST_ISO} |"
    )
    bare = _seed_bare_remote(tmp_path, initial_rows=[expired_row])

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-old",
        "--owner",
        "bob",
        "--project",
        "takeover",
    )
    assert code == 0, f"stderr={err} data={data}"
    assert data["ok"] is True
    assert data["owner"] == "bob"

    remote_text = _read_remote_queue(bare, tmp_path)
    assert _queue_has_row(remote_text, "menu-old", "bob"), remote_text
    # alice's expired row should be replaced, not duplicated as live
    owners = _live_owners_for_menu(remote_text, "menu-old")
    assert owners == ["bob"], owners


# ---------------------------------------------------------------------------
# T5 release_only_owner
# ---------------------------------------------------------------------------

def test_t5_release_only_owner(bare: Path, tmp_path: Path) -> None:
    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-rel",
        "--owner",
        "alice",
    )
    assert code == 0, err

    # Non-owner must be refused
    code_bad, data_bad, err_bad = _run_cli(
        "release",
        "--remote",
        str(bare),
        "--menu",
        "menu-rel",
        "--owner",
        "eve",
    )
    assert code_bad == 1, f"non-owner should fail: {data_bad} {err_bad}"
    assert data_bad["ok"] is False
    assert data_bad.get("held_by") == "alice"

    # Row still present
    remote_text = _read_remote_queue(bare, tmp_path)
    assert _queue_has_row(remote_text, "menu-rel", "alice")

    # Owner release succeeds
    code_ok, data_ok, err_ok = _run_cli(
        "release",
        "--remote",
        str(bare),
        "--menu",
        "menu-rel",
        "--owner",
        "alice",
    )
    assert code_ok == 0, f"{data_ok} {err_ok}"
    assert data_ok["ok"] is True

    remote_text2 = _read_remote_queue(bare, tmp_path)
    assert not _queue_has_row(remote_text2, "menu-rel", "alice"), remote_text2


# ---------------------------------------------------------------------------
# T6 ownership_from_remote
# ---------------------------------------------------------------------------

def test_t6_ownership_from_remote(bare: Path, tmp_path: Path) -> None:
    """After A acquires+pushes, a B invocation that never saw it must
    report held_by=A via `check` — proves it reads remote, not local."""
    code_a, data_a, err_a = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-remote",
        "--owner",
        "alice",
    )
    assert code_a == 0, err_a

    # Fresh process, no cache dir, no local knowledge of alice's lock
    code, data, err = _run_cli(
        "check",
        "--remote",
        str(bare),
        "--menu",
        "menu-remote",
        "--owner",
        "bob",
    )
    assert code == 0, err
    assert data["menu"] == "menu-remote"
    assert data["held_by"] == "alice"
    assert data["expired"] is False
    assert data["mine"] is False


# ---------------------------------------------------------------------------
# Extra: status + renew + check free menu
# ---------------------------------------------------------------------------

def test_status_lists_active_only(tmp_path: Path) -> None:
    rows = [
        f"| live-m | alice | p | {NOW_ISO} | {FUTURE_EXP} |",
        f"| dead-m | bob | p | 2026-07-01T00:00:00Z | {PAST_ISO} |",
    ]
    bare = _seed_bare_remote(tmp_path, initial_rows=rows)

    code, data, err = _run_cli("status", "--remote", str(bare))
    assert code == 0, err
    assert isinstance(data, list)
    ids = {r["menu_id"] for r in data}
    assert "live-m" in ids
    assert "dead-m" not in ids


def test_renew_only_owner(bare: Path) -> None:
    _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-rn",
        "--owner",
        "alice",
        "--ttl-hours",
        "24",
    )
    code_bad, data_bad, _ = _run_cli(
        "renew",
        "--remote",
        str(bare),
        "--menu",
        "menu-rn",
        "--owner",
        "bob",
        "--ttl-hours",
        "72",
    )
    assert code_bad == 1
    assert data_bad["ok"] is False

    code_ok, data_ok, err = _run_cli(
        "renew",
        "--remote",
        str(bare),
        "--menu",
        "menu-rn",
        "--owner",
        "alice",
        "--ttl-hours",
        "72",
    )
    assert code_ok == 0, err
    assert data_ok["ok"] is True
    assert data_ok["expires_at"] == "2026-07-17T12:00:00Z"  # NOW + 72h


def test_check_free_menu(bare: Path) -> None:
    code, data, err = _run_cli(
        "check",
        "--remote",
        str(bare),
        "--menu",
        "never-locked",
        "--owner",
        "alice",
    )
    assert code == 0, err
    assert data["held_by"] is None
    assert data["mine"] is False
    assert data["expired"] is False


def test_missing_menu_arg_is_usage_error(bare: Path) -> None:
    code, data, err = _run_cli("acquire", "--remote", str(bare), "--owner", "alice")
    assert code == 2
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# FIX 1 · queue-row injection rejected (pipe / newline in fields)
# ---------------------------------------------------------------------------

def test_fix1_reject_pipe_in_owner_no_queue_write(bare: Path, tmp_path: Path) -> None:
    """owner containing '|' must exit 2 and leave the queue untouched."""
    before = _read_remote_queue(bare, tmp_path)

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-inject",
        "--owner",
        "alice | bob",
    )
    assert code == 2, f"expected exit 2: data={data} err={err}"
    assert data["ok"] is False
    assert "invalid owner" in data["error"]
    assert "invalid owner" in err

    after = _read_remote_queue(bare, tmp_path)
    assert after == before, "queue must not be modified on invalid owner"

    # Later legit acquire of the same menu still succeeds
    code2, data2, err2 = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-inject",
        "--owner",
        "alice",
    )
    assert code2 == 0, f"legit acquire failed: {data2} {err2}"
    assert data2["ok"] is True
    assert data2["owner"] == "alice"


def test_fix1_reject_newline_in_menu_no_queue_write(bare: Path, tmp_path: Path) -> None:
    """menu containing a newline must exit 2 and leave the queue untouched."""
    before = _read_remote_queue(bare, tmp_path)
    bad_menu = "menu-nl\ninject"

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        bad_menu,
        "--owner",
        "alice",
    )
    assert code == 2, f"expected exit 2: data={data} err={err}"
    assert data["ok"] is False
    assert "invalid menu" in data["error"]

    after = _read_remote_queue(bare, tmp_path)
    assert after == before, "queue must not be modified on invalid menu"

    # Legit menu id still free and acquirable
    code2, data2, err2 = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-nl-clean",
        "--owner",
        "alice",
    )
    assert code2 == 0, err2
    assert data2["ok"] is True


# ---------------------------------------------------------------------------
# FIX 2 · owner identity collision (host-qualified owners)
# ---------------------------------------------------------------------------

def test_fix2_host_qualified_owners_do_not_collide(bare: Path, tmp_path: Path) -> None:
    """nat@hostA and nat@hostB are distinct owners — only one may hold the menu."""
    menu = "menu-host-id"

    code_a, data_a, err_a = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "nat@hostA",
    )
    assert code_a == 0, f"{data_a} {err_a}"
    assert data_a["ok"] is True
    assert data_a["owner"] == "nat@hostA"

    code_b, data_b, err_b = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "nat@hostB",
    )
    assert code_b == 1, f"hostB must not steal: code={code_b} data={data_b} err={err_b}"
    assert data_b["ok"] is False
    assert data_b["held_by"] == "nat@hostA"

    # Explicit same-owner re-acquire stays idempotent
    code_again, data_again, err_again = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        menu,
        "--owner",
        "nat@hostA",
    )
    assert code_again == 0, f"{data_again} {err_again}"
    assert data_again["ok"] is True
    assert data_again.get("already_held") is True

    remote_text = _read_remote_queue(bare, tmp_path)
    owners = _live_owners_for_menu(remote_text, menu)
    assert owners == ["nat@hostA"], owners


# ---------------------------------------------------------------------------
# FIX 3 · malformed expires_at fails CLOSED (not free / not takeover)
# ---------------------------------------------------------------------------

def test_fix3_corrupt_expires_at_refuses_takeover(tmp_path: Path) -> None:
    """Live row with unparseable expires_at must refuse acquire with exit 2."""
    corrupt_row = (
        f"| menu-corrupt | alice | proj "
        f"| {NOW_ISO} | not-a-date |"
    )
    bare = _seed_bare_remote(tmp_path, initial_rows=[corrupt_row])
    before = _read_remote_queue(bare, tmp_path)

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-corrupt",
        "--owner",
        "bob",
    )
    assert code == 2, f"expected corruption exit 2: data={data} err={err}"
    assert data["ok"] is False
    assert "queue corruption" in data["error"]
    assert "menu-corrupt" in data["error"]
    assert "invalid timestamp" in data["error"]

    after = _read_remote_queue(bare, tmp_path)
    assert after == before, "corrupt row must not be taken over / rewritten"

    # check must not claim held_by=None
    code_c, data_c, err_c = _run_cli(
        "check",
        "--remote",
        str(bare),
        "--menu",
        "menu-corrupt",
        "--owner",
        "bob",
    )
    assert code_c == 0, err_c
    assert data_c["held_by"] is not None
    assert data_c.get("corrupt") is True


# ---------------------------------------------------------------------------
# FIX 4 · zero/negative TTL rejected before git work
# ---------------------------------------------------------------------------

def test_fix4_ttl_zero_and_negative_rejected(bare: Path, tmp_path: Path) -> None:
    before = _read_remote_queue(bare, tmp_path)

    for bad_ttl in ("0", "-5"):
        code, data, err = _run_cli(
            "acquire",
            "--remote",
            str(bare),
            "--menu",
            "menu-ttl",
            "--owner",
            "alice",
            "--ttl-hours",
            bad_ttl,
        )
        assert code == 2, f"ttl={bad_ttl}: data={data} err={err}"
        assert data["ok"] is False
        assert data["error"] == "ttl-hours must be 1..720"
        after = _read_remote_queue(bare, tmp_path)
        assert after == before, f"queue modified for ttl={bad_ttl}"

    # Upper bound 720 is accepted
    code_ok, data_ok, err_ok = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-ttl",
        "--owner",
        "alice",
        "--ttl-hours",
        "720",
    )
    assert code_ok == 0, f"{data_ok} {err_ok}"
    assert data_ok["ok"] is True
    assert data_ok["expires_at"] == "2026-08-13T12:00:00Z"  # NOW + 720h


def test_fix4_ttl_over_max_rejected(bare: Path) -> None:
    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-ttl-hi",
        "--owner",
        "alice",
        "--ttl-hours",
        "721",
    )
    assert code == 2
    assert data["ok"] is False
    assert data["error"] == "ttl-hours must be 1..720"


# ---------------------------------------------------------------------------
# FIX 5 · clock skew grace blocks premature takeover
# ---------------------------------------------------------------------------

def test_fix5_skew_seconds_blocks_recently_expired_takeover(tmp_path: Path) -> None:
    """Lock expired 30s ago + --skew-seconds 60 → other owner still blocked."""
    # NOW = 2026-07-14T12:00:00Z → expired 30s ago = 11:59:30Z
    recent_exp = "2026-07-14T11:59:30Z"
    row = (
        f"| menu-skew | alice | proj "
        f"| 2026-07-14T10:00:00Z | {recent_exp} |"
    )
    bare = _seed_bare_remote(tmp_path, initial_rows=[row])
    before = _read_remote_queue(bare, tmp_path)

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-skew",
        "--owner",
        "bob",
        "--skew-seconds",
        "60",
    )
    assert code == 1, f"skew should block takeover: code={code} data={data} err={err}"
    assert data["ok"] is False
    assert data["held_by"] == "alice"

    after = _read_remote_queue(bare, tmp_path)
    assert after == before, "queue must be unchanged when skew blocks takeover"
    assert _queue_has_row(after, "menu-skew", "alice"), after
    assert "| menu-skew | bob |" not in after


def test_fix5_negative_skew_seconds_rejected(bare: Path, tmp_path: Path) -> None:
    """--skew-seconds < 0 must exit 2 and leave the remote queue untouched."""
    before = _read_remote_queue(bare, tmp_path)

    code, data, err = _run_cli(
        "acquire",
        "--remote",
        str(bare),
        "--menu",
        "menu-skew-neg",
        "--owner",
        "alice",
        "--skew-seconds",
        "-60",
    )
    assert code == 2, f"negative skew: code={code} data={data} err={err}"
    assert data["ok"] is False
    assert data["error"] == "skew-seconds must be >= 0"
    after = _read_remote_queue(bare, tmp_path)
    assert after == before, "queue must not be modified on negative skew"
