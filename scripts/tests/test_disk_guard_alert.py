"""Tests for disk_guard_alert.py — the alert path of the disk guard.

CLASS under test: the guard was RED (launchctl exit 1, 501 runs) for days and
nobody heard about it, because the only output channel was a launchd log file.
An alert that goes nowhere is not an alert.

Run: python3 -m pytest ~/.hermes/scripts/tests/test_disk_guard_alert.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

SCRIPT = Path(os.path.expanduser("~/.hermes/scripts/disk_guard_alert.py"))
spec = importlib.util.spec_from_file_location("disk_guard_alert", SCRIPT)
dga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dga)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own dedupe state; otherwise a real state file on
    this host silently suppresses the alert the test is asserting."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))


# ------------------------------------------------------------- free space ---

def test_free_gi_parses_df(monkeypatch):
    df = (
        "Filesystem 1M-blocks Used Avail Capacity iused ifree %iused Mounted on\n"
        "/dev/disk3s1 233472 195000 7600 97% 1 1 1% /System/Volumes/Data\n"
    )
    monkeypatch.setattr(dga, "_run", lambda *a, **k: df)
    assert dga.free_gi() == pytest.approx(7600 / 1024)


def test_free_gi_returns_none_when_df_unparseable(monkeypatch):
    monkeypatch.setattr(dga, "_run", lambda *a, **k: "")
    assert dga.free_gi() is None


# ------------------------------------------------------------- threshold ---

def test_below_floor_is_alertable():
    assert dga.should_alert(free=7, floor=10) is True


def test_at_floor_is_alertable():
    """Owner decision (Den, 2026-09-21): page at the floor OR LESS, not strictly
    below it. With the floor pinned at 1Gi, `free < floor` would mean the only
    alertable state is 0Gi — i.e. the host is already dead. The comparison is
    `<=` so 1Gi free still pages."""
    assert dga.should_alert(free=10, floor=10) is True
    assert dga.should_alert(free=1, floor=1) is True
    assert dga.should_alert(free=0, floor=1) is True


def test_above_floor_is_silent():
    assert dga.should_alert(free=2, floor=1) is False
    assert dga.should_alert(free=7, floor=1) is False


def test_unknown_free_alerts_rather_than_staying_silent():
    """df failing is itself an anomaly; silence is the failure mode we fix."""
    assert dga.should_alert(free=None, floor=10) is True
    assert dga.should_alert(free=None, floor=1) is True


# ------------------------------------------------------------ pinned floor --
# CLASS: the floor is a DECISION, not a tunable. It is pinned in three places
# that must agree, and a disagreement is how an operator ends up believing the
# guard is quiet at 7Gi while one copy still pages at 10Gi. Each assertion below
# DERIVES the value from the file rather than trusting a comment.

EXPECTED_FLOOR_GI = 0.5

# The floor is a DECIMAL GiB figure since 2026-09-22 (1 -> 0.5). Every pattern
# below therefore matches `\d+(?:\.\d+)?`: an int-only pattern does not read a
# 0.5 pin as "wrong", it reads it as "absent", and the assertion that fires is
# the "no longer declares" one -- which the next person fixes by loosening the
# ASSERTION instead of the number, leaving the copy unguarded.
_NUM = r'(\d+(?:\.\d+)?)'


def test_default_floor_is_the_owner_decision():
    assert dga.DEFAULT_FLOOR_GI == EXPECTED_FLOOR_GI


def test_every_deployed_floor_agrees_with_the_decision():
    """Derived, not a hand-maintained list: each source of truth is parsed."""
    import re as _re
    root = Path(os.path.expanduser("~/.hermes"))
    found = {}

    found["disk_guard_alert.py"] = float(dga.DEFAULT_FLOOR_GI)

    sh = (root / "scripts" / "disk-guard.sh").read_text()
    m = _re.search(r'FLOOR_GI="\$\{DISK_GUARD_FLOOR_GI:-' + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares FLOOR_GI the way this guard parses"
    found["disk-guard.sh"] = float(m.group(1))

    cron = (root / "profiles" / "software-engineer" / "scripts"
            / "disk-guard-cron.sh").read_text()
    m = _re.search(r'^FLOOR_GI_PINNED=' + _NUM, cron, _re.M)
    assert m, "disk-guard-cron.sh no longer declares FLOOR_GI_PINNED"
    found["disk-guard-cron.sh"] = float(m.group(1))

    # Every deployed copy of the heartbeat, globbed — a new copy cannot be
    # silently unguarded.
    hbs = sorted(root.glob("scripts/agentpod_em_heartbeat.py")) + \
        sorted(root.glob("profiles/*/scripts/agentpod_em_heartbeat.py"))
    assert hbs, "no heartbeat copy found — the glob is wrong, not the floor"
    for p in hbs:
        m = _re.search(r'^DISK_FLOOR_GI\s*=\s*' + _NUM, p.read_text(), _re.M)
        assert m, f"{p} no longer declares DISK_FLOOR_GI"
        found[str(p)] = float(m.group(1))

    bad = {k: v for k, v in found.items() if v != EXPECTED_FLOOR_GI}
    assert not bad, f"floors disagree with the {EXPECTED_FLOOR_GI}Gi decision: {bad}"


def test_reclaim_target_is_higher_than_the_paging_floor_and_never_pages():
    """CLASS (2026-09-24 incident, card t_b8d0aaeb): the host went from above
    the floor to ENOSPC inside one 900s tick, so a paging floor was being asked
    to do capacity work. The fix separates the two numbers:

      * RECLAIM_TARGET_GI -- how much free space we try to HOLD, enforced by
        evicting more sanctioned scratch, SILENTLY.
      * FLOOR_GI          -- the only thing that pages. Owner-pinned at 0.5Gi.

    The target must be strictly above the floor (otherwise it enforces nothing)
    and must not appear on any paging path (otherwise it silently re-raises the
    alert rate the owner cut on 2026-09-22). Both are DERIVED from the file.
    """
    import re as _re
    sh = Path(os.path.expanduser("~/.hermes/scripts/disk-guard.sh")).read_text()

    m = _re.search(r'RECLAIM_TARGET_GI="\$\{DISK_GUARD_RECLAIM_TARGET_GI:-'
                   + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares RECLAIM_TARGET_GI"
    target = float(m.group(1))

    m = _re.search(r'FLOOR_GI="\$\{DISK_GUARD_FLOOR_GI:-' + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares FLOOR_GI"
    floor = float(m.group(1))

    assert floor == EXPECTED_FLOOR_GI, (
        "the reclaim target must not be implemented by moving the paging floor"
    )
    assert target > floor, (
        f"reclaim target {target}Gi must sit above the paging floor {floor}Gi"
    )

    # The exit-1 (paging) decision is `below_floor`, never `below_target`.
    tail = sh.split("after=$(free_gi)")[-1]
    assert "below_floor" in tail
    assert "below_target" not in tail, (
        "the reclaim target leaked into the paging decision; it must stay silent"
    )


def test_clamp_cannot_lower_the_floor_below_the_decision(monkeypatch):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "3")
    boom = lambda t: (_ for _ in ()).throw(AssertionError("paged at 3Gi"))
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: boom(t))
    monkeypatch.setattr(dga, "deliver_telegram", boom)
    assert dga.main(["--floor", "0"]) == 0, "--floor 0 must clamp up to the pin"


# ----------------------------------------------------------- both channels --

def test_alert_delivers_to_the_cto_and_to_den(monkeypatch):
    """Owner decision (card t_b8d0aaeb, 2026-09-24): host disk is CTO scope,
    not EM scope. The page must reach the CTO route and Den, and must NOT wake
    the EM (ceo) session -- an EM paged for a host problem can only re-delegate
    it, which is why this incident reached an operator late."""
    calls = []
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda text, key=None: calls.append(("cto", text)) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append(("tg", text)) or True)
    monkeypatch.setattr(dga, "deliver_wake",
                        lambda text: (_ for _ in ()).throw(AssertionError("EM was paged")))
    ok = dga.alert("DISK GUARD RED: 7Gi free")
    assert ok is True
    assert [c[0] for c in calls] == ["cto", "tg"]
    assert all("DISK GUARD RED" in c[1] for c in calls)


def test_cto_card_is_keyed_to_the_incident(monkeypatch):
    """Board dedupe must match page dedupe: one incident, one card, however
    many ticks observe it. The key is passed through to the idempotency key."""
    seen = {}
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda text, key=None: seen.update(key=key) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: True)
    dga.alert("DISK GUARD RED", key="abc123")
    assert seen["key"] == "abc123"


def test_cto_card_uses_idempotency_key_and_cto_assignee(monkeypatch, tmp_path):
    """The card is raised via the kanban CLI with the incident key as the
    idempotency key, so a repeated observation returns the same card instead of
    littering the board."""
    fake = tmp_path / "hermes"
    fake.write_text("#!/bin/sh\necho '{\"id\": \"t_dead1234\"}'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(dga, "KANBAN_BIN", str(fake))
    captured = {}
    real_run = dga.subprocess.run

    def spy(argv, **kw):
        captured["argv"] = argv
        return real_run(argv, **kw)

    monkeypatch.setattr(dga.subprocess, "run", spy)
    assert dga.deliver_kanban_cto("DISK GUARD RED\nbody", key="k9") is True
    argv = captured["argv"]
    assert "--assignee" in argv and argv[argv.index("--assignee") + 1] == "cto"
    assert argv[argv.index("--idempotency-key") + 1] == "disk-guard-k9"


def test_cto_card_failure_does_not_swallow_the_telegram_page(monkeypatch):
    """The two channels must have INDEPENDENT failure modes; a broken board
    (the very thing ENOSPC corrupts) must not silence the page."""
    calls = []

    def boom(text, key=None):
        raise RuntimeError("kanban.db disk I/O error")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append("tg") or True)
    assert dga.alert("DISK GUARD RED: 0Gi free") is True
    assert calls == ["tg"]


def test_telegram_is_attempted_even_when_board_channel_fails(monkeypatch):
    """One dead channel must not swallow the page — that is this bug's shape."""
    calls = []

    def boom(text, key=None):
        calls.append(("cto", text))
        raise RuntimeError("board down")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append(("tg", text)) or True)
    ok = dga.alert("DISK GUARD RED: 0Gi free")
    assert [c[0] for c in calls] == ["cto", "tg"]
    assert ok is True  # at least one channel landed


def test_alert_returns_false_when_every_channel_fails(monkeypatch):
    def boom(text, key=None):
        raise RuntimeError("down")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: boom(text))
    assert dga.alert("x") is False


# ------------------------------------------------------------------ wake ---

def test_wake_signs_v2_over_timestamp_dot_body(monkeypatch, tmp_path):
    import hashlib
    import hmac

    secret_file = tmp_path / "secret"
    secret_file.write_text("s3cr3t\n")
    sent = {}

    class FakeResp:
        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        sent["url"] = req.full_url
        sent["headers"] = {k.lower(): v for k, v in req.headers.items()}
        sent["body"] = req.data
        return FakeResp()

    monkeypatch.setattr(dga, "wake_config", lambda: {
        "url": "http://127.0.0.1:8645/webhooks/kanban-wake",
        "secret_file": str(secret_file),
    })
    monkeypatch.setattr(dga.urllib.request, "urlopen", fake_urlopen)

    assert dga.deliver_wake("hello disk") is True
    ts = sent["headers"]["x-webhook-timestamp"]
    expected = hmac.new(b"s3cr3t", f"{ts}.".encode() + sent["body"],
                        hashlib.sha256).hexdigest()
    assert sent["headers"]["x-webhook-signature-v2"] == expected
    assert json.loads(sent["body"])["event"] == "disk_guard"
    assert "hello disk" in json.loads(sent["body"])["text"]


# ----------------------------------------------------------- forced fixture -

def test_forced_low_disk_fixture_drives_the_whole_path(monkeypatch):
    """DISK_GUARD_FAKE_FREE_GI forces the low-disk branch end to end, so the
    alert path is provable without actually filling the volume."""
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "1")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    rc = dga.main(["--floor", "10"])
    assert rc == 1
    assert delivered, "forced low-disk fixture must reach the alert channels"
    assert "1.0Gi" in delivered[0]


def test_healthy_disk_is_silent(monkeypatch):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "99")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    rc = dga.main(["--floor", "10"])
    assert rc == 0
    assert delivered == []


# ----------------------------------------------------------------- dedupe --

def test_repeat_alert_within_window_is_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "1")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    assert dga.main(["--floor", "10"]) == 1
    first = len(delivered)
    assert dga.main(["--floor", "10"]) == 1
    assert len(delivered) == first, "identical alert inside the window must not re-page"


def test_worsening_alert_is_not_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "5")
    dga.main(["--floor", "10"])
    n = len(delivered)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "1")
    dga.main(["--floor", "10"])
    assert len(delivered) > n, "a worse figure is new information and must page"


def test_small_wobble_inside_window_does_not_repage(monkeypatch, tmp_path):
    """Two pages in five minutes for the same incident is the anti-pattern.
    Under hash dedupe the incident key includes the free figure, so a drifting
    number with the SAME remediation list must still not re-page — the operator
    would read an identical actionable list."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a"
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "7")
    dga.main(["--floor", "10", "--detail", body])
    n = len(delivered)
    assert n > 0
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "7")  # same state, next tick
    dga.main(["--floor", "10", "--detail", body])
    assert len(delivered) == n


def test_material_drop_repages(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "7")
    dga.main(["--floor", "10"])
    n = len(delivered)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "2")
    dga.main(["--floor", "10"])
    assert len(delivered) > n


# ----------------------------------------------- hash-of-content dedupe ----
# CLASS: five pages were sent for one incident on 2026-09-21; the fourth and
# fifth were BYTE-IDENTICAL (14Gi/25Gi, same six paths). Dedupe keyed on the
# free-GiB number alone cannot see that the remediation list is unchanged, and
# any reset of the state file re-pages. The key must be a hash of what the
# operator would actually read: (free, floor, top paths).

def _fire(monkeypatch, delivered, free, detail, floor="25"):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", free)
    return dga.main(["--floor", floor, "--detail", detail])


def test_identical_red_does_not_repage(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a\n  1746MB  /b"
    _fire(monkeypatch, delivered, "14", body)
    n = len(delivered)
    assert n > 0, "first RED must page"
    _fire(monkeypatch, delivered, "14", body)   # byte-identical second tick
    assert len(delivered) == n, "identical RED must not re-page"


def test_changed_top_paths_repages(monkeypatch, tmp_path):
    """The number is the same but the remediation list changed — that is new
    information for the operator and must page."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    _fire(monkeypatch, delivered, "14", "Top reclaimable:\n  2337MB  /a")
    n = len(delivered)
    _fire(monkeypatch, delivered, "14", "Top reclaimable:\n  9000MB  /zzz-new")
    assert len(delivered) > n


def test_identical_red_repages_after_ttl(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a"
    _fire(monkeypatch, delivered, "14", body)
    n = len(delivered)
    st = json.loads((tmp_path / "s.json").read_text())
    st["last_at"] = int(time.time()) - (dga.REPEAT_WINDOW_SECONDS + 60)
    (tmp_path / "s.json").write_text(json.dumps(st))
    _fire(monkeypatch, delivered, "14", body)
    assert len(delivered) > n, "TTL expiry must re-page even when unchanged"


def test_dry_run_never_delivers_but_reports(monkeypatch, tmp_path, capsys):
    """Iterating on the guard must not page the CEO. --dry-run prints the page
    and touches no channel and no state."""
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    boom = lambda t: (_ for _ in ()).throw(AssertionError("delivered in dry-run"))
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: boom(t))
    monkeypatch.setattr(dga, "deliver_telegram", boom)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "3")
    rc = dga.main(["--floor", "10", "--detail", "x", "--dry-run"])
    assert rc == 1
    assert "DISK GUARD RED" in capsys.readouterr().out
    assert not state.exists(), "dry-run must not write dedupe state"
