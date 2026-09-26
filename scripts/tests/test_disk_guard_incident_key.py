"""Invariant tests for disk_guard_alert.incident_key.

CLASS under test — measured on 2026-09-21: the disk guard paged 17 times in one
day (last six at 07:58 08:13 08:29 08:46 09:03 09:19, every ~16 min) for ONE
unchanging state: 8Gi free, floor 10Gi, no owned reclaim. The 6h dedupe was
armed and never engaged, because incident_key() hashed the RENDERED detail text
and that text is not a function of the state. Two renderings alternated:

  (a) "No single reclaimable path over 100MB — ... Check state.db / postgres /
       purgeable."
  (b) "  1746MB  .../pnpm/store/v10\n  443MB  .../.copilot/session-state/<uuid>"

Reproduced cause (not the one originally assumed): (a) is NOT "everything is
under the 100MB threshold". Both paths are far over it on every tick. (a) is
produced when disk-guard.sh's lsof-based `path_in_use` filter transiently drops
those directories because some unrelated process holds a descriptor on them —
verified by opening fds on both dirs and watching `--top-reclaimable` return
empty. Absence of a ranking is absence of EVIDENCE, not evidence of change.

The bodies below are the verbatim renderings recorded in
~/.hermes/logs/disk-guard.log on 2026-09-21.

Run: python3 -m pytest ~/.hermes/scripts/tests/test_disk_guard_incident_key.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

SCRIPT = Path(os.path.expanduser("~/.hermes/scripts/disk_guard_alert.py"))
spec = importlib.util.spec_from_file_location("disk_guard_alert_key", SCRIPT)
dga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dga)

HOME = os.path.expanduser("~")

# --- verbatim from ~/.hermes/logs/disk-guard.log, 2026-09-21 ----------------
RENDER_A = (
    "No single reclaimable path over 100MB — the space is not in agent "
    "scratch. Check state.db / postgres / purgeable."
)
RENDER_B_443 = (
    "Top reclaimable, largest first (du-apparent; confirm with scratch-cost.sh "
    "before sizing capacity):\n"
    f"  1746MB  {HOME}/Library/pnpm/store/v10\n"
    f"  443MB  {HOME}/.copilot/session-state/6cd6710e-3a32-41ee-8f3b-25f25f3cb90d"
)
RENDER_B_440 = RENDER_B_443.replace("443MB", "440MB")
RENDER_B_428 = RENDER_B_443.replace("443MB", "428MB")
# Same incident, different copilot session uuid (the dir is recreated per session).
RENDER_B_OTHER_UUID = RENDER_B_443.replace(
    "6cd6710e-3a32-41ee-8f3b-25f25f3cb90d", "11112222-3333-4444-5555-666677778888"
)

FREE, FLOOR = 8, 10


def _chain(renderings, free=FREE, floor=FLOOR):
    """Walk ticks the way main() does, carrying state forward. Returns the keys."""
    state, keys = {}, []
    for detail in renderings:
        key, paths, gen = dga.resolve_incident(free, floor, detail, state)
        keys.append(key)
        state = {"last_free": free, "last_floor": floor, "last_key": key,
                 "last_paths": paths, "last_gen": gen}
    return keys


# ----------------------------------------------------- the reported defect --

def test_recorded_a_and_b_renderings_collapse_to_one_key():
    """The exact alternation that paged 17 times must be ONE incident."""
    keys = _chain([RENDER_B_443, RENDER_A, RENDER_B_443, RENDER_A])
    assert len(set(keys)) == 1, f"flapping renderings minted {len(set(keys))} keys: {keys}"


def test_copilot_dir_drifting_a_few_mb_does_not_change_the_key():
    """440 -> 443MB is a 0.7% move. Sizes change the message, not the key."""
    keys = _chain([RENDER_B_428, RENDER_B_440, RENDER_B_443])
    assert len(set(keys)) == 1, keys


def test_full_recorded_sequence_is_one_incident():
    keys = _chain([RENDER_B_428, RENDER_A, RENDER_B_440, RENDER_A,
                   RENDER_B_443, RENDER_B_443, RENDER_A])
    assert len(set(keys)) == 1, keys


def test_session_uuid_change_does_not_change_the_key():
    keys = _chain([RENDER_B_443, RENDER_B_OTHER_UUID])
    assert len(set(keys)) == 1, keys


# ------------------------------------------- genuinely new info still pages --

def test_a_genuinely_new_path_changes_the_key():
    new = RENDER_B_443 + f"\n  9000MB  {HOME}/workspace/brand-new-hog"
    keys = _chain([RENDER_B_443, new])
    assert keys[0] != keys[1], "a new ranked path is new information and must page"


def test_a_disappearing_path_changes_the_key():
    """Not the empty case — a non-empty list that lost a member is real news."""
    shrunk = (
        "Top reclaimable, largest first:\n"
        f"  1746MB  {HOME}/Library/pnpm/store/v10"
    )
    keys = _chain([RENDER_B_443, shrunk])
    assert keys[0] != keys[1]


def test_free_bucket_change_changes_the_key():
    a = dga.resolve_incident(8, FLOOR, RENDER_B_443, {})[0]
    b = dga.resolve_incident(5, FLOOR, RENDER_B_443, {})[0]
    assert a != b


def test_floor_change_changes_the_key():
    a = dga.resolve_incident(FREE, 10, RENDER_B_443, {})[0]
    b = dga.resolve_incident(FREE, 25, RENDER_B_443, {})[0]
    assert a != b


def test_material_size_growth_over_25pct_changes_the_key():
    big = RENDER_B_443.replace("443MB", "900MB")   # +103%
    keys = _chain([RENDER_B_443, big])
    assert keys[0] != keys[1], "a path more than doubling is new information"


def test_size_change_just_under_the_band_does_not_change_the_key():
    near = RENDER_B_443.replace("443MB", "540MB")  # +21.9%, under 25%
    keys = _chain([RENDER_B_443, near])
    assert keys[0] == keys[1]


def test_first_red_of_a_fresh_incident_pages():
    """Empty prior state must not inherit anything — there is nothing to inherit."""
    key, paths, gen = dga.resolve_incident(FREE, FLOOR, RENDER_A, {})
    assert key and gen == 0 and paths == {}


def test_empty_ranking_under_a_different_free_figure_is_a_new_incident():
    """The inherit rule is scoped to the SAME state; it must not swallow a
    genuinely different incident that happens to have no ranking."""
    k1, p1, g1 = dga.resolve_incident(8, FLOOR, RENDER_B_443, {})
    st = {"last_free": 8, "last_floor": FLOOR, "last_key": k1,
          "last_paths": p1, "last_gen": g1}
    k2 = dga.resolve_incident(2, FLOOR, RENDER_A, st)[0]
    assert k2 != k1


# --------------------------------------------------------- normalization ----

def test_normalize_folds_instance_leaves_and_home():
    assert dga.normalize_path(f"{HOME}/.copilot/session-state/6cd6710e-3a32-41ee-8f3b-25f25f3cb90d") \
        == "~/.copilot/session-state"
    assert dga.normalize_path(f"{HOME}/workspace/AgentPod-worktrees/t_08a9fbe4") \
        == "~/workspace/AgentPod-worktrees"
    # A real place with a wordy leaf is NOT folded away.
    assert dga.normalize_path(f"{HOME}/Library/pnpm/store/v10") \
        == "~/Library/pnpm/store/v10"


def test_parse_paths_is_derived_from_the_text_not_a_known_list():
    """A root nobody enumerated here must still be parsed — the guard may not
    depend on a hand-maintained inventory of expected paths."""
    got = dga.parse_paths("  4242MB  /var/never/seen/before/thing")
    assert got == {"/var/never/seen/before/thing": 4242}


# ------------------------------------------- end-to-end through main() ------

def test_two_ticks_on_unchanged_state_page_exactly_once(monkeypatch, tmp_path):
    """The acceptance criterion, driven through main() and the real state file."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    delivered = []
    # deliver_wake is the RETIRED EM route (card t_b8d0aaeb). Patching it here
    # made these tests assert against a channel alert() no longer calls, so a
    # real one-page-two-channels regression would have read as 1 == 2.
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    for detail in (RENDER_B_443, RENDER_A, RENDER_B_440, RENDER_A, RENDER_B_443):
        assert dga.main(["--floor", "10", "--detail", detail]) == 1
    assert len(delivered) == 2, (
        f"one incident, five ticks -> expected 1 page on 2 channels, got {len(delivered)}")


def test_incident_beginning_on_an_empty_ranking_pages_exactly_once(
        monkeypatch, tmp_path):
    """Reviewer-reported leak: when the incident's FIRST tick is the lsof-
    suppressed rendering (a), the inherited key must survive the tick that
    regains the ranking. Previously tick 2 recomputed a fresh hash and paged a
    second time for the same state."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    delivered = []
    # deliver_wake is the RETIRED EM route (card t_b8d0aaeb). Patching it here
    # made these tests assert against a channel alert() no longer calls, so a
    # real one-page-two-channels regression would have read as 1 == 2.
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    for detail in (RENDER_A, RENDER_B_443, RENDER_B_443, RENDER_B_440, RENDER_A):
        assert dga.main(["--floor", "10", "--detail", detail]) == 1
    assert len(delivered) == 2, (
        f"A,B,B,B,A on unchanged state -> expected 1 page on 2 channels, "
        f"got {len(delivered)}")


def test_empty_ranking_first_then_ranking_is_one_key():
    """Same leak at the resolve_incident level: A,B,B must be one key."""
    keys = _chain([RENDER_A, RENDER_B_443, RENDER_B_443])
    assert len(set(keys)) == 1, keys


def test_new_path_after_an_empty_ranking_tick_still_pages():
    """The inherit must not swallow real news: identity regained as
    {pnpm, copilot}, then a genuinely new root appears -> different key."""
    new = RENDER_B_443 + f"\n  9000MB  {HOME}/workspace/brand-new-hog"
    keys = _chain([RENDER_A, RENDER_B_443, new])
    assert keys[0] == keys[1]
    assert keys[2] != keys[1], "a new ranked path is new information and must page"


def test_ttl_still_repages_the_same_incident(monkeypatch, tmp_path):
    """Dedupe must not become silence: after 6h an unchanged RED pages again."""
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    delivered = []
    # deliver_wake is the RETIRED EM route (card t_b8d0aaeb). Patching it here
    # made these tests assert against a channel alert() no longer calls, so a
    # real one-page-two-channels regression would have read as 1 == 2.
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    dga.main(["--floor", "10", "--detail", RENDER_B_443])
    n = len(delivered)
    st = json.loads(state.read_text())
    st["last_at"] = int(time.time()) - (dga.REPEAT_WINDOW_SECONDS + 60)
    state.write_text(json.dumps(st))
    dga.main(["--floor", "10", "--detail", RENDER_A])
    assert len(delivered) > n


def test_suppressed_tick_does_not_extend_the_ttl(monkeypatch, tmp_path):
    """A suppressed tick refreshing last_at would push the 6h re-page out
    forever — the dedupe would become permanent silence."""
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    # deliver_kanban_cto must be patched like every other channel. Leaving it
    # unpatched here is what created real CTO cards t_849189ed / t_0fbf5b8d on
    # 2026-09-25 at 19:04 from this test's literals (8Gi / floor 10Gi) while
    # the host had 23Gi free. disk_guard_alert now refuses to page from a
    # fixture measurement, so this is defence in depth, not the fix.
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: True)
    monkeypatch.setattr(dga, "deliver_wake", lambda t: True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: True)
    dga.main(["--floor", "10", "--detail", RENDER_A])
    first_at = json.loads(state.read_text())["last_at"]
    time.sleep(1.1)
    dga.main(["--floor", "10", "--detail", RENDER_B_443])   # suppressed
    assert json.loads(state.read_text())["last_at"] == first_at
