"""Regressions for the ESTOP pause repair (PR #121054): F-001..F-004 + the TTL overflow.

Every test in this file fails on the pre-fix head and passes after the fix; the negative
control (pre/post run, same interpreter, same file) is recorded on the card.

F-001  the deadman retires ONLY the generation it examined. A successor that re-arms in
       the interval between the expiry DECISION and the removal survives, with
       ``is_engaged()`` still True. Pre-fix the removal re-derived the key from the file
       at removal time — the file compared with ITSELF — so the successor was deleted and
       the pause silently vanished.
F-002  ``is_allowed`` composes over EVERY active hold (deny by intersection). Pre-fix it
       read only the FIRST parsable sentinel, so a permissive profile-local hold weakened a
       stricter fleet-root one.
F-003  ``profiles`` NARROWS the identity grant and never grants alone: a routing
       coordinate must not become an authority.
F-004  an OMITTED allowlist/ttl on a re-arm PRESERVES the standing sentinel's, so an
       in-band ``/pause <reason>`` or a no-flag ``hermes pause`` cannot strip authority.
P2     ``engage()`` never raises on any ttl: an out-of-range one arms WITHOUT a deadman,
       and the CLI says so LOUDLY instead of parking the fleet with no auto-resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from agent import estop

OPERATOR = "operator-uid-7"
PEER = "bot-peer-1"


def _stamp(delta_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)).isoformat()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reset estop module log state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    estop._logged_components.clear()
    estop._expired_logged.clear()
    return tmp_path


def _profile_env(tmp_path, monkeypatch, lane="lane-a"):
    """HERMES_HOME = <root>/profiles/<lane>, so BOTH candidate sentinel paths exist."""
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / lane
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    estop._logged_components.clear()
    estop._expired_logged.clear()
    return root, profile


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- F-001 (deadman)


def test_expiry_spares_a_successor_that_re_arms_after_the_decision(hermes_home, monkeypatch):
    """F-001: the successor lands AFTER the expiry decision, BEFORE the claim.

    The re-arm is injected from inside ``_read_payload`` — the same read that produced the
    decision — so the file on disk is the successor by the time the removal runs. Pre-fix
    the removal then measured the file against ITSELF and deleted the re-arm.
    """
    path = hermes_home / "ESTOP"
    _write(path, {"generation": "a" * 32, "reason": "dead window", "expires_at": _stamp(-60)})

    real_read = estop._read_payload
    fired: list = []

    def _read_then_rearm(p):
        payload = real_read(p)
        if p == path and not fired:
            fired.append(True)
            estop.engage(reason="successor", ttl="1h")  # lands after the expiry decision
        return payload

    monkeypatch.setattr(estop, "_read_payload", _read_then_rearm)

    assert estop.is_engaged() is True, "the successor re-arm must survive the retirement"
    assert fired == [True]
    assert json.loads(path.read_text(encoding="utf-8"))["reason"] == "successor"


def test_expiry_spares_a_successor_that_re_arms_in_the_unlink_interval(hermes_home, monkeypatch):
    """F-001: the second interleaving — a re-arm landing INSIDE the removal's unlink.

    Pre-fix the removal unlinks the SENTINEL path, so a successor sitting there is deleted
    with it. The fix removes a CLAIM (a rename-aside sibling), leaving the sentinel path
    free for the successor the re-arm just wrote.
    """
    path = hermes_home / "ESTOP"
    _write(path, {"generation": "a" * 32, "reason": "dead window", "expires_at": _stamp(-60)})

    real_unlink = pathlib.Path.unlink
    fired: list = []

    def _unlink_with_rearm(self, *args, **kwargs):
        if "ESTOP" in self.name and not fired:
            fired.append(True)
            estop.engage(reason="successor", ttl="1h")  # lands inside the unlink interval
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", _unlink_with_rearm)

    assert estop.is_engaged() is True, "the successor must not be unlinked with the dead one"
    assert fired == [True]
    assert json.loads(path.read_text(encoding="utf-8"))["reason"] == "successor"


def test_expiry_spares_a_successor_with_an_identical_file_key(hermes_home, monkeypatch):
    """F-001: identity is the payload TOKEN, not ``mtime_ns:size``.

    The successor is the same byte LENGTH and carries the same ``mtime_ns`` as the dead
    generation, so only the ``generation`` token tells them apart.
    """
    path = hermes_home / "ESTOP"
    dead = {
        "generation": "a" * 32,
        "engaged_at": _stamp(-120),
        "reason": "window",
        "expires_at": _stamp(-60),
    }
    successor = {**dead, "generation": "b" * 32, "expires_at": _stamp(3600)}
    dead_bytes = (json.dumps(dead, indent=2) + "\n").encode("utf-8")
    successor_bytes = (json.dumps(successor, indent=2) + "\n").encode("utf-8")
    assert len(dead_bytes) == len(successor_bytes), "test premise: identical lengths"

    path.write_bytes(dead_bytes)
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(stamp, stamp))

    real_read = estop._read_payload
    fired: list = []

    def _read_then_rearm(p):
        payload = real_read(p)
        if p == path and not fired:
            fired.append(True)
            path.write_bytes(successor_bytes)  # different token, same length
            os.utime(path, ns=(stamp, stamp))  # ... and the SAME mtime_ns
        return payload

    monkeypatch.setattr(estop, "_read_payload", _read_then_rearm)

    assert estop.is_engaged() is True, "an identical file key must not condemn the successor"
    assert fired == [True]
    assert json.loads(path.read_text(encoding="utf-8"))["generation"] == "b" * 32


# ---------------------------------------------------------- F-002 (deny by intersection)


def test_is_allowed_composes_over_every_active_hold_in_both_orders(tmp_path, monkeypatch):
    """F-002: deny wins — a permissive hold must never weaken a stricter one.

    Pre-fix ``is_allowed`` read only the FIRST parsable sentinel, so the permissive hold on
    the first candidate path admitted an identity the stricter hold refused.
    """
    root, profile = _profile_env(tmp_path, monkeypatch)
    permissive = {"generation": "p" * 32, "reason": "permissive", "allow": {"user_ids": [OPERATOR]}}
    strict = {"generation": "s" * 32, "reason": "strict", "allow": {"user_ids": [PEER]}}

    _write(profile / "ESTOP", permissive)  # order 1: permissive on the first candidate path
    _write(root / "ESTOP", strict)
    assert estop.is_allowed(OPERATOR, "lane-a") is False, "the strict root hold must veto"
    assert estop.is_allowed(PEER, "lane-a") is False, "the strict hold must not veto itself in"

    _write(profile / "ESTOP", strict)  # order 2: strict first, permissive on the root
    _write(root / "ESTOP", permissive)
    assert estop.is_allowed(OPERATOR, "lane-a") is False, "no hold may grant what another denies"
    assert estop.is_allowed(PEER, "lane-a") is False, "no hold may grant what another denies"


def test_is_allowed_denies_by_intersection_when_a_hold_is_corrupt(tmp_path, monkeypatch):
    """F-002: an unreadable hold admits nobody — it cannot be skipped as if it were absent."""
    root, profile = _profile_env(tmp_path, monkeypatch)
    _write(profile / "ESTOP", {"generation": "p" * 32, "allow": {"user_ids": [OPERATOR]}})
    (root / "ESTOP").write_text("{not json", encoding="utf-8")

    assert estop.is_allowed(OPERATOR, "lane-a") is False


def test_is_allowed_ignores_a_dead_hold_and_keeps_a_fresh_one(tmp_path, monkeypatch):
    """F-002: only ACTIVE holds compose — a dead deadman neither holds nor grants."""
    root, profile = _profile_env(tmp_path, monkeypatch)
    _write(profile / "ESTOP", {
        "generation": "p" * 32, "reason": "dead", "expires_at": _stamp(-60),
        "allow": {"user_ids": [OPERATOR]}})
    _write(root / "ESTOP", {
        "generation": "r" * 32, "reason": "fresh", "expires_at": _stamp(600),
        "allow": {"user_ids": [PEER]}})

    assert estop.is_allowed(OPERATOR, "lane-a") is False, "the fresh root hold governs"
    assert estop.is_allowed(PEER, "lane-a") is True, "the dead hold is gone, not a veto"


# -------------------------------------------------------------- F-003 (profiles narrow)


def test_profiles_never_grant_alone_and_narrow_the_identity(hermes_home):
    """F-003: a serving profile is a routing coordinate, not an authority."""
    estop.engage(allow={"profiles": ["primary-lane"]})
    assert estop.is_allowed(None, "primary-lane") is False, "a profile alone admits nobody"
    assert estop.is_allowed(OPERATOR, "primary-lane") is False, "a profile alone admits nobody"

    estop.disengage()
    estop.engage(allow={"user_ids": [OPERATOR], "profiles": ["primary-lane"]})
    assert estop.is_allowed(OPERATOR, "primary-lane") is True
    assert estop.is_allowed(OPERATOR, "other-lane") is False, "the wrong profile is held"
    assert estop.is_allowed(OPERATOR) is False, "profiles present: no profile, no grant"
    assert estop.is_allowed("someone-else", "primary-lane") is False, "the id gate still applies"


def test_cli_pause_refuses_allow_profile_without_allow_user(hermes_home, capsys):
    """F-003: the CLI cannot arm a pause whose allowlist grants nobody."""
    from hermes_cli.subcommands.pause import cmd_pause

    rc = cmd_pause(argparse.Namespace(
        reason=None, allow_user=None, allow_profile=["primary-lane"], ttl=None))

    assert rc == 2
    assert estop.is_engaged() is False, "a refused pause must not half-arm"
    assert "--allow-user" in capsys.readouterr().out


# ------------------------------------------------------------- F-004 (re-arm preserves)


def test_reengage_preserves_a_standing_allowlist_and_deadman(hermes_home):
    """F-004: a re-arm that does not SUPPLY a field must not strip it."""
    estop.engage(
        reason="first", allow={"user_ids": [OPERATOR], "profiles": ["primary-lane"]}, ttl="30m")
    before = estop.get_state()
    assert before["expires_at"]

    estop.engage(reason="re-arm")  # nothing supplied
    after = estop.get_state()
    assert after["reason"] == "re-arm"
    assert after["allow"] == before["allow"], "an omitted allowlist must be KEPT"
    assert after["expires_at"] == before["expires_at"], "an omitted ttl must KEEP the deadman"
    assert estop.is_allowed(OPERATOR, "primary-lane") is True

    estop.engage(reason="replace", allow={"user_ids": [PEER]}, ttl="5m")  # explicit replaces
    replaced = estop.get_state()
    assert replaced["allow"] == {"user_ids": [PEER]}
    assert replaced["expires_at"] != before["expires_at"]
    assert estop.is_allowed(OPERATOR, "primary-lane") is False


def test_cli_no_flag_rearm_preserves_the_standing_sentinel(hermes_home, capsys):
    """F-004: `hermes pause` with no flags must not strip what the first arm set."""
    from hermes_cli.subcommands.pause import cmd_pause

    assert cmd_pause(argparse.Namespace(
        reason="window", allow_user=[OPERATOR], allow_profile=["primary-lane"], ttl="30m")) == 0
    before = estop.get_state()

    capsys.readouterr()
    assert cmd_pause(argparse.Namespace(
        reason=None, allow_user=None, allow_profile=None, ttl=None)) == 0

    after = estop.get_state()
    assert after["allow"] == before["allow"]
    assert after["expires_at"] == before["expires_at"]
    assert estop.is_allowed(OPERATOR, "primary-lane") is True
    assert "allowlist" in capsys.readouterr().out


def test_inband_pause_rearm_preserves_the_standing_sentinel(hermes_home):
    """F-004: the same for the real in-band `/pause <reason>` path."""
    from gateway.run import GatewayRunner

    estop.engage(reason="window", allow={"user_ids": [OPERATOR]}, ttl="30m")
    before = estop.get_state()

    class _Source:
        chat_id = "c1"
        user_id = OPERATOR
        profile = "primary-lane"

    class _Event:
        source = _Source()

        def get_command_args(self):
            return "second window"

    reply = asyncio.run(object.__new__(GatewayRunner)._handle_pause_command(_Event()))

    assert "second window" in reply
    after = estop.get_state()
    assert after["reason"] == "second window"
    assert after["allow"] == before["allow"], "the in-band re-arm must KEEP the allowlist"
    assert after["expires_at"] == before["expires_at"], "the in-band re-arm must KEEP the deadman"


# --------------------------------------------------------------- P2 (ttl must not raise)


def test_engage_never_raises_on_an_out_of_range_ttl(hermes_home):
    """P2: `engage()` must never raise on any ttl, and never arm an absurd deadman."""
    for ttl in ("999999999d", "999999999999999999999d", 10 ** 15, "1h"):
        estop.disengage()
        estop.engage(reason="window", ttl=ttl)  # must not raise
        assert estop.is_engaged() is True
        expires = estop.get_state().get("expires_at")
        if ttl == "1h":
            assert expires, "a usable ttl must still arm its deadman"
        else:
            assert expires is None, "an out-of-range ttl must arm WITHOUT a deadman"


def test_cli_out_of_range_ttl_arms_without_a_deadman_and_says_so_loudly(hermes_home, capsys):
    """P2: the CLI must report that the ttl was not applied, instead of a false 'deadman'."""
    from hermes_cli.subcommands.pause import cmd_pause

    rc = cmd_pause(argparse.Namespace(
        reason=None, allow_user=None, allow_profile=None, ttl="999999999d"))

    assert rc == 0
    assert estop.is_engaged() is True
    state = estop.get_state()
    assert state is not None and not state.get("expires_at"), "no absurd/stale deadman"
    out = capsys.readouterr().out
    assert "NOT applied" in out and "auto-resume" in out


def test_cli_still_reports_a_real_deadman(hermes_home, capsys):
    """P2 guard: a WORKING ttl must keep its normal 'deadman' line, with no false warning."""
    from hermes_cli.subcommands.pause import cmd_pause

    assert cmd_pause(argparse.Namespace(
        reason=None, allow_user=None, allow_profile=None, ttl="45m")) == 0

    out = capsys.readouterr().out
    assert "deadman" in out and "NOT applied" not in out
    assert estop.get_state().get("expires_at")
