"""Tests for the mem0 destructive tools — mem0_forget / mem0_delete.

Covers the spec invariants C1–C9:
  ~/.hermes/plans/2026-06-10_mem0-destructive-tools-spec.md

Uses a FakeClient that emulates the hosted mem0 SDK surface we depend on
(get/get_all/search/update/delete) with an in-memory store, so the full
handler logic (gate, ledger, mint-store, mass-floor, caps, velocity, dry-run/
token, recall-hide, restore) is exercised without network.
"""

import json
import os
import stat
import time

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider, _FORGOTTEN_PREFIX


# ---------------------------------------------------------------------------
# Fake hosted-mem0 client
# ---------------------------------------------------------------------------

class MemoryNotFound(Exception):
    pass


class FakeClient:
    """In-memory emulation of the mem0 MemoryClient surface used by the plugin."""

    def __init__(self):
        self._store = {}      # id -> {"id","memory","metadata","user_id","agent_id"}
        self._counter = 0
        self.delete_calls = []

    def _seed(self, text, user_id="ace", agent_id="apollo", metadata=None):
        self._counter += 1
        mid = f"mem-{self._counter:04d}"
        self._store[mid] = {"id": mid, "memory": text, "metadata": metadata or {},
                            "user_id": user_id, "agent_id": agent_id}
        return mid

    # --- SDK surface ---
    def get(self, memory_id):
        if memory_id not in self._store:
            raise MemoryNotFound(json.dumps({"error": "Memory not found!"}))
        return dict(self._store[memory_id])

    def get_all(self, filters=None, **kwargs):
        uid = (filters or {}).get("user_id")
        items = [dict(m) for m in self._store.values()
                 if uid is None or m["user_id"] == uid]
        return {"results": items}

    def search(self, query=None, filters=None, rerank=False, top_k=10, **kwargs):
        uid = (filters or {}).get("user_id")
        items = [dict(m) for m in self._store.values()
                 if (uid is None or m["user_id"] == uid)
                 and (not query or query.lower() in m["memory"].lower())]
        return {"results": items[:top_k]}

    def update(self, memory_id, text=None, metadata=None, timestamp=None, **kwargs):
        if memory_id not in self._store:
            raise MemoryNotFound(json.dumps({"error": "Memory not found!"}))
        if text is not None:
            self._store[memory_id]["memory"] = text
        if metadata is not None:
            self._store[memory_id]["metadata"] = metadata
        return {"id": memory_id}

    def delete(self, memory_id, delete_linked=False):
        self.delete_calls.append((memory_id, delete_linked))
        if memory_id not in self._store:
            raise MemoryNotFound(json.dumps({"error": "Memory not found!"}))
        del self._store[memory_id]
        return {"id": memory_id}

    def history(self, memory_id):
        return []


# ---------------------------------------------------------------------------
# Provider fixture (gated ON, isolated HERMES_HOME)
# ---------------------------------------------------------------------------

def _provider(monkeypatch, tmp_path, client, *, enabled=True, agent_id="apollo", cfg=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setenv("MEM0_DESTRUCTIVE_TOOLS", "true" if enabled else "false")
    # speed up: tiny config overrides if given
    if cfg:
        (tmp_path / "mem0.json").write_text(json.dumps(cfg))
    p = Mem0MemoryProvider()
    p.initialize("test-session")
    p._user_id = "ace"
    p._agent_id = agent_id
    monkeypatch.setattr(p, "_get_client", lambda: client)
    monkeypatch.setattr(p, "_hermes_home", lambda: tmp_path)
    return p


def _call(p, tool, args):
    return json.loads(p.handle_tool_call(tool, args))


# ---------------------------------------------------------------------------
# C1 — gate (fail-closed)
# ---------------------------------------------------------------------------

class TestGate:
    def test_gate_off_three_schemas(self, monkeypatch, tmp_path):
        p = _provider(monkeypatch, tmp_path, FakeClient(), enabled=False)
        names = [s["name"] for s in p.get_tool_schemas()]
        assert names == ["mem0_profile", "mem0_search", "mem0_conclude"]

    def test_gate_on_five_schemas(self, monkeypatch, tmp_path):
        p = _provider(monkeypatch, tmp_path, FakeClient(), enabled=True)
        names = [s["name"] for s in p.get_tool_schemas()]
        assert names == ["mem0_profile", "mem0_search", "mem0_conclude",
                         "mem0_forget", "mem0_delete"]

    def test_gate_off_call_is_unknown_tool(self, monkeypatch, tmp_path):
        p = _provider(monkeypatch, tmp_path, FakeClient(), enabled=False)
        out = p.handle_tool_call("mem0_delete", {"memory_id": "x"})
        assert "Unknown tool" in out

    def test_profile_config_without_flag_stays_recall_only(self, monkeypatch, tmp_path):
        # Simulate a Daedalus/Athena profile: mem0.json without the flag, no env.
        monkeypatch.delenv("MEM0_DESTRUCTIVE_TOOLS", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "k")
        (tmp_path / "mem0.json").write_text(json.dumps({"user_id": "ace"}))
        p = Mem0MemoryProvider()
        p.initialize("s")
        assert p._destructive_enabled is False
        assert len(p.get_tool_schemas()) == 3


# ---------------------------------------------------------------------------
# C2 — read-before-destroy (by-id)
# ---------------------------------------------------------------------------

class TestByIdDelete:
    def test_delete_reads_before_and_returns_text(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("the sky is plaid")
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_delete", {"memory_id": mid})
        assert out["deleted"] == 1
        assert out["results"][0]["was"] == "the sky is plaid"
        assert mid not in c._store

    def test_delete_bogus_id_not_found(self, monkeypatch, tmp_path):
        c = FakeClient()
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_delete", {"memory_id": "nope"})
        assert out["deleted"] == 0
        assert out["results"][0]["outcome"] == "not_found"

    def test_delete_linked_passed_through(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("x")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_delete", {"memory_id": mid, "delete_linked": True})
        assert c.delete_calls == [(mid, True)]


# ---------------------------------------------------------------------------
# C4 — ledger (append-only, 0o600, write-before-act)
# ---------------------------------------------------------------------------

class TestLedger:
    def test_delete_writes_pending_then_ok(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("doomed")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_delete", {"memory_id": mid})
        ledger = (tmp_path / "mem0-destructive-ledger.jsonl").read_text().strip().splitlines()
        recs = [json.loads(l) for l in ledger]
        outcomes = [r["outcome"] for r in recs]
        assert "pending" in outcomes and "ok" in outcomes
        pend = [r for r in recs if r["outcome"] == "pending"][0]
        assert pend["was"] == "doomed"
        assert pend["agent_id"] == "apollo"

    def test_ledger_file_is_0600(self, monkeypatch, tmp_path):
        if os.name == "nt":
            pytest.skip("POSIX mode bits")
        c = FakeClient()
        mid = c._seed("x")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_delete", {"memory_id": mid})
        mode = stat.S_IMODE((tmp_path / "mem0-destructive-ledger.jsonl").stat().st_mode)
        assert mode == 0o600

    def test_ledger_append_only_no_line_rewrite(self, monkeypatch, tmp_path):
        c = FakeClient()
        m1, m2 = c._seed("a"), c._seed("b")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_delete", {"memory_id": m1})
        after_first = (tmp_path / "mem0-destructive-ledger.jsonl").read_text()
        _call(p, "mem0_delete", {"memory_id": m2})
        after_second = (tmp_path / "mem0-destructive-ledger.jsonl").read_text()
        # earlier bytes are a strict prefix of the later file (append-only)
        assert after_second.startswith(after_first)


# ---------------------------------------------------------------------------
# C5 — forget is reversible (tombstone + restore)
# ---------------------------------------------------------------------------

class TestForgetRestore:
    def test_forget_tombstones(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("old fact")
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_forget", {"memory_id": mid, "reason": "superseded"})
        assert out["forgotten"] == 1
        stored = c._store[mid]
        assert stored["memory"].startswith(_FORGOTTEN_PREFIX)
        assert stored["metadata"]["forgotten"] is True
        assert stored["metadata"]["original_text"] == "old fact"
        assert stored["metadata"]["forgotten_by"] == "apollo"

    def test_restore_brings_back_original(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("precious original")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_forget", {"memory_id": mid})
        out = _call(p, "mem0_forget", {"memory_id": mid, "restore": True})
        assert out["result"] == "restored"
        assert c._store[mid]["memory"] == "precious original"
        assert c._store[mid]["metadata"]["forgotten"] is False

    def test_restore_of_never_forgotten_is_noop(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("live memory")
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_forget", {"memory_id": mid, "restore": True})
        assert "no-op" in out["result"]

    def test_forget_reason_never_echoes_newlines(self, monkeypatch, tmp_path):
        c = FakeClient()
        mid = c._seed("x")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_forget", {"memory_id": mid, "reason": "line1\nline2"})
        assert "\n" not in c._store[mid]["memory"]


# ---------------------------------------------------------------------------
# C6/C9 — recall hides forgotten memories (single choke point)
# ---------------------------------------------------------------------------

class TestRecallHide:
    def test_forgotten_absent_from_search_and_profile(self, monkeypatch, tmp_path):
        c = FakeClient()
        keep = c._seed("keep me visible")
        gone = c._seed("forget me")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_forget", {"memory_id": gone})

        prof = _call(p, "mem0_profile", {})
        assert "keep me visible" in prof["result"]
        assert _FORGOTTEN_PREFIX not in prof["result"]
        assert "forget me" not in prof["result"]

        srch = _call(p, "mem0_search", {"query": "me"})
        memories = [r["memory"] for r in srch.get("results", [])]
        assert any("keep me visible" in m for m in memories)
        assert all(not m.startswith(_FORGOTTEN_PREFIX) for m in memories)

    def test_raw_get_all_still_has_forgotten(self, monkeypatch, tmp_path):
        # The tombstone row physically survives (forget != delete).
        c = FakeClient()
        gone = c._seed("forget me")
        p = _provider(monkeypatch, tmp_path, c)
        _call(p, "mem0_forget", {"memory_id": gone})
        raw = c.get_all(filters={"user_id": "ace"})["results"]
        assert any(m["id"] == gone for m in raw)

    def test_drop_forgotten_helper(self):
        live = {"id": "1", "memory": "hi", "metadata": {}}
        meta_tomb = {"id": "2", "memory": "x", "metadata": {"forgotten": True}}
        text_tomb = {"id": "3", "memory": f"{_FORGOTTEN_PREFIX} gone", "metadata": {}}
        out = Mem0MemoryProvider._drop_forgotten([live, meta_tomb, text_tomb])
        assert out == [live]


# ---------------------------------------------------------------------------
# C3 — by-filter dry-run + confirm-token
# ---------------------------------------------------------------------------

class TestBulkToken:
    def _seed_many(self, c, n, prefix="fact"):
        return [c._seed(f"{prefix} {i}") for i in range(n)]

    @staticmethod
    def _pad(c, n=60):
        """Seed unrelated memories so a small filter match is a minority of the
        store (real store ~321) — otherwise the dry-run hits the C8a mass floor."""
        for i in range(n):
            c._seed(f"unrelated padding row {i}")

    def test_dryrun_returns_token_deletes_nothing(self, monkeypatch, tmp_path):
        c = FakeClient()
        self._seed_many(c, 5, "alpha")
        self._pad(c)
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_delete", {"filter": "alpha"})
        assert out["dry_run"] is True
        assert out["count"] == 5
        assert out["confirm_token"]
        assert len(c._store) == 65  # nothing deleted

    def test_execute_with_token_deletes_set(self, monkeypatch, tmp_path):
        c = FakeClient()
        self._seed_many(c, 5, "beta")
        c._seed("unrelated gamma")
        self._pad(c)
        p = _provider(monkeypatch, tmp_path, c)
        dry = _call(p, "mem0_delete", {"filter": "beta"})
        out = _call(p, "mem0_delete", {"filter": "beta", "confirm_token": dry["confirm_token"]})
        assert out["deleted"] == 5
        assert not any("beta" in m["memory"] for m in c._store.values())

    def test_fabricated_token_refused(self, monkeypatch, tmp_path):
        c = FakeClient()
        self._seed_many(c, 3, "delta")
        self._pad(c)
        p = _provider(monkeypatch, tmp_path, c)
        out = p.handle_tool_call("mem0_delete", {"filter": "delta", "confirm_token": "made-up-token"})
        assert "no valid dry-run token" in out
        assert sum(1 for m in c._store.values() if "delta" in m["memory"]) == 3

    def test_token_single_use_replay_refused(self, monkeypatch, tmp_path):
        c = FakeClient()
        self._seed_many(c, 3, "epsilon")
        self._pad(c)
        p = _provider(monkeypatch, tmp_path, c)
        dry = _call(p, "mem0_delete", {"filter": "epsilon"})
        tok = dry["confirm_token"]
        _call(p, "mem0_delete", {"filter": "epsilon", "confirm_token": tok})
        # replay (store now empty for epsilon; set-hash differs AND consumed)
        out2 = p.handle_tool_call("mem0_delete", {"filter": "epsilon", "confirm_token": tok})
        assert "token" in out2.lower()  # refused

    def test_expired_token_refused(self, monkeypatch, tmp_path):
        c = FakeClient()
        self._seed_many(c, 3, "zeta")
        self._pad(c)
        p = _provider(monkeypatch, tmp_path, c, cfg={"token_ttl_seconds": 1})
        dry = _call(p, "mem0_delete", {"filter": "zeta"})
        time.sleep(1.1)
        out = p.handle_tool_call("mem0_delete", {"filter": "zeta", "confirm_token": dry["confirm_token"]})
        assert "expired" in out.lower()

    def test_toctou_set_change_refused(self, monkeypatch, tmp_path):
        c = FakeClient()
        ids = self._seed_many(c, 4, "eta")
        for i in range(50):
            c._seed(f"pad {i}")  # big denominator so 4/54 isn't a mass op
        p = _provider(monkeypatch, tmp_path, c)
        dry = _call(p, "mem0_delete", {"filter": "eta"})
        # set changes between dry-run and execute (add another match)
        c._seed("eta extra")
        out = p.handle_tool_call("mem0_delete", {"filter": "eta", "confirm_token": dry["confirm_token"]})
        assert "TOCTOU" in out or "set changed" in out


# ---------------------------------------------------------------------------
# C8 — catastrophe floor + caps + velocity
# ---------------------------------------------------------------------------

class TestCatastropheFloor:
    def test_mass_ratio_refused_no_token(self, monkeypatch, tmp_path):
        c = FakeClient()
        for i in range(10):
            c._seed(f"common term {i}")  # filter 'common' matches all 10/10 = 100%
        p = _provider(monkeypatch, tmp_path, c)
        out = p.handle_tool_call("mem0_delete", {"filter": "common"})
        assert "mass" in out.lower() or "refused" in out.lower()
        assert "confirm_token" not in out  # no token offered

    def test_mass_ratio_refused_for_forget_too(self, monkeypatch, tmp_path):
        c = FakeClient()
        for i in range(10):
            c._seed(f"blanket {i}")
        p = _provider(monkeypatch, tmp_path, c)
        out = p.handle_tool_call("mem0_forget", {"filter": "blanket"})
        assert "refused" in out.lower() or "mass" in out.lower()

    def test_empty_filter_is_mass(self, monkeypatch, tmp_path):
        c = FakeClient()
        for i in range(10):
            c._seed(f"thing {i}")
        p = _provider(monkeypatch, tmp_path, c)
        out = p.handle_tool_call("mem0_delete", {"filter": ""})
        assert "refused" in out.lower() or "mass" in out.lower()

    def test_absolute_mass_floor(self, monkeypatch, tmp_path):
        c = FakeClient()
        # 60 matches out of 300 total (20%) — under ratio but over a low floor
        for i in range(60):
            c._seed(f"match {i}")
        for i in range(240):
            c._seed(f"other {i}")
        p = _provider(monkeypatch, tmp_path, c, cfg={"absolute_mass_floor": 50})
        out = p.handle_tool_call("mem0_delete", {"filter": "match"})
        assert "mass floor" in out.lower() or "refused" in out.lower()

    def test_hard_delete_ceiling_force_cannot_breach(self, monkeypatch, tmp_path):
        c = FakeClient()
        # 30 matches; ceiling 10 via cfg; force still can't exceed ceiling
        for i in range(30):
            c._seed(f"target {i}")
        for i in range(300):
            c._seed(f"pad {i}")  # big denominator so 30/330 isn't a mass op
        p = _provider(monkeypatch, tmp_path, c,
                      cfg={"max_bulk": 5, "max_bulk_hard_force": 10, "absolute_mass_floor": 500})
        dry = _call(p, "mem0_delete", {"filter": "target"})
        out = p.handle_tool_call("mem0_delete",
                                 {"filter": "target", "confirm_token": dry["confirm_token"], "force": True})
        assert "ceiling" in out.lower()

    def test_soft_cap_needs_force(self, monkeypatch, tmp_path):
        c = FakeClient()
        for i in range(8):
            c._seed(f"soft {i}")
        for i in range(200):
            c._seed(f"pad {i}")
        p = _provider(monkeypatch, tmp_path, c,
                      cfg={"max_bulk": 5, "max_bulk_hard_force": 100, "absolute_mass_floor": 500})
        dry = _call(p, "mem0_delete", {"filter": "soft"})
        # without force → refused with soft-cap message
        out = p.handle_tool_call("mem0_delete",
                                 {"filter": "soft", "confirm_token": dry["confirm_token"]})
        assert "soft review cap" in out.lower()

    def test_bulk_delete_dryrun_has_forget_hint(self, monkeypatch, tmp_path):
        c = FakeClient()
        for i in range(3):
            c._seed(f"hintcheck {i}")
        for i in range(60):
            c._seed(f"pad {i}")  # keep the 3 matches a minority
        p = _provider(monkeypatch, tmp_path, c)
        out = _call(p, "mem0_delete", {"filter": "hintcheck"})
        assert "mem0_forget" in out["hint"]


class TestVelocity:
    def test_delete_velocity_guard(self, monkeypatch, tmp_path):
        c = FakeClient()
        ids = [c._seed(f"v {i}") for i in range(5)]
        p = _provider(monkeypatch, tmp_path, c, cfg={"max_delete_per_hour": 3})
        # delete 3 individually (ok), 4th should refuse
        for mid in ids[:3]:
            _call(p, "mem0_delete", {"memory_id": mid})
        out = p.handle_tool_call("mem0_delete", {"memory_id": ids[3]})
        assert "velocity" in out.lower()

    def test_velocity_from_durable_ledger_survives_new_instance(self, monkeypatch, tmp_path):
        c = FakeClient()
        ids = [c._seed(f"w {i}") for i in range(5)]
        p1 = _provider(monkeypatch, tmp_path, c, cfg={"max_delete_per_hour": 3})
        for mid in ids[:3]:
            _call(p1, "mem0_delete", {"memory_id": mid})
        # brand new provider instance (simulates gateway restart) — same HERMES_HOME
        p2 = _provider(monkeypatch, tmp_path, c, cfg={"max_delete_per_hour": 3})
        out = p2.handle_tool_call("mem0_delete", {"memory_id": ids[3]})
        assert "velocity" in out.lower()  # restart did NOT reset the guard


# ---------------------------------------------------------------------------
# C7 — scope safety
# ---------------------------------------------------------------------------

class TestScope:
    def test_foreign_user_filter_only_sees_in_scope(self, monkeypatch, tmp_path):
        c = FakeClient()
        c._seed("mine", user_id="ace")
        c._seed("theirs", user_id="someone-else")
        p = _provider(monkeypatch, tmp_path, c)
        # dry-run a filter that would match both texts; only in-scope (ace) seen
        out = _call(p, "mem0_delete", {"filter": ""})  # would be 'all'
        # 'all' in-scope = 1 (just 'mine'); 1/1 = mass → refused, but the point is
        # the foreign memory is never in scope. Verify via resolve directly:
        matches = p._resolve_filter(c, "")
        ids = [m["id"] for m in matches]
        assert all(c._store[i]["user_id"] == "ace" for i in ids)
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# Forget velocity (C8d on forget too)
# ---------------------------------------------------------------------------

class TestForgetVelocity:
    def test_forget_velocity_guard(self, monkeypatch, tmp_path):
        c = FakeClient()
        ids = [c._seed(f"fv {i}") for i in range(5)]
        p = _provider(monkeypatch, tmp_path, c, cfg={"max_forget_per_hour": 2})
        for mid in ids[:2]:
            _call(p, "mem0_forget", {"memory_id": mid})
        out = p.handle_tool_call("mem0_forget", {"memory_id": ids[2]})
        assert "velocity" in out.lower()


# ---------------------------------------------------------------------------
# C9 — recall-hiding single choke point (source guard)
# ---------------------------------------------------------------------------

class TestChokePointGuard:
    """Every read path that returns memories MUST wrap the result in
    _drop_forgotten. A new unwrapped client.search(/client.get_all( in a read
    surface would silently leak forgotten memories back into recall."""

    def test_all_read_search_get_all_calls_are_forgotten_filtered(self):
        import inspect
        from plugins.memory import mem0 as mod
        src = inspect.getsource(mod)
        lines = src.splitlines()
        # Allowed unwrapped call sites: destructive internals that operate on the
        # RAW store on purpose (resolve/total/restore-history/_seed are not recall).
        ALLOW_SUBSTR = (
            "self._unwrap_results(client.get_all(filters=self._read_filters()))",  # _resolve_filter / _in_scope_total raw scope
        )
        offenders = []
        for i, line in enumerate(lines):
            if "client.search(" in line or "client.get_all(" in line:
                # gather a small window to check for _drop_forgotten wrapping
                window = "\n".join(lines[max(0, i - 1):i + 6])
                if "_drop_forgotten" in window:
                    continue
                if any(a in line for a in ALLOW_SUBSTR):
                    continue
                offenders.append((i + 1, line.strip()))
        assert not offenders, (
            "Unwrapped recall read path(s) found (must route through _drop_forgotten "
            f"or be an allowlisted raw-scope call): {offenders}")

