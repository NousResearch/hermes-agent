"""P3b contract SPEC (audit 2026-10-04, item P3b).

``create_task`` so far took ``completion_contract`` straight from
``validate_contract`` (local-only / OWNER-REPO / GitHub PR URL); a JSON
requirements dict failed the regex and the only way in was a raw SQL UPDATE
(what test_kanban_completion_contract does directly). That left
``create_task`` unable to persist a declared ``required_artifacts`` list and
let unvalidated list entries (relative / blank / non-str) reach the gate,
where the TypeError-tolerant gate ignores them (fail-open) — a declared but
malformed requirement silently enforced nothing.

P3b adds creation-time validation in :mod:`hermes_cli.kanban_db`:

  * Legacy create_task contracts stay exactly as before: ``None`` is stored
    as 'local-only'; 'local-only', OWNER/REPO, GitHub PR URL accepted and
    stored as given; invalid shapes ('x', 'a b c', '') raise PLAIN ValueError.
  * NEW: a JSON object with a ``required_artifacts`` LIST of absolute
    non-blank string paths is accepted, canonicalized (compact separators),
    stored and returned unchanged by SELECT (round-trip).
    Bad shapes raise :class:`ContractSpecError` (a ``ValueError`` subclass,
    defined next to :class:`ContractCompletionError`).

The completion gate itself is unchanged — same inert/fail-open shapes as
test_kanban_completion_contract pins — EXCEPT the new TOCTOU re-check: the
contract is revalidated against the value read INSIDE the ``complete_task``
write transaction, after the UPDATE fence succeeds (rowcount == 1), so a
contract swapped by a concurrent writer between the gate check and this txn
cannot flip the gate: the completion raises ``ContractSpecError`` (the card
stays in its prior status; the completion txn, including the status flip,
rolls back) and an audible ``completion_blocked_contract_changed`` event
records the divergence. That event fires ONLY for a requirements-dict
contract that a concurrent writer swapped — inert/PR contracts keep the
exact pre-P3b behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

SCRATCH_WS = ".hermes/kanban/scratch"


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    kb.init_db()
    return home


@pytest.fixture
def home_init(tmp_path, monkeypatch):
    """Init-only home fixture for pass-through CLI tests (no conn)."""
    _home(tmp_path, monkeypatch)


def _row_value(conn, task_id, column):
    return conn.execute(
        f"SELECT {column} FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


def _event_kinds(conn, task_id):
    return [
        r[0]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _scratch_task(conn, tmp_path, title, contract=None):
    """Task with a managed-scratch workspace dir under kanban/workspaces."""
    from hermes_cli import kanban_db_workspace as kbw

    t = kb.create_task(conn, title=title, completion_contract="local-only")
    ws = tmp_path / ".hermes" / "kanban" / "workspaces" / "default" / t
    ws.mkdir(parents=True, exist_ok=True)
    kbw.set_workspace_path(conn, t, str(ws))
    return t, ws


def _event_payloads(conn, task_id, kind):
    return [
        json.loads(r[0]) if r[0] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


class TestContractSpec:
    """P3b required_artifacts JSON contracts settable at create_task."""

    pretty = '{"required_artifacts": ["/tmp/rel/file.txt"]}'
    canonical = '{"required_artifacts":["/tmp/rel/file.txt"]}'

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        return _home(tmp_path, monkeypatch)

    @pytest.fixture
    def conn(self, home):
        with kbc.connect() as c:
            yield c

    # (a) JSON round-trip ------------------------------------------------------

    def test_json_contract_roundtrip(self, conn):
        tid = kb.create_task(conn, title="spec json", completion_contract=self.pretty)
        assert _row_value(conn, tid, "completion_contract") == self.canonical
        assert kb.get_task(conn, tid).completion_contract == self.canonical

    def test_canonicalization_is_stable(self, conn):
        """The canonical string revalidates to itself (idempotent validator)."""
        tid = kb.create_task(conn, title="spec canon", completion_contract=self.pretty)
        stored = _row_value(conn, tid, "completion_contract")
        assert kb._validate_contract_spec(stored) == stored

    # (b) entry/type enforcement -> ContractSpecError -------------------------

    def test_artifact_non_str_entries_rejected(self, conn):
        for contract in (
            '{"required_artifacts": [1, 2]}',
            '{"required_artifacts": [null]}',
            '{"required_artifacts": ["/tmp/rel/file.txt", 7]}',
        ):
            with pytest.raises(kb.ContractSpecError):
                kb.create_task(conn, title="spec bad entry", completion_contract=contract)

    def test_artifact_relative_path_rejected(self, conn):
        with pytest.raises(kb.ContractSpecError):
            kb.create_task(
                conn,
                title="spec relative",
                completion_contract='{"required_artifacts": ["rel/file.txt"]}',
            )

    def test_artifact_blank_path_rejected(self, conn):
        for contract in ('{"required_artifacts": [""]}', '{"required_artifacts": ["   "]}'):
            with pytest.raises(kb.ContractSpecError):
                kb.create_task(conn, title="spec blank", completion_contract=contract)

    def test_required_artifacts_non_list_rejected(self, conn):
        with pytest.raises(kb.ContractSpecError):
            kb.create_task(
                conn,
                title="spec non list key",
                completion_contract='{"required_artifacts": "x"}',
            )

    # (c) top-level non-dict ----------------------------------------------------

    def test_top_level_non_dict_rejected(self, conn):
        for contract in ('["a", "b"]', '"just a string"', "42"):
            with pytest.raises(kb.ContractSpecError):
                kb.create_task(conn, title="spec non dict", completion_contract=contract)

    # (e) empty list ------------------------------------------------------------

    def test_empty_required_list_is_legal(self, conn):
        tid = kb.create_task(
            conn, title="spec empty required", completion_contract='{"required_artifacts": []}'
        )
        stored = _row_value(conn, tid, "completion_contract")
        assert stored == '{"required_artifacts":[]}'
        assert kb.get_task(conn, tid).completion_contract == '{"required_artifacts":[]}'


class TestLegacyContractsUnchanged:
    """Classic create_task contracts survive P3b exactly as before."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        return _home(tmp_path, monkeypatch)

    @pytest.fixture
    def conn(self, home):
        with kbc.connect() as c:
            yield c

    def test_none_maps_to_local_only(self, conn):
        tid = kb.create_task(conn, title="p3b None")
        assert kb.get_task(conn, tid).completion_contract == "local-only"

    def test_local_only_stored_as_is(self, conn):
        tid = kb.create_task(conn, title="p3b local-only", completion_contract="local-only")
        assert kb.get_task(conn, tid).completion_contract == "local-only"

    def test_repo_contract_stored_as_is(self, conn):
        tid = kb.create_task(conn, title="p3b repo", completion_contract="owner/repo")
        assert kb.get_task(conn, tid).completion_contract == "owner/repo"

    def test_pr_url_contract_stored_as_is(self, conn):
        tid = kb.create_task(
            conn, title="p3b pr url", completion_contract="https://github.com/o/r/pull/9"
        )
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/o/r/pull/9"

    def test_invalid_shapes_raise_plain_valueerror(self, conn):
        """'x' / 'a b c' / '' keep raising the legacy PLAIN ValueError — the
        legacy message and the NOT-a-ContractSpecError type are both pinned."""
        for contract in ("x", "a b c", ""):
            with pytest.raises(ValueError) as excinfo:
                kb.create_task(conn, title="p3b invalid", completion_contract=contract)
            assert "must be local-only" in str(excinfo.value)
            assert not isinstance(excinfo.value, kb.ContractSpecError)


# ---------------------------------------------------------------------------
# 2.x Bridge: the CLI passes the raw string through
# ---------------------------------------------------------------------------


def _cli_args(contract: str, title: str = "cli contract") -> argparse.Namespace:
    """Namespace from the REAL kanban parser (``kanban new`` defaults) with
    the completion contract set. P3b does not touch the parser: the arg stays
    a single raw string; create_task (not the parser) does the validation."""
    from hermes_cli.kanban_parser import _add_commands, _SPECS

    parser = argparse.ArgumentParser(prog="hermes kanban")
    _add_commands(parser.add_subparsers(dest="kanban_action"), _SPECS)
    ns = parser.parse_args(["create", title, "--completion-contract", contract])
    assert getattr(ns, "completion_contract", None) == contract
    return ns


@pytest.mark.usefixtures("home_init")
class TestCliContract:
    """The CLI is a pure pass-through: the raw --completion-contract string
    reaches create_task, and the JSON form creates the card with the contract
    canonicalized exactly as the API stores it."""

    def test_cli_json_contract_creates_card(self, home_init):
        from hermes_cli import kanban as kc

        rc = kc._cmd_create(
            _cli_args('{"required_artifacts": ["/tmp/rel/file.txt"]}', title="cli json")
        )
        assert rc == 0
        with kbc.connect() as conn:
            row = conn.execute(
                "SELECT completion_contract, title FROM tasks ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        assert row[0] == '{"required_artifacts":["/tmp/rel/file.txt"]}'
        assert row[1] == "cli json"

    def test_cli_local_only_contract_creates_card(self, home_init):
        from hermes_cli import kanban as kc

        assert kc._cmd_create(_cli_args("local-only", title="cli local-only")) == 0
        with kbc.connect() as conn:
            row = conn.execute(
                "SELECT completion_contract, title FROM tasks ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "local-only"
        assert row[1] == "cli local-only"

    def test_cli_repo_contract_creates_card(self, home_init):
        from hermes_cli import kanban as kc

        assert kc._cmd_create(_cli_args("owner/repo", title="cli repo")) == 0
        with kbc.connect() as conn:
            row = conn.execute(
                "SELECT completion_contract, title FROM tasks ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "owner/repo"
        assert row[1] == "cli repo"


# ---------------------------------------------------------------------------
# TOCTOU: contract swapped between gate check and the completion txn
# ---------------------------------------------------------------------------


class TestContractChangedInTxn:
    """Revalidation against the in-txn read: a concurrent writer that swaps a
    requirements-dict contract after the gate check meets a refused completion
    with the card's state intact."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        return _home(tmp_path, monkeypatch)

    @pytest.fixture
    def conn(self, home):
        with kbc.connect() as c:
            yield c

    def _claimed_scratch_task(self, conn, home, title="p3b toctou"):
        """Task with a managed-scratch workspace holding satisfying.md,
        claimed ready->running (proof-of-ownership run for completion)."""
        from hermes_cli import kanban_db_workspace as kbw

        tid = kb.create_task(conn, title=title, completion_contract="local-only")
        ws = home / "kanban" / "workspaces" / "default" / tid
        ws.mkdir(parents=True, exist_ok=True)
        kbw.set_workspace_path(conn, tid, str(ws))
        good = ws / "satisfy.md"
        good.write_text("# ok", encoding="utf-8")
        bad = ws / "ghost.md"
        gate_contract = json.dumps(
            {"required_artifacts": [str(good)]}, separators=(",", ":")
        )
        bad_contract = json.dumps(
            {"required_artifacts": [str(bad)]}, separators=(",", ":")
        )
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?",
            (gate_contract, tid),
        )
        conn.commit()
        claimed = kb.claim_task(conn, tid, claimer="host:mock")
        assert claimed is not None, "claim must succeed for the completion path"
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None
        return tid, ws, run_id

    def test_contract_swapped_to_stricter_refused(self, conn, home, monkeypatch):
        """Gate-time contract passes; the txn sees a swapped contract whose
        requirement is missing on disk. The completion raises
        ContractSpecError, the card stays running and the divergence event
        records both contract versions."""
        tid, _ws, run_id = self._claimed_scratch_task(conn, home)
        bad_contract = json.dumps(
            {"required_artifacts": [str(_ws / "ghost.md")]}, separators=(",", ":")
        )

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                # The concurrent writer commits its stricter contract here —
                # own connection: it must NOT ride the completion txn.
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (bad_contract, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)

        with pytest.raises(kb.ContractSpecError) as excinfo:
            kb.complete_task(
                conn, tid, result="submitted against contract A", expected_run_id=run_id
            )
        # No payload on ContractSpecError (the divergence event carries the
        # detail): the message names the swap; the EVENT names both versions.
        assert not hasattr(excinfo.value, "payload")
        assert "contract changed between the artifact gate" in str(excinfo.value)
        # The status flip is part of the rolled-back txn: card stays running.
        assert _row_value(conn, tid, "status") == "running"
        payloads = _event_payloads(conn, tid, "completion_blocked_contract_changed")
        assert payloads, "the divergence must be audible"
        assert any("ghost.md" in json.dumps(p) for p in payloads)

    def test_contract_inert_to_strict_in_window_refused(self, conn, home, monkeypatch):
        """Review 5 ``inert_to_strict`` HIGH (reproduced 2026-10-04): the gate read an
        INERT contract (nothing stashed, nothing enforced) and a concurrent writer
        swapped a requirements dict with a missing artifact into the gate→txn window —
        the old recheck returned early on a None stash and the completion closed the
        card with a fence it never tested. Now the None-stash side re-reads the row and
        refuses ANY unenforced→enforced class change (inert→strict especially)."""
        tid, _ws, run_id = self._claimed_scratch_task(conn, home, title="p3b inert->strict")
        # The gate runs on the INERT value; the writer swaps strict mid-window.
        conn.execute(
            "UPDATE tasks SET completion_contract = 'local-only' WHERE id = ?", (tid,)
        )
        conn.commit()
        bad_contract = json.dumps(
            {"required_artifacts": [str(_ws / "ghost.md")]}, separators=(",", ":")
        )
        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (bad_contract, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        with pytest.raises(kb.ContractSpecError) as excinfo:
            kb.complete_task(
                conn, tid, result="submitted against unexamined contract",
                expected_run_id=run_id,
            )
        assert "contract changed between the artifact gate" in str(excinfo.value)
        assert _row_value(conn, tid, "status") == "running"
        payloads = _event_payloads(conn, tid, "completion_blocked_contract_changed")
        assert payloads, "inert→strict swap must be audible"
        assert any("ghost.md" in json.dumps(p) for p in payloads)

    def test_contract_inert_to_strict_refused_even_with_artifact_present(
        self, conn, home, monkeypatch
    ):
        """Same inert→strict swap but the artifact EXISTS: the gate's no-enforcement
        decision still cannot close the card on a contract it never examined — refusal
        regardless of artifact presence (the fence rides the examined contract only)."""
        tid, ws, run_id = self._claimed_scratch_task(
            conn, home, title="p3b inert->strict-persistent"
        )
        conn.execute(
            "UPDATE tasks SET completion_contract = 'local-only' WHERE id = ?", (tid,)
        )
        conn.commit()
        present = ws / "present.md"
        present.write_text("# exists", encoding="utf-8")
        strict_contract = json.dumps(
            {"required_artifacts": [str(present)]}, separators=(",", ":")
        )
        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (strict_contract, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(
                conn, tid, result="artifact present, contract unexamined",
                expected_run_id=run_id,
            )
        assert _row_value(conn, tid, "status") == "running"

    def test_unenforced_to_unenforced_still_passes(self, conn, home, monkeypatch):
        """The informative no-op: inert/''/'local-only'/OWNER-REPO/PR-URL/unparseable
        fail-open shapes are ONE unenforced class — a 'swap' inside that class (e.g.
        o/r→local-only) changes no enforcement and must NOT add a
        contract_changed refusal (pre-P3b byte-for-byte completion semantics)."""
        tid, _ws, run_id = self._claimed_scratch_task(conn, home, title="p3b inert class")
        conn.execute(
            "UPDATE tasks SET completion_contract = 'o/r' WHERE id = ?", (tid,)
        )
        conn.commit()
        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = 'local-only' WHERE id = ?",
                        (task_id,),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        # The completion may be refused by OTHER pre-existing fences (the PR-acceptance
        # store owns 'o/r' and refuses without its machinery) — but the artifact gate
        # must NOT have added a contract_changed event: the swap changes no enforcement.
        kb.complete_task(
            conn, tid, result="class-internal swap", expected_run_id=run_id
        )
        assert not _event_payloads(conn, tid, "completion_blocked_contract_changed")

    def test_contract_swapped_to_inert_refused(self, conn, home, monkeypatch):
        """The OTHER direction: gate saw a requirements dict, the txn sees the
        contract swapped to inert 'local-only'. The dict contract the gate
        checked is no longer the one the card carries — the swapped IN result
        must replay deterministically against the checked contract, so this
        direction refuses too (enforcement-decision integrity, not only
        tightening)."""
        tid, _ws, run_id = self._claimed_scratch_task(conn, home, title="p3b toctou inert")

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = 'local-only' WHERE id = ?",
                        (task_id,),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)

        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(
                conn, tid, result="submitted against dict contract",
                expected_run_id=run_id,
            )
        assert _row_value(conn, tid, "status") == "running"
        assert "completion_blocked_contract_changed" in _event_kinds(conn, tid)

    def test_contract_event_payload_carries_divergence(self, conn, home, monkeypatch):
        """The divergence event names both versions: gate-time and in-txn."""
        tid, ws, run_id = self._claimed_scratch_task(conn, home, title="p3b payload")
        bad_contract = json.dumps(
            {"required_artifacts": [str(ws / "ghost.md")]}, separators=(",", ":")
        )

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (bad_contract, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(conn, tid, result="submitted", expected_run_id=run_id)
        payloads = _event_payloads(conn, tid, "completion_blocked_contract_changed")
        assert payloads
        blob = json.dumps(payloads)
        assert "satisfy.md" in blob, "gate-time contract must be recorded"
        assert "ghost.md" in blob, "in-txn contract must be recorded"


class TestContractSwappedSchemaInvalidInTxn:
    """Round 3 (review 5 MEDIUM, `swap_between_parse_and_stash` probe): the
    in-txn recheck compared TEXT and re-checked artifact EXISTENCE but never
    re-validated the SCHEMA of the value the fence rode on. Two reproductions:

    * A rogue writer (raw UPDATE, bypassing create_task's canonization) puts
      a schema-INVALID dict (``required_artifacts`` a non-list) on the card
      BEFORE the gate reads it: the gate's tolerant loop sees a non-iterable
      key, declares zero requirements, stashes the raw text — and the
      same-text recheck (only missing-artifacts) completed the card with the
      contract carrying an illegal shape. The recheck must now run the SAME
      pure helper creation uses (``_contract_spec_reason``) on the in-txn
      text and refuse with reason ``schema_invalid_in_txn``.
    * A valid dict at gate time is swapped in-window for a DIFFERENT-text
      schema-invalid dict: the text-diff branch already refused, but with
      the imprecise ``contract_changed_in_txn``; the reason must upgrade to
      ``schema_invalid_in_txn`` because the value that sits in the row is
      itself illegal (what the audit needs to distinguish a reneged
      requirement from garbage).
    """

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        return _home(tmp_path, monkeypatch)

    @pytest.fixture
    def conn(self, home):
        with kbc.connect() as c:
            yield c

    def test_schema_invalid_written_directly_completes_NO_MORE(self, conn, home):
        """t1 (RED round 3): schema-invalid dict written by raw UPDATE; the
        recheck must refuse the completion (ContractSpecError via the
        swapped-in-txn path), no completed event, card stays running."""
        tid, ws, run_id = self._claimed_scratch_task(conn, home, title="p3b r3 direct")
        bad_text = json.dumps({"required_artifacts": 42}, separators=(",", ":"))
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?", (bad_text, tid)
        )
        conn.commit()
        with pytest.raises(kb.ContractSpecError) as excinfo:
            kb.complete_task(conn, tid, result="submitted", expected_run_id=run_id)
        assert "schema invalid in this transaction" in str(excinfo.value)
        assert _row_value(conn, tid, "status") == "running"
        payloads = _event_payloads(conn, tid, "completion_blocked_contract_changed")
        assert payloads, "the schema divergence must be audible"
        # Burst discipline: a duplicate-artifact TEXT must fail the same way
        # (the schema check is a value rule, not only a top-level-key rule).
        bad_text2 = json.dumps({"required_artifacts": "x"}, separators=(",", ":"))
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?", (bad_text2, tid)
        )
        conn.commit()
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(conn, tid, result="submitted again", expected_run_id=run_id)

    def _claimed_scratch_task(self, conn, home, title="p3b r3"):
        from hermes_cli import kanban_db_workspace as kbw

        tid = kb.create_task(conn, title=title, completion_contract="local-only")
        ws = home / "kanban" / "workspaces" / "default" / tid
        ws.mkdir(parents=True, exist_ok=True)
        kbw.set_workspace_path(conn, tid, str(ws))
        good = ws / "satisfy.md"
        good.write_text("# ok", encoding="utf-8")
        claimed = kb.claim_task(conn, tid, claimer="host:mock")
        assert claimed is not None, "claim must succeed for the completion path"
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None
        return tid, ws, run_id

    def test_gate_valid_swapped_to_schema_invalid_in_window(
        self, conn, home, monkeypatch
    ):
        """t2 (RED round 3): gate checked a LEGAL dict; a concurrent writer
        swapped a schema-invalid dict DIFFERENT text in-window. Old behavior:
        refused with the generic reason contract_changed_in_txn. Round-3
        upgrade: the reason must be schema_invalid_in_txn (the in-txn value
        is illegal per se) — the event payload names it, and the
        ContractSpecError message reaches the caller with the underlying
        spec reason (required_artifacts must be a list)."""
        tid, ws, run_id = self._claimed_scratch_task(conn, home, title="p3b r3 swap")
        good = ws / "satisfy.md"
        gate_contract = json.dumps(
            {"required_artifacts": [str(good)]}, separators=(",", ":")
        )
        bad_text = json.dumps({"required_artifacts": 42}, separators=(",", ":"))
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?",
            (gate_contract, tid),
        )
        conn.commit()

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (bad_text, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        with pytest.raises(kb.ContractSpecError) as excinfo:
            kb.complete_task(
                conn, tid, result="submitted against swapped garbage",
                expected_run_id=run_id,
            )
        assert _row_value(conn, tid, "status") == "running"
        payloads = _event_payloads(conn, tid, "completion_blocked_contract_changed")
        assert payloads, "the swap must be audible"
        assert any(
            p.get("reason") == "schema_invalid_in_txn" for p in payloads
        ), payloads
        # The message carries the ORIGINAL spec reason from the pure helper.
        assert "required_artifacts must be a list" in str(excinfo.value)


def test_contract_spec_error_is_valueerror():
    """ContractSpecError must be a ValueError subclass (recoverable per the
    tool error handlers' contract) covering the json-spec rejections."""
    assert issubclass(kb.ContractSpecError, ValueError)
    with pytest.raises(kb.ContractSpecError):
        kb._validate_contract_spec('{"required_artifacts": ["rel/file.txt"]}')
    with pytest.raises(kb.ContractSpecError):
        kb._validate_contract_spec('["just", "a", "list"]')
    # Legal shapes pass through untouched.
    assert kb._validate_contract_spec(None) is None
    assert kb._validate_contract_spec("local-only") == "local-only"
