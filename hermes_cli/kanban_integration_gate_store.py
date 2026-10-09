"""Integration-gate declarations and the gate half of the terminal write.

Owns the ``integration_gates`` rows and the snapshot discipline around
:func:`~hermes_cli.kanban_integration_gate.collect_gate_acceptance`: EVERY
board fact the verification was computed from is fingerprinted first, the
network and git work happens with no transaction open, and that fingerprint is
re-read and compared inside ``complete_task``'s write transaction before the
terminal UPDATE. Any change to the gate's run/status, the declaration, the
parents (their rows AND the two ``task_links`` edges that make them parents at
all), the implementation's contract, its accepted acceptance receipt, or the
completed QA run's own summary/metadata rejects the attempt rather than
promoting the gate's child on evidence that has since moved.

``complete_task`` stays the terminal transition owner; this module only
answers "may it?" and records the receipt.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import astuple
from pathlib import Path
from typing import Optional

from hermes_cli.kanban_completion_attempt import stamp_attempt
from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_integration_gate import GateConfig, collect_gate_acceptance

#: The columns one declaration is read back from. ``require_human_merge`` is
#: deliberately absent: the column still exists so a board written by an earlier
#: build opens unchanged, but NOTHING reads it — a human merger is mandatory for
#: every gate (:class:`~hermes_cli.kanban_integration_gate.GateConfig`), so a
#: row that stored ``0`` must not be able to weaken one. Writes pin it back to
#: ``1`` rather than leaving a stale value behind.
_GATE_COLUMNS = (
    "gate_task_id, implementation_task_id, qa_task_id, repository_path, "
    "integration_remote, integration_branch"
)
#: Statuses ``complete_task`` accepts as a source for its terminal UPDATE.
_COMPLETABLE = {"running", "ready", "blocked", "review"}


class IntegrationGateConfigError(ValueError):
    """A gate declaration the board cannot honour (bad ids, graph or paths)."""


# --- declarations ---

def get_gate(conn, gate_task_id: str) -> Optional[GateConfig]:
    """The gate declared for ``gate_task_id``, or None when the card is an
    ordinary one. An empty ``integration_gates`` table makes every card
    ordinary, which is what keeps the feature opt-in."""
    row = conn.execute(
        f"SELECT {_GATE_COLUMNS} FROM integration_gates WHERE gate_task_id = ?", (gate_task_id,),
    ).fetchone()
    return _row_to_config(row) if row is not None else None


def list_gates(conn) -> list[GateConfig]:
    return [_row_to_config(row) for row in conn.execute(
        f"SELECT {_GATE_COLUMNS} FROM integration_gates ORDER BY gate_task_id")]


def _row_to_config(row) -> GateConfig:
    return GateConfig(
        gate_task_id=row["gate_task_id"],
        implementation_task_id=row["implementation_task_id"],
        qa_task_id=row["qa_task_id"],
        repository_path=row["repository_path"],
        integration_remote=row["integration_remote"],
        integration_branch=row["integration_branch"],
    )


def configure_gate(
    conn, gate_task_id: str, *, implementation_task_id: str, qa_task_id: str,
    repository_path: str, integration_remote: str = "origin",
    integration_branch: str = "develop",
) -> GateConfig:
    """Declare (or re-declare) the gate on ``gate_task_id``.

    Implementation and QA must ALREADY be direct parents of the gate card:
    declaring a gate is not allowed to rewrite an existing board's graph
    implicitly, so a missing edge is an error naming the ``hermes kanban link``
    that fixes it. Only the declaration row and an audit event are written —
    no card changes status, assignee or links.

    There is no human-merge parameter: every gate requires a human merger, and
    re-declaring one repairs a row an earlier build stored otherwise.
    """
    gate_task_id = _require_id(gate_task_id, "gate task id")
    implementation_task_id = _require_id(implementation_task_id, "implementation task id")
    qa_task_id = _require_id(qa_task_id, "QA task id")
    if len({gate_task_id, implementation_task_id, qa_task_id}) != 3:
        raise IntegrationGateConfigError(
            "gate, implementation and QA must be three different tasks")
    remote = _require_ref_token(integration_remote, "integration remote")
    branch = _require_ref_token(integration_branch, "integration branch")
    repo_path = _validated_repository_path(repository_path)
    from hermes_cli.kanban_db import _append_event

    with write_txn(conn):
        for label, task_id in (("gate", gate_task_id),
                               ("implementation", implementation_task_id),
                               ("QA", qa_task_id)):
            if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
                raise IntegrationGateConfigError(f"unknown {label} task {task_id}")
        for label, parent_id in (("implementation", implementation_task_id), ("QA", qa_task_id)):
            if conn.execute(
                "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, gate_task_id),
            ).fetchone() is None:
                raise IntegrationGateConfigError(
                    f"{label} task {parent_id} is not a direct parent of gate {gate_task_id}; "
                    f"link it first (`hermes kanban link {parent_id} {gate_task_id}`). "
                    f"Configuring a gate never rewrites the board graph."
                )
        conn.execute(
            f"INSERT INTO integration_gates ({_GATE_COLUMNS}, require_human_merge, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, ?) "
            "ON CONFLICT(gate_task_id) DO UPDATE SET "
            "implementation_task_id = excluded.implementation_task_id, "
            "qa_task_id = excluded.qa_task_id, repository_path = excluded.repository_path, "
            "integration_remote = excluded.integration_remote, "
            "integration_branch = excluded.integration_branch, "
            # Pinned, never carried over: re-declaring repairs a row an earlier
            # build wrote with the retired opt-out.
            "require_human_merge = 1",
            (gate_task_id, implementation_task_id, qa_task_id, repo_path, remote, branch,
             int(time.time())),
        )
        config = get_gate(conn, gate_task_id)
        _append_event(conn, gate_task_id, "integration_gate_configured", config.as_dict())
    return config


def remove_gate(conn, gate_task_id: str) -> bool:
    """Drop the declaration, returning the card to ordinary parent gating."""
    from hermes_cli.kanban_db import _append_event

    with write_txn(conn):
        removed = conn.execute(
            "DELETE FROM integration_gates WHERE gate_task_id = ?", (gate_task_id,)).rowcount == 1
        if removed:
            _append_event(conn, gate_task_id, "integration_gate_removed", None)
    return removed


def _require_id(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationGateConfigError(f"{label} is required")
    return value.strip()


def _require_ref_token(value, label: str) -> str:
    """Remote/branch names are interpolated into a git refspec, so reject
    anything that is not a plain name."""
    token = value.strip() if isinstance(value, str) else ""
    if not token or token.startswith("-") or any(c.isspace() for c in token) or ".." in token \
            or any(c in token for c in "~^:?*[\\"):
        raise IntegrationGateConfigError(f"{label} must be a plain git ref name, got {value!r}")
    return token


def _validated_repository_path(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationGateConfigError("repository path is required")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise IntegrationGateConfigError(
            f"repository path must be absolute, got {value!r} (the gate is verified by whichever "
            f"process completes it, not from your shell's cwd)")
    if not path.is_dir():
        raise IntegrationGateConfigError(f"repository path {path} is not a directory")
    return str(path)


# --- terminal-transition gate ---

def prepare_integration_gate(conn, task_id: str, expected_run_id: Optional[int], *,
                             attempt_id: Optional[str] = None):
    """``None`` when the card is not a declared gate, ``False`` when the
    caller's run/status snapshot is already lost, else ``(fingerprint, receipt)``.

    The external verification runs here — deliberately BEFORE
    ``complete_task`` opens its write transaction, so no network or git call
    ever holds the board's write lock. ``attempt_id`` identifies the completion
    attempt the resulting receipt belongs to.
    """
    if get_gate(conn, task_id) is None:
        # No declaration: an ordinary card, which the feature must not touch.
        return None
    state = _gate_state(conn, task_id)
    if state is None:
        return False
    fingerprint, evidence, config = state
    run_id, status = fingerprint[0], fingerprint[1]
    if status not in _COMPLETABLE or (expected_run_id is not None and run_id != expected_run_id):
        return False
    # The declaration verified against is the one the fingerprint covers.
    return fingerprint, stamp_attempt(collect_gate_acceptance(config, evidence), attempt_id)


def record_integration_gate(conn, task_id: str, prepared) -> bool:
    """Called under ``complete_task``'s write_txn, before its terminal UPDATE.

    Re-reads every fact the receipt was approved from and refuses unless all of
    them are byte-for-byte what the verification saw, then — on the success path
    only — asks once more, by name, whether the gate's two declared parent edges
    still exist. Persists the immutable receipt either way: a failed gate must
    leave diagnostics behind without making the gate — or its child —
    executable.
    """
    from hermes_cli.kanban_db import _append_event

    fingerprint, receipt = prepared
    state = _gate_state(conn, task_id)
    if state is None or state[0] != fingerprint:
        return False
    _append_event(conn, task_id, "integration_acceptance", receipt, run_id=fingerprint[0])
    if not receipt["ok"]:
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            (f"Integration gate {receipt['phase']}: {receipt.get('detail', '')} "
             f"{receipt['recovery']}", task_id),
        )
        return False
    # The two declared parent edges, asked for again and by name. The
    # fingerprint already covers them, so this cannot normally differ — it is
    # here because the one thing the terminal write must never do is promote a
    # gate's child while the gate's own declared parents are not its parents,
    # and that must not depend on reading a digest correctly.
    if not all(_declared_parent_edges(conn, state[2]).values()):
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("Integration gate declared_parents_linked: the gate's declared implementation/QA "
             "edges are gone from the board graph. " + _RELINK_HINT, task_id),
        )
        return False
    return True


def _gate_state(conn, task_id: str):
    """``(fingerprint, evidence, config)`` for the gate, or ``None`` when the
    card or its declaration is gone.

    The fingerprint covers EVERY fact the verification is allowed to approve
    from, not just the gate's own run/status pair — because almost all of that
    evidence stays mutable while the network and git work runs with no
    transaction open. A completed QA run's summary and metadata are editable
    after the fact (``edit_task(result=…)``, which changes neither the
    run id nor any status), the implementation's pinned
    ``completion_contract`` and its accepted ``pr_acceptance`` receipt are rows
    like any other, the two ``task_links`` edges that make implementation and
    QA the gate's PARENTS can be unlinked, and the declaration itself can be
    re-configured. Comparing only the ids would let ``complete_task`` promote a
    gate's child on evidence that no longer exists; comparing this tuple
    refuses instead.

    The parent edges are in the tuple explicitly rather than only inside the
    digest: losing one is the one change that makes the board's own dependency
    check *easier* to satisfy (fewer parents to be done, and with both gone it
    is vacuously true), so it is the last thing that may be covered by
    implication.
    """
    row = conn.execute(
        "SELECT current_run_id, status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    config = get_gate(conn, task_id)
    if row is None or config is None:
        return None
    evidence = _board_evidence(conn, config)
    parents = tuple(
        (conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (parent_id,))
         .fetchone() or {"status": None, "current_run_id": None})
        for parent_id in (config.implementation_task_id, config.qa_task_id)
    )
    fingerprint = (
        row["current_run_id"], row["status"], astuple(config),
        tuple((p["status"], p["current_run_id"]) for p in parents),
        (evidence["implementation_linked"], evidence["qa_linked"]),
        evidence["evidence_digest"],
    )
    return fingerprint, evidence, config


def _board_evidence(conn, config: GateConfig) -> dict:
    """Everything the verification needs from the board, read in one pass.

    ``evidence_digest`` fingerprints the RAW rows each condition is decided
    from rather than the derived fields above them, so a rewrite that leaves
    every id and status alone still invalidates the snapshot. It is a digest
    rather than the values themselves because the summary, metadata and
    acceptance payload it covers are free text the snapshot has no reason to
    hold, log or compare in the clear.
    """
    # ``assignee`` rides along because the gate's GitHub read runs as THAT
    # profile's ``gh`` login — this is the implementation card's PR (#122689).
    impl = conn.execute(
        "SELECT status, assignee, current_run_id, completion_contract FROM tasks WHERE id = ?",
        (config.implementation_task_id,)).fetchone()
    qa = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (config.qa_task_id,)).fetchone()
    qa_run = conn.execute(
        "SELECT id, summary, metadata FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
        "ORDER BY id DESC LIMIT 1", (config.qa_task_id,)).fetchone()
    qa_metadata = _json_dict(qa_run["metadata"]) if qa_run is not None else {}
    accepted_event_id, accepted_payload, accepted = _accepted_acceptance_receipt(
        conn, config.implementation_task_id)
    edges = _declared_parent_edges(conn, config)
    return {
        "implementation_linked": edges[config.implementation_task_id],
        "qa_linked": edges[config.qa_task_id],
        "implementation_status": impl["status"] if impl is not None else None,
        "implementation_assignee": impl["assignee"] if impl is not None else None,
        "contract": impl["completion_contract"] if impl is not None else None,
        "qa_status": qa["status"] if qa is not None else None,
        "qa_run_id": int(qa_run["id"]) if qa_run is not None else None,
        "qa_decision": _str_or_none(qa_metadata.get("decision")),
        "qa_revision": _str_or_none(qa_metadata.get("revision")),
        "qa_summary_mentions_pass": "pass" in str(
            (qa_run["summary"] if qa_run is not None else "") or "").lower(),
        # Coerced because the receipt is JSON read back off the event log: a
        # non-string here would otherwise raise inside the regex checks and
        # abort the completion instead of blocking the gate.
        "accepted_head_sha": _str_or_none(accepted.get("head_sha")),
        "pr_url": _str_or_none(accepted.get("pr_url")),
        "evidence_digest": _digest([
            astuple(config),
            # ``assignee`` is in the digest because it chooses the gh identity the
            # PR was read as: a reassignment mid-verification means the merge
            # evidence came from a login the terminal write is no longer about.
            None if impl is None else [impl["status"], impl["current_run_id"],
                                      impl["assignee"], impl["completion_contract"]],
            None if qa is None else [qa["status"], qa["current_run_id"]],
            None if qa_run is None else [qa_run["id"], qa_run["summary"], qa_run["metadata"]],
            [accepted_event_id, accepted_payload],
        ]),
    }


#: Shown on a refusal that only an operator's `link` can fix.
_RELINK_HINT = ("Restore the edge with `hermes kanban link <parent> <gate>`, or remove the "
                "declaration with `hermes kanban integration-gate rm <gate>`.")


def _declared_parent_edges(conn, config: GateConfig) -> dict:
    """``{task_id: is still a direct parent of the gate}`` for both declarations."""
    return {
        parent_id: conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, config.gate_task_id),
        ).fetchone() is not None
        for parent_id in (config.implementation_task_id, config.qa_task_id)
    }


def _digest(parts) -> str:
    """Stable digest of the evidence rows. ``default=str`` keeps a surprising
    column type (a BLOB summary, a float id) from raising here — the snapshot's
    job is to notice change, and an undigestible row must not abort a
    completion the gate would otherwise refuse with a receipt."""
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _is_canonical_acceptance(payload: dict) -> bool:
    """Whether ``payload`` is the one receipt shape PR acceptance writes when it
    accepts (``collect_acceptance``'s ``ok=True, classification="success",
    phase="accepted"``).

    All three are compared exactly, and ``ok`` by identity, because this payload
    is JSON read back off the event log rather than a value this process
    produced: ``"false"`` and ``1`` both pass a truthiness test and ``1``
    passes an equality one, so a hand-written, half-written or
    differently-versioned event could otherwise hand the gate an accepted head
    that no PR acceptance ever approved — and promote the gate's child on it.
    The two verdict fields are in the test because they are what names the
    verdict: ``ok`` is the only field a refusal receipt and an acceptance
    receipt share a key for.
    """
    return (
        payload.get("ok") is True
        and payload.get("classification") == "success"
        and payload.get("phase") == "accepted"
    )


def _accepted_acceptance_receipt(conn, implementation_task_id: str):
    """``(event_id, raw_payload, payload)`` of the newest ACCEPTED
    ``pr_acceptance`` receipt on the implementation card, else
    ``(None, None, {})``.

    That receipt is the only durable record of which exact head GitHub
    acceptance actually passed, which is the head QA's revision must match — so
    both WHICH event it is and what it says belong in the snapshot. Anything
    that is not the canonical accepted shape is not a receipt at all here: no
    accepted head is projected from it, which is the refusal the gate's
    ``accepted_head_known`` condition already reports.
    """
    for row in conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = 'pr_acceptance' "
        "ORDER BY id DESC", (implementation_task_id,),
    ):
        payload = _json_dict(row["payload"])
        if _is_canonical_acceptance(payload):
            return int(row["id"]), row["payload"], payload
    return None, None, {}


def _json_dict(raw) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _str_or_none(value):
    return value if isinstance(value, str) and value.strip() else None
