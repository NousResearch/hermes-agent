"""C1N-T15 — executable source/AST closure guards for the ``task_events`` writers.

These guards assert *relationships*, never magnitudes. They deliberately do not
freeze the number of ``_append_event`` call sites, the size of
``KANBAN_EVENT_KINDS``, or the number of writers in any module: those are
exact-base evidence and would turn an incidental count into API.

The scan is not literal-only. ``_rebuild_drifted_tables`` builds its
``INSERT INTO {table} ... SELECT ... FROM {table}_legacy`` by f-string
interpolation over ``_REBUILD_SPECS``, so a scan matching only the literal
string ``INSERT INTO task_events`` under-reports the writer closure. Every hit
is then classified as a *semantic new event* (must be instrumented) or a
*non-semantic copy/mutation* (must emit nothing).
"""

from __future__ import annotations

import ast
from pathlib import Path

from hermes_cli import kanban_db as kb

REPO_ROOT = Path(__file__).resolve().parents[2]
KANBAN_DB = REPO_ROOT / "hermes_cli" / "kanban_db.py"

SEAM = "_append_event"

#: The functions allowed to touch ``task_events`` without emitting anything,
#: because they create no semantically new event. Membership here is the claim
#: under test, not a waiver: each is asserted to emit nothing below.
NON_SEMANTIC_WRITERS = {
    # Copies already-existing rows into a rebuilt table during migration.
    "_rebuild_drifted_tables",
    # Rewrites the ``kind`` of existing rows in place during migration
    # (the one-shot legacy-kind rename pass).
    "_migrate_add_optional_columns",
    # Row loss surfaces — deletion is not an event.
    "delete_archived_task",
    "delete_task",
    "gc_events",
}


def production_python_files():
    """Every production ``.py`` file — paths with a ``tests`` component excluded."""
    out = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        parts = set(rel.parts)
        if "tests" in parts or "tests-js" in parts:
            continue
        if any(p in parts for p in (".git", "node_modules", "venv", ".venv", "build")):
            continue
        out.append(path)
    return out


def parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return None


def enclosing_functions(tree):
    """Return ``(start, end, name)`` for every function in *tree*."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    return spans


def owner_of(spans, lineno):
    best = None
    for start, end, name in spans:
        if start <= lineno <= end and (best is None or start > best[0]):
            best = (start, name)
    return best[1] if best else None


def rebuild_spec_functions(tree):
    """Functions that range over ``_REBUILD_SPECS``.

    Only inside these is it meaningful to resolve an interpolated table name
    against the spec — rendering every f-string in the tree against every spec
    table would manufacture matches that the source cannot produce.
    """
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        }
        if "_REBUILD_SPECS" in names:
            out.add(node.name)
    return out


def sql_strings(tree, spec_functions=frozenset(), spans=()):
    """Yield ``(lineno, sql_text, is_dynamic)`` for every SQL-ish string.

    Three passes, because a literal-only scan under-reports the closure:

    1. every plain string constant (this also covers the literal fragments of
       an f-string);
    2. each f-string's literal parts joined with a placeholder, so a phrase
       that is *not* split by interpolation is still seen;
    3. inside functions that range over ``_REBUILD_SPECS`` only, each f-string
       rendered once per spec table — which is what makes
       ``INSERT INTO {table} ... FROM {table}_legacy`` visible.
    """
    tables = sorted(kb._REBUILD_SPECS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value, False
        elif isinstance(node, ast.JoinedStr):
            parts = [
                v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                else "\x00"
                for v in node.values
            ]
            yield node.lineno, "".join(parts), False
            if owner_of(spans, node.lineno) in spec_functions:
                for table in tables:
                    yield node.lineno, "".join(
                        table if p == "\x00" else p for p in parts
                    ), True


# --------------------------------------------------------------------------
# The seam is the only semantic new-event writer
# --------------------------------------------------------------------------


def test_c1n_t15_only_the_seam_inserts_new_task_event_rows():
    """Every semantic new-event INSERT lives in ``_append_event``.

    The five bundled-dashboard direct inserts were routed through the seam, so
    no production module outside it constructs one any more.
    """
    offenders = []
    dynamic_hits = []
    for path in production_python_files():
        tree = parse(path)
        if tree is None:
            continue
        spans = enclosing_functions(tree)
        specs = rebuild_spec_functions(tree)
        for lineno, sql, is_dynamic in sql_strings(tree, specs, spans):
            upper = " ".join(sql.upper().split())
            if "INSERT INTO TASK_EVENTS" not in upper:
                continue
            owner = owner_of(spans, lineno)
            rel = path.relative_to(REPO_ROOT)
            if is_dynamic:
                dynamic_hits.append((str(rel), lineno, owner))
                continue
            if not (path == KANBAN_DB and owner == SEAM):
                offenders.append((str(rel), lineno, owner))

    assert offenders == [], (
        "semantic new-event INSERTs must route through _append_event: "
        f"{offenders}"
    )
    # The scan must actually be able to see the interpolated rebuild INSERT —
    # otherwise this whole guard would silently under-report.
    assert dynamic_hits, "dynamic INSERT INTO {table} was not resolved by the scan"
    for _rel, _lineno, owner in dynamic_hits:
        assert owner in NON_SEMANTIC_WRITERS, (
            f"dynamically-built task_events INSERT in {owner!r} is unclassified"
        )


def test_c1n_t15_task_event_mutations_and_deletions_are_classified():
    """``UPDATE`` / ``DELETE`` on ``task_events`` create no event and emit nothing."""
    unclassified = []
    for path in production_python_files():
        tree = parse(path)
        if tree is None:
            continue
        spans = enclosing_functions(tree)
        specs = rebuild_spec_functions(tree)
        for lineno, sql, _dyn in sql_strings(tree, specs, spans):
            upper = " ".join(sql.upper().split())
            if not (
                "UPDATE TASK_EVENTS" in upper
                or "DELETE FROM TASK_EVENTS" in upper
            ):
                continue
            owner = owner_of(spans, lineno)
            if owner not in NON_SEMANTIC_WRITERS:
                unclassified.append((str(path.relative_to(REPO_ROOT)), lineno, owner))
    assert unclassified == [], (
        "every task_events mutation/deletion must be a classified non-semantic "
        f"writer: {unclassified}"
    )


def test_c1n_t15_non_semantic_writers_never_call_the_seam():
    """A migration copy or an in-place rewrite must install no frame and emit nothing."""
    tree = ast.parse(KANBAN_DB.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in NON_SEMANTIC_WRITERS:
            continue
        # ``init_db`` legitimately calls other helpers; only the rebuild/rename
        # migration writers must be free of the seam, and none of these five
        # may reach it directly.
        calls = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert SEAM not in calls, f"{node.name} must not emit task events"


def test_c1n_t15_rebuild_runs_outside_write_txn():
    """The rebuild owns a raw ``BEGIN IMMEDIATE``; it never enters ``write_txn``."""
    tree = ast.parse(KANBAN_DB.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_rebuild_drifted_tables"
    )
    with_calls = [
        item.context_expr for node in ast.walk(fn)
        if isinstance(node, ast.With) for item in node.items
    ]
    for expr in with_calls:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            assert expr.func.id != "write_txn"
    raw = [
        s for _l, s, _d in sql_strings(fn)
        if "BEGIN IMMEDIATE" in s.upper()
    ]
    assert raw, "expected the rebuild's raw BEGIN IMMEDIATE"


# --------------------------------------------------------------------------
# Every emitted kind is classified and declared
# --------------------------------------------------------------------------


def append_event_sites():
    """``(path, lineno, owner, kind_node)`` for every production seam call.

    Matches BOTH call forms — bare ``_append_event(...)`` (``ast.Name``, how
    ``kanban_db`` calls itself) and ``<mod>._append_event(...)``
    (``ast.Attribute``, how ``kanban_swarm`` and the bundled dashboard call
    it). A ``ast.Name``-only match makes an importing writer invisible to
    every guard built on this helper; see the C1N-T41 section.
    """
    sites = []
    for path in production_python_files():
        tree = parse(path)
        if tree is None:
            continue
        spans = enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name != SEAM:
                continue
            kind = node.args[2] if len(node.args) >= 3 else None
            sites.append((path, node.lineno, owner_of(spans, node.lineno), kind))
    return sites


def test_c1n_t15_every_seam_call_is_classified_and_declared():
    """Literal kinds must be declared; dynamic kinds must have a resolved domain.

    No assertion on how many call sites there are — that count is evidence, not
    API, and adding a writer must not break this guard.
    """
    sites = append_event_sites()
    assert sites, "the AST scan found no _append_event call sites at all"

    dynamic = []
    for path, lineno, owner, kind in sites:
        rel = path.relative_to(REPO_ROOT)
        assert kind is not None, f"{rel}:{lineno} passes no positional kind"
        if isinstance(kind, ast.Constant) and isinstance(kind.value, str):
            assert kind.value in kb.KANBAN_EVENT_KINDS, (
                f"{rel}:{lineno} emits undeclared kind {kind.value!r}"
            )
        else:
            dynamic.append((rel, lineno, owner, ast.unparse(kind)))

    # Each dynamic site's domain is resolved from its own function, not assumed.
    resolved = {owner for _r, _l, owner, _e in dynamic}
    assert resolved <= {"detect_crashed_workers", "_record_task_failure"}, (
        f"unclassified dynamic kind expression: {dynamic}"
    )


def test_c1n_t15_crash_classifier_domain_is_declared():
    """Every string ``detect_crashed_workers`` assigns to ``event_kind`` is declared."""
    tree = ast.parse(KANBAN_DB.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "detect_crashed_workers"
    )
    values = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "event_kind":
                    assert isinstance(node.value, ast.Constant), (
                        "event_kind must stay a fixed identifier"
                    )
                    values.add(node.value.value)
    assert values, "crash classifier assigned no event_kind"
    for value in values:
        assert value in kb.KANBAN_EVENT_KINDS, f"{value!r} is emitted but undeclared"


def test_c1n_t15_dynamic_outcome_domain_is_declared():
    """Every production caller that can reach the dynamic append is declared.

    ``_record_task_failure``'s dynamic append is gated on ``end_run=True``, and
    the writer closure reaches beyond ``kanban_db.py``: ``agent/turn_finalizer``
    calls it too (D-4/D-5).
    """
    reaching = set()
    for path in production_python_files():
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in ("_record_task_failure", "_record_spawn_failure"):
                continue
            kwargs = {
                kw.arg: kw.value for kw in node.keywords if kw.arg is not None
            }
            outcome = kwargs.get("outcome")
            end_run = kwargs.get("end_run")
            # _record_spawn_failure is the end_run=True alias.
            reaches = name == "_record_spawn_failure" or (
                isinstance(end_run, ast.Constant) and end_run.value is True
            )
            if reaches and isinstance(outcome, ast.Constant):
                reaching.add(outcome.value)
    assert reaching, "no production caller reaching the dynamic append was found"
    for value in reaching:
        assert value in kb.KANBAN_EVENT_KINDS, f"{value!r} is emitted but undeclared"


def test_c1n_t15_writer_closure_includes_modules_outside_kanban_db():
    """The closure is not confined to ``kanban_db.py`` (D-5)."""
    finalizer = REPO_ROOT / "agent" / "turn_finalizer.py"
    tree = parse(finalizer)
    assert tree is not None
    names = {
        getattr(n.func, "attr", None) or getattr(n.func, "id", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "_record_task_failure" in names or "_record_spawn_failure" in names


def test_c1n_t15_declaration_covers_the_dashboard_writers():
    """The bundled dashboard's kinds are declared, and it adds no second notifier."""
    dash = REPO_ROOT / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    tree = parse(dash)
    assert tree is not None
    kinds = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = getattr(node.func, "attr", None)
        if target != SEAM:
            continue
        kind = node.args[2] if len(node.args) >= 3 else None
        assert isinstance(kind, ast.Constant), "dashboard kinds must be literals"
        kinds.add(kind.value)
    assert kinds, "the dashboard no longer reaches the shared seam"
    for value in kinds:
        assert value in kb.KANBAN_EVENT_KINDS
    # No dashboard-local observer logic: it must not invoke the hook itself.
    source = dash.read_text()
    assert "kanban_task_event" not in source
    assert "invoke_hook" not in source


def test_c1n_t15_capture_path_never_consults_ambient_board():
    """``get_current_board()`` is forbidden anywhere in the C-1 capture path (D-17)."""
    tree = ast.parse(KANBAN_DB.read_text())
    capture_path = {
        "_append_event",
        "write_txn",
        "_resolve_board_for_connection",
        "_dispatch_task_events",
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in capture_path:
            continue
        names = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "get_current_board" not in names, (
            f"{node.name} must resolve board from the connection, not ambient state"
        )


# --------------------------------------------------------------------------
# C1N-T26 / C1N-T15 — every seam call site carries its §6.6 classification
# --------------------------------------------------------------------------

PLUGIN_API = REPO_ROOT / "plugins" / "kanban" / "dashboard" / "plugin_api.py"

#: The packet's §6.6 matrix, transcribed as a CLASSIFICATION map — never a
#: count. Each entry is ``(function, kind) -> how status_to is supplied``:
#:
#:   ``literal:<status>``  a fixed VALID_STATUSES member from the writer
#:   ``local:<name>``      the writer's own local transition scalar
#:   ``conditional``       gated on a captured rowcount (the D-9 writers)
#:   ``omitted``           the writer establishes no status
#:
#: Behavioural tests in ``test_kanban_generic_event_hook.py`` drive these rows;
#: this guard is what makes the matrix *exhaustive*. A new writer, or a writer
#: whose status source changes, must be classified here rather than silently
#: joining the seam — which is the failure mode a per-writer behavioural test
#: cannot catch on its own. It freezes no magnitude: not the site count, not
#: the vocabulary size, not a line number.
STATUS_TO_MATRIX = {
    ("kanban_db.py", "create_task", "created"): "local:task_status",
    ("kanban_db.py", "assign_task", "assigned"): "omitted",
    ("kanban_db.py", "set_model_override", "model_override_set"): "omitted",
    ("kanban_db.py", "set_reasoning_effort", "reasoning_effort_set"): "omitted",
    ("kanban_db.py", "link_tasks", "linked"): "conditional",
    ("kanban_db.py", "unlink_tasks", "unlinked"): "omitted",
    ("kanban_db.py", "add_comment", "commented"): "omitted",
    ("kanban_db.py", "add_attachment", "attached"): "omitted",
    ("kanban_db.py", "delete_attachment", "attachment_removed"): "omitted",
    ("kanban_db.py", "recompute_ready", "promoted"): "conditional",
    ("kanban_db.py", "claim_task", "claim_rejected"): "conditional",
    ("kanban_db.py", "claim_task", "claimed"): "literal:running",
    ("kanban_db.py", "claim_review_task", "claimed"): "literal:running",
    # Review-lane gating: a review card whose parents were reopened is demoted
    # back to todo, and the event is only emitted when that UPDATE matched.
    ("kanban_db.py", "claim_review_task", "dependency_wait"): "literal:todo",
    ("kanban_db.py", "release_stale_claims", "claim_extended"): "omitted",
    # The reclaim/timeout/crash writers restore the claimed SOURCE phase
    # (``review`` for a review-lane run), so they report the local
    # ``retry_status`` rather than an assumed 'ready'.
    ("kanban_db.py", "release_stale_claims", "reclaimed"): "local:retry_status",
    ("kanban_db.py", "reclaim_task", "reclaimed"): "local:retry_status",
    ("kanban_db.py", "complete_task", "completion_blocked_hallucination"): "omitted",
    ("kanban_db.py", "complete_task", "completed"): "literal:done",
    ("kanban_db.py", "complete_task", "suspected_hallucinated_references"): "omitted",
    ("kanban_db.py", "_insert_completion_attachment", "attached"): "omitted",
    ("kanban_db.py", "_maybe_emit_scratch_tip", "tip_scratch_workspace"): "omitted",
    ("kanban_db.py", "edit_completed_task_result", "edited"): "omitted",
    ("kanban_db.py", "block_task", "dependency_wait"): "literal:todo",
    ("kanban_db.py", "block_task", "block_loop_detected"): "literal:triage",
    ("kanban_db.py", "block_task", "blocked"): "literal:blocked",
    ("kanban_db.py", "promote_task", "promoted_manual"): "literal:ready",
    ("kanban_db.py", "unblock_task", "unblocked"): "local:new_status",
    # Review-lane transitions. Each sits behind a guarded UPDATE that returns
    # early unless it matched exactly one row.
    ("kanban_db.py", "request_review", "review_requested"): "literal:review",
    ("kanban_db.py", "request_changes", "changes_requested"): "local:new_status",
    ("kanban_db.py", "reopen_review_task", "review_reopened"): "local:new_status",
    # Done-reopen descendant retraction. Both events come from the same writer
    # and the same UPDATE, so both carry the status it established.
    (
        "kanban_db.py", "invalidate_descendants_for_parent_reopen",
        "descendant_invalidated",
    ): "local:demoted_to",
    (
        "kanban_db.py", "invalidate_descendants_for_parent_reopen", "status",
    ): "local:demoted_to",
    ("kanban_db.py", "specify_triage_task", "specified"): "literal:todo",
    ("kanban_db.py", "decompose_triage_task", "created"): "literal:todo",
    ("kanban_db.py", "decompose_triage_task", "linked"): "omitted",
    ("kanban_db.py", "decompose_triage_task", "decomposed"): "literal:todo",
    ("kanban_db.py", "archive_task", "archived"): "literal:archived",
    ("kanban_db.py", "schedule_task", "scheduled"): "literal:scheduled",
    ("kanban_db.py", "_defer_reclaim_for_live_worker", "reclaim_deferred"): "omitted",
    ("kanban_db.py", "heartbeat_worker", "heartbeat"): "omitted",
    ("kanban_db.py", "enforce_max_runtime", "timed_out"): "local:retry_status",
    ("kanban_db.py", "detect_stale_running", "stale"): "local:retry_status",
    # Still a literal: reconcile's UPDATE hard-codes 'ready'.
    ("kanban_db.py", "reconcile_orphaned_running", "reconciled"): "literal:ready",
    # The two dynamic-kind writers. Their kinds are bounded domains, asserted
    # separately by the declaration guards above.
    ("kanban_db.py", "detect_crashed_workers", "<dynamic>"): "local:retry_status",
    ("kanban_db.py", "_record_task_failure", "gave_up"): "conditional",
    ("kanban_db.py", "_record_task_failure", "<dynamic>"): "conditional",
    ("kanban_db.py", "_set_worker_pid", "spawned"): "omitted",
    ("kanban_db.py", "_dispatch_once_locked", "assigned"): "omitted",
    ("kanban_db.py", "_dispatch_once_locked", "respawn_guarded"): "omitted",
    # Review-lane respawn guard. Like its implementer-lane twin it only notes
    # why a spawn was withheld; it establishes no status.
    ("kanban_db.py", "_dispatch_once_locked", "respawn_guarded#2"): "omitted",
    # The four bundled-dashboard writers, now on the shared seam. The former
    # ``status#2`` (reopened-parent child demotion) moved into the domain
    # layer's ``invalidate_descendants_for_parent_reopen``.
    ("plugin_api.py", "update_task", "reprioritized"): "omitted",
    ("plugin_api.py", "update_task", "edited"): "omitted",
    # ``effective_status`` is what the guarded UPDATE wrote — the requested
    # status may have been re-gated to something else.
    ("plugin_api.py", "_set_status_direct", "status"): "local:effective_status",
    ("plugin_api.py", "bulk_update", "reprioritized"): "omitted",
}


def _classify_status_to(call):
    """Describe how one seam call supplies ``status_to``."""
    for kw in call.keywords:
        if kw.arg != "status_to":
            continue
        value = kw.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return f"literal:{value.value}"
        if isinstance(value, ast.IfExp):
            return "conditional"
        if isinstance(value, ast.Name):
            return f"local:{value.id}"
        return f"other:{ast.unparse(value)}"
    return "omitted"


def _seam_calls(path):
    """Yield ``((file, function, kind), classification, call_node)`` per site.

    A second call in the same function emitting the same kind is disambiguated
    with a ``#2`` suffix so both remain individually classified.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans = enclosing_functions(tree)
    seen = {}
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None
        )
        if name != SEAM:
            continue
        calls.append(node)
    for node in sorted(calls, key=lambda n: n.lineno):
        kind_node = node.args[2] if len(node.args) > 2 else None
        if isinstance(kind_node, ast.Constant) and isinstance(kind_node.value, str):
            kind = kind_node.value
        else:
            kind = "<dynamic>"
        key = (path.name, owner_of(spans, node.lineno), kind)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = (key[0], key[1], f"{key[2]}#{seen[key]}")
        yield key, _classify_status_to(node), node


def test_c1n_t26_every_seam_call_site_is_classified_by_the_status_matrix():
    """Exhaustive §6.6 coverage: no writer reaches the seam unclassified.

    Asserts a relationship (which writer supplies ``status_to`` and how), not a
    magnitude. Adding a writer without a matrix row — or changing where an
    existing one gets its status from — fails here, which is precisely the
    regression a per-writer behavioural test cannot see.
    """
    observed = {}
    for path in (KANBAN_DB, PLUGIN_API):
        for key, classification, _node in _seam_calls(path):
            assert key not in observed, f"ambiguous seam key {key}"
            observed[key] = classification

    missing = sorted(set(STATUS_TO_MATRIX) - set(observed))
    extra = sorted(set(observed) - set(STATUS_TO_MATRIX))
    assert not extra, f"unclassified seam writer(s): {extra}"
    assert not missing, f"matrix row(s) with no seam call site: {missing}"

    wrong = {
        k: (STATUS_TO_MATRIX[k], observed[k])
        for k in STATUS_TO_MATRIX
        if STATUS_TO_MATRIX[k] != observed[k]
    }
    assert not wrong, f"status_to source changed: {wrong}"


def test_c1n_t26_literal_status_to_values_are_valid_statuses():
    """A literal ``status_to`` is always a closed ``VALID_STATUSES`` member.

    ``_append_event`` drops anything else, so a typo would silently become an
    omission rather than a visible failure.
    """
    literals = {
        v.split(":", 1)[1]
        for v in STATUS_TO_MATRIX.values() if v.startswith("literal:")
    }
    assert literals, "expected at least one literal status writer"
    for status in literals:
        assert status in kb.VALID_STATUSES, f"{status} is not a valid status"


def test_c1n_t27_every_conditional_status_to_is_gated_on_a_rowcount():
    """D-9: a conditional ``status_to`` is gated on a *captured rowcount*.

    The four unchecked writers the packet names each ran a ``WHERE``-guarded
    ``UPDATE`` and appended without inspecting ``cur.rowcount`` — several
    without even binding the cursor. This asserts the gate is the rowcount
    itself, not some other predicate that merely looks conditional.
    """
    conditional = 0
    for path in (KANBAN_DB, PLUGIN_API):
        for key, classification, node in _seam_calls(path):
            if classification != "conditional":
                continue
            conditional += 1
            expr = next(
                ast.unparse(kw.value)
                for kw in node.keywords if kw.arg == "status_to"
            )
            test = ast.unparse(
                next(
                    kw.value for kw in node.keywords if kw.arg == "status_to"
                ).test
            )
            assert "rowcount" in test or test.endswith("_rows == 1"), (
                f"{key} gates status_to on {test!r}, not on a captured rowcount"
            )
            assert expr.endswith("else None"), (
                f"{key} must omit status_to on a zero-row match, got {expr!r}"
            )
    # A relationship, not a count: at minimum the D-9 writers must be here.
    assert conditional >= 4


def test_c1n_t26_failure_count_is_extracted_at_exactly_the_two_failure_writers():
    """§6.7: the single payload-derived value is confined to the two writers
    that already hold it as a bounded local integer."""
    holders = set()
    for path in (KANBAN_DB, PLUGIN_API):
        for key, _classification, node in _seam_calls(path):
            if any(kw.arg == "failure_count" for kw in node.keywords):
                holders.add(key)
    assert holders == {
        ("kanban_db.py", "_record_task_failure", "gave_up"),
        ("kanban_db.py", "_record_task_failure", "<dynamic>"),
    }, f"failure_count escaped its two writers: {sorted(holders)}"


# --------------------------------------------------------------------------
# C1N-T41 — the seam scan must see BOTH call forms
#
# ``append_event_sites`` above matches only bare ``_append_event(...)``
# (``ast.Name``). A writer that imports the module and calls
# ``kb._append_event(...)`` is an ``ast.Attribute`` and was invisible to every
# guard built on it. These tests close that hole and keep it closed.
#
# No test in this section asserts a frozen number of call sites: adding a
# legitimate writer must never break the guard, it must only be classified.
# --------------------------------------------------------------------------

#: Production modules that are allowed to reach the seam. This is a
#: CLASSIFICATION, not a budget — a new module here is a deliberate review
#: decision, and its call sites still have to satisfy every other guard.
SEAM_CALLER_MODULES = {
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_swarm.py",
    "plugins/kanban/dashboard/plugin_api.py",
}


def all_seam_call_sites():
    """``(rel, lineno, owner, call_node)`` for EVERY production seam call.

    Recognizes both ``_append_event(...)`` (``ast.Name``) and
    ``<mod>._append_event(...)`` (``ast.Attribute``).
    """
    sites = []
    for path in production_python_files():
        tree = parse(path)
        if tree is None:
            continue
        spans = enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name != SEAM:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            sites.append((rel, node.lineno, owner_of(spans, node.lineno), node))
    return sites


def test_c1n_t41_attribute_style_seam_calls_are_visible_to_the_scan():
    """The attribute-aware scan sees the swarm writer, and every module is known."""
    sites = all_seam_call_sites()
    assert sites, "the attribute-aware scan found no seam call sites at all"

    modules = {rel for rel, _l, _o, _n in sites}
    assert "hermes_cli/kanban_swarm.py" in modules, (
        "the swarm writer calls kb._append_event and must be scanned"
    )
    unclassified = modules - SEAM_CALLER_MODULES
    assert unclassified == set(), (
        f"unclassified production module reaches the seam: {sorted(unclassified)}"
    )


def test_c1n_t41_classification_is_exhaustive_over_both_call_forms():
    """No seam call may escape the kind-declaration guard by its call form.

    Every site the attribute-aware scan finds must pass a literal declared
    kind, or be one of the two already-classified dynamic-kind writers.
    """
    dynamic_owners = set()
    for rel, lineno, owner, node in all_seam_call_sites():
        kind = node.args[2] if len(node.args) >= 3 else None
        assert kind is not None, f"{rel}:{lineno} passes no positional kind"
        if isinstance(kind, ast.Constant) and isinstance(kind.value, str):
            assert kind.value in kb.KANBAN_EVENT_KINDS, (
                f"{rel}:{lineno} emits undeclared kind {kind.value!r}"
            )
        else:
            dynamic_owners.add(owner)
    assert dynamic_owners <= {"detect_crashed_workers", "_record_task_failure"}, (
        f"unclassified dynamic kind expression in {sorted(dynamic_owners)}"
    )


def test_c1n_t41_swarm_root_completion_declares_its_status():
    """The swarm's inline root completion must pass ``status_to='done'``.

    ``_activate_root_inline`` performs a guarded ``blocked -> done`` CAS and
    returns False unless it matched exactly one row, so by the time it appends
    its event it has honestly established ``done`` — exactly the condition
    ``status_to`` exists to report. Being invisible to the scan is why this
    writer alone never carried it.
    """
    tree = ast.parse((REPO_ROOT / "hermes_cli" / "kanban_swarm.py").read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_activate_root_inline"
    )
    calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) == SEAM
    ]
    assert len(calls) == 1, "expected exactly one seam call in _activate_root_inline"
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords if kw.arg is not None}
    status_to = kwargs.get("status_to")
    assert isinstance(status_to, ast.Constant) and status_to.value == "done", (
        "the swarm root completion establishes 'done' and must report it"
    )


def test_c1n_t41_the_kind_declaration_guard_sees_every_seam_module():
    """``append_event_sites`` must not under-report by call form.

    Every other kind/declaration guard in this file is built on
    ``append_event_sites``. If that helper matches only ``ast.Name`` calls,
    a module that does ``import kanban_db as kb; kb._append_event(...)``
    silently satisfies all of them by being invisible — which is exactly how
    the swarm writer's missing ``status_to`` survived review.
    """
    scanned = {rel for rel, _l, _o, _n in all_seam_call_sites()}
    guarded = {
        str(path.relative_to(REPO_ROOT))
        for path, _l, _o, _k in append_event_sites()
    }
    missed = scanned - guarded
    assert missed == set(), (
        "these modules reach the seam but escape the declaration guard: "
        f"{sorted(missed)}"
    )
