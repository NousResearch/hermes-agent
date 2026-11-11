"""#47 Phase 2 — CROSS-MODE INTEGRATION tests on the COMBINED implementation.

The unit suite (``tests/gateway/test_session_kanban_ownership.py``, 9/9) proves
the ledger and the Kanban admission gate in isolation. This module composes
them — ledger + ``gateway/session_kanban.py::admit_claim`` — and drives the
result end to end against a private ``tmp_path`` board, plus REAL production
card data (frozen snapshot + read-only cross-check).

The Direct path has no hook wired in this phase (RFC v2 §2 hook is not yet
implemented), so Direct-mode tests call the mode-agnostic ledger API
(:func:`cross_session_ownership.claim_for_dispatch` with ``mode='direct'``).
That is the same primitive the not-yet-wired hook would call.

Groups (L0 handoff matrix)
--------------------------
 01 Direct <-> Direct           : 2 sessions, same issue -> exactly 1 winner
 02 Kanban <-> Kanban           : 2 cards, same source issue -> exactly 1 winner
                                ; repo-less GH-n fails closed to kanban:<card_id> (C7)
 03 Direct <-> Kanban           : both directions conflict on the same key
 04 Concurrent claim races     : N threads -> exactly 1 winner, 0 double dispatch
 05 Batch nested-goal           : top-level goal != tasks[].goal -> tasks[] scanned
 06 Canonical GH-n on real cards: 5 production cards + canary (read-only cross-check)
 07 Still-live child, stale lease: safe takeover, no double execute
 08 Claim-lock fencing         : old owner denied heartbeat/release after takeover
 09 Refused-claim release       : card claim_lock NULL, status not running, no leak
 10 Nested transaction failure  : rollback on raise, refusal persists
 11 Restart / retry / 429       : idempotent, no duplicate rows
 12 Lock contention             : many claimants, contention detected
 13 Duplicate issue events      : no duplicate claim / dispatch
 14 Broken / missing ledger     : fail-closed
 15 REVIEW independent + phases : distinct phase scopes do not false-lock

Invariants asserted throughout
------------------------------
* exactly 1 ACTIVE conflicting execution owner per scope
* 0 duplicate dispatch / 0 duplicate ledger writes
* an independent REVIEW and a different phase of the same issue are allowed

Production safety: the production DB is opened ``mode=ro`` only, and its
SHA256 is asserted unchanged. Every other write lands in ``tmp_path``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from cross_session_ownership import (
    STATE_ACTIVE,
    STATE_RELEASED,
    VERDICT_OWNED,
    VERDICT_STALE,
    OwnershipError,
    canonical_key_from_kanban,
    claim_for_dispatch,
    claim_ownership,
    detect_contention,
    github_scope,
    heartbeat_ownership,
    init_schema,
    inspect_ownership,
    list_ownership,
    release_ownership,
    scope_key,
)
from hermes_state_runtime import RuntimeStoreError

# R1 (#47 Phase 2): import the recovery module at COLLECTION time, before any
# test patches ``hermes_state_local.local_receipt`` (see test_kanban_finite_admission).
import gateway.session_local_recovery as _slr


# --------------------------------------------------------------------------
# constants / helpers
# --------------------------------------------------------------------------
BOARD = 'xmode'
OWNER = 'owner-session'
OTHER = 'other-session'
BODY = 'source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 33'
SCOPE = 'gh:andrew288369/andrew-grp#33'
REPO_OWNER = 'Andrew288369'
REPO_NAME = 'ANDREW-GRP'

PROD_DB = Path(r'D:\ANDREW-GRP\_HERMES_HOME\kanban.db')

# Frozen 2026-11-10 read-only snapshot of the production card identity fields
# (title / body / idempotency_key) — the exact inputs ``canonical_key_from_kanban``
# reads. Used when the production DB is not present, and cross-checked against
# the live DB when it is.
PRODUCTION_SNAPSHOT = json.loads(r'''{
 "t_c1c41445": {
  "id": "t_c1c41445",
  "title": "PILOT: Verify Hermes Kanban profile dispatch",
  "body": "Read-only smoke test of Hermes Kanban dispatch. Confirm the worker is the aert-cto profile using its actual HERMES_HOME and SOUL path; inspect this task via kanban_show; report the actual model/provider/session evidence available to you and whether the task-scoped Kanban lifecycle works. Do not edit or create any files, do not change config, do not inspect credentials, do not dispatch other agents, do not access unrelated tasks, and do not claim anything you cannot verify. Complete this card with a concise structured result; if any prerequisite fails, block with the exact blocker.",
  "idempotency_key": "bta-hermes-kanban-pilot-20261026"
 },
 "t_7db9535f": {
  "id": "t_7db9535f",
  "title": "Thiết kế bộ sậu AI ChatGPT Work + Codex cho bạn của Mập",
  "body": "Mập chỉ đạo lưu vào backlog mức ưu tiên thấp nhất; chưa bắt đầu tự động, chỉ lôi ra khi thật sự rảnh hoặc Mập chủ động mở lại.\n\nMục tiêu: Tìm hiểu và thiết kế cho bạn của Mập một bộ sậu/quy trình làm việc dùng CẢ ChatGPT Work và Codex thường xuyên, lấy cảm hứng từ cách làm của Hermes nhưng không bê nguyên 5 AG x 7 ADR. Tham khảo advisory CAP-CODEX-TEAM-20261028-002 tại D:/ANDREW-GRP/_HERMES_HOME/profiles/aert-cto/cache/scratch/CAP-CODEX-TEAM-20261028-002/REPORT-CODEX-TEAM-20261028-002.md (tư vấn, chưa triển khai).\n\nKhi Mập kích hoạt lại: hỏi ngắn về lĩnh vực/công việc lặp lại, thiết bị/tài khoản/công cụ và quyền chia sẻ của người bạn; đề xuất vai trò Work vs Codex, hồ sơ chung, cách bàn giao/review/human approval, cách kiểm soát quota và quyền; lập Plan xin NEO + SAGE review trước khi có Brief/triển khai. Prototype 1 công việc thật, bounded, chỉ sau phê duyệt; kiểm tra handoff, independent QA, resume từ trạng thái lưu, giới hạn quyền và usage.\n\nKhông tự liên hệ người bạn; không đăng nhập tài khoản của họ, cài đặt hay sửa máy/cấu hình, tạo chi phí, publish hoặc chia sẻ dữ liệu riêng nếu chưa có phê duyệt riêng. Hiện chưa biết ngành nghề và nhu cầu cụ thể của người bạn — không suy đoán.\n\nParked intentionally: chờ Mập/NEO kích hoạt thủ công; không auto-dispatch. Assignee neo chỉ giữ ownership điều phối khi task được mở lại. Acceptance: phương án tối giản dựa trên nhu cầu đã xác nhận + demo end-to-end được kiểm chứng + giới hạn/quota/evidence nêu rõ.",
  "idempotency_key": "CAP-CODEX-WORKFLOW-FRIEND-20261028-001"
 },
 "t_97a8c77f": {
  "id": "t_97a8c77f",
  "title": "[GH-33 / P0 / READ-ONLY] Bridge preflight GitHub -> BTA (intake, dedup, dry-run)",
  "body": "# GH-33 / P0 / READ-ONLY — Bridge preflight GitHub → BTA\n\ntask_id: GH-33\nsource_issue: 33\nsource_repo: Andrew288369/ANDREW-GRP\npriority: P0\nowner: bta-coc (BTA, local commander)\nmode: READ_ONLY — no dispatch, no code change, no runtime change\nidempotency_key: GH-33\nmodel_override: meituan/longcat-2.5:free\nprovider_override: nous\nprovenance: L0 queue directive comment #6735105258 + coordination comment #6738136227 (2026-11-09)\ntransport_state: POSTED_TO_GITHUB; BRIDGE_UNVERIFIED; NOT SEEN_BY_BRIDGE / RECEIVED_BY_BTA / ACCEPTED / EXECUTED / VERIFIED\n\nScope:\n- Inspect existing Hermes Cron / gateway intake path and the current Delegate-profile BTA entrypoint.\n- Verify identity, permissions, least-privilege token, and fail-closed authorization.\n- Verify durable source_issue <-> kanban_task_id dedup ledger and event cursor/outbox.\n- One authenticated READ-ONLY dry-run intake test (no execution).\n- Confirm no second dispatcher/queue is created.\n\nForbidden:\n- No kanban dispatch, no worker spawn, no delegate call.\n- No code review execution, no source edits, no tests/diagnostics run.\n- No cron changes, no merge/push/deploy, no Windows snapshot collection.\n- No model/provider change beyond the pinned override on this card.\n\nAcceptance:\n- Report actual state with source SHA, opaque local receipt id, redacted evidence, blockers, next safe action.\n- If bridge missing -> BLOCKED with exact reason; do not fabricate delivery.\n\nGate: BTA preflight -> NEO process gate -> SAGE strategic gate (if long-term change) -> EMA audit. No AI self-approval.\n",
  "idempotency_key": "GH-33"
 },
 "t_468defb4": {
  "id": "t_468defb4",
  "title": "[GH-40 / P1 / READ-ONLY] AERT static review Pipeline Board (3 findings) + NEO/SAGE boundary + EMA evidence",
  "body": "# GH-40 / P1 / READ-ONLY — Pipeline Board static audit\n\ntask_id: GH-40\nsource_issue: 40\nsource_repo: Andrew288369/ANDREW-GRP\npriority: P1\nowner: aert-cto (AERT)\nmode: READ_ONLY — static review only; no code change, no merge\nidempotency_key: GH-40\nmodel_override: meituan/longcat-2.5:free\nprovider_override: nous\ndependency: GH-33 (parent) — must stay blocked until GH-33 preflight passes\nprovenance: L0 queue directive #6735105258 + coordination #6738136227\n\nScope:\n- Validate Copilot's 3 findings on 20_PMO/dashboard-api + dashboard-ui at commit 0024c75e4bafbed157897782e04fa9c5ced3fb98:\n  1) duplicate tasks in /api/pipeline and /api/tasks\n  2) stats not serialized into pipeline payload\n  3) silent WebSocket exception swallow\n- NEO/SAGE check scope boundaries; EMA confirms evidence independently.\n\nForbidden: no automatic code changes, no merge, no deploy, no state-modifying tests.\n\nAcceptance: finding-by-finding verdict with file:line evidence (redacted) + remediation proposal awaiting L0. No ACCEPTED/EXECUTED claim without EMA.\n\nGate: BTA drafts remediation -> NEO/SAGE review constraints -> EMA evidence check -> L0 decision.\n",
  "idempotency_key": "GH-40"
 },
 "t_fb97fb93": {
  "id": "t_fb97fb93",
  "title": "[GH-39 / P2 / READ-ONLY] NEO/EMA review diagnostic script + redaction before data collection",
  "body": "# GH-39 / P2 / READ-ONLY — Windows diagnostics script + redaction review\n\ntask_id: GH-39\nsource_issue: 39\nsource_repo: Andrew288369/ANDREW-GRP\npriority: P2\nowner: neo (NEO) — EMA independent review\nmode: READ_ONLY — review only; no data collection yet\nidempotency_key: GH-39\nmodel_override: meituan/longcat-2.5:free\nprovider_override: nous\ndependency: GH-33 (parent) — must stay blocked until GH-33 preflight passes\nprovenance: L0 queue directive #6735105258 + coordination #6738136227\n\nScope:\n- Review tools/windows/Collect-ChatGPTDiagnostics.ps1 (commit 66b2dcdc2c0e2dab8175e52e87176d850636dda) for read-only behavior.\n- Confirm no access to secrets, cookies, browser profile, session token, or chat content.\n- Confirm output stays local under %USERPROFILE%\\Desktop\\ChatGPT-Diagnostics\\ and no raw logs/IP/username/paths are pushed.\n- NEO + EMA confirm data scope before any snapshot collection.\n\nForbidden: no snapshot collection, no reboot, no system config/driver/firewall/proxy change, no raw logs on GitHub.\n\nAcceptance: redacted scope verdict + owner/path on PC; HEALTHY/INCIDENT snapshots only after gate.\n\nGate: BTA Plan -> NEO Gate -> SAGE Strategic Gate (if long-term) -> EMA Audit. No AI self-approval.\n",
  "idempotency_key": "GH-39"
 },
 "p_dd4c4bd1": {
  "id": "p_dd4c4bd1",
  "title": "[TASK] canary #46",
  "body": "Source: gh-issue #46\n",
  "idempotency_key": "gh-issue:Andrew288369/ANDREW-GRP:46"
 }
}''')

# Expected canonical key per production card. Four derive a GH-n key from real
# identity fields; two fall back to their own card id (never collide with Direct).
PRODUCTION_EXPECTED = {
    'p_dd4c4bd1': 'gh:andrew288369/andrew-grp#46',   # legacy gh-issue:<repo>:<n>
    't_468defb4': 'gh:andrew288369/andrew-grp#40',   # body source_repo+source_issue
    't_7db9535f': 'kanban:t_7db9535f',                # fallback (no GH identity)
    't_97a8c77f': 'gh:andrew288369/andrew-grp#33',   # body source_repo+source_issue
    't_fb97fb93': 'gh:andrew288369/andrew-grp#39',   # body source_repo+source_issue
    't_c1c41445': 'kanban:t_c1c41445',                # fallback (pilot card)
}


# --- ledger helpers ---------------------------------------------------------
def _ledger(tmp_path, name='ledger.db'):
    """Private ownership ledger (sandbox copy, never the production DB)."""
    conn = sqlite3.connect(tmp_path / name, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA busy_timeout=30000')
    init_schema(conn)
    return conn


def _ledger_ro(path):
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _assert_single_owner(conn, scope, expected_owner=None):
    """Invariant: exactly 1 ACTIVE conflicting execution owner for a scope."""
    rows = [r for r in list_ownership(conn, active_only=True) if r['scope_key'] == scope]
    assert len(rows) == 1, f'expected 1 active owner for {scope}, got {rows}'
    if expected_owner is not None:
        assert rows[0]['owner_session'] == expected_owner, rows[0]
    return rows[0]


def _assert_one_winner(results, expected_owner=None):
    """Invariant: 0 duplicate dispatch — exactly one caller won the scope."""
    winners = [r for r in results if r.get('ok')]
    assert len(winners) == 1, f'expected exactly 1 winner, got {len(winners)}: {results}'
    if expected_owner is not None:
        assert winners[0].get('owner') == expected_owner, winners[0]
    return winners[0]


def _gh_card(task_id, n, *, source_repo=REPO_OWNER + '/' + REPO_NAME, idem=None):
    return {
        'id': task_id,
        'title': f'[GH-{n} / P1] card {task_id}',
        'body': f'task_id: GH-{n}\nsource_issue: {n}\nsource_repo: {source_repo}\n'
                f'idempotency_key: GH-{n}\n',
        'idempotency_key': idem if idem is not None else f'GH-{n}',
    }


# --- Kanban board / run_task helpers (private tmp_path board) ---------------
def _prepare_board(tmp_path, monkeypatch, *, body=BODY, consecutive_failures=0):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    kb.create_board(BOARD)
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        task_id = kb.create_task(conn, title='Owned', assignee='default', workspace_kind='dir',
                                 workspace_path=str(workspace), body=body)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        if consecutive_failures:
            conn.execute('UPDATE tasks SET consecutive_failures=? WHERE id=?',
                         (consecutive_failures, task_id))
            task = kb.get_task(conn, task_id)
    return workspace, task


def _run_task(tmp_path, task, monkeypatch, *, session_id=OWNER, enforce='1'):
    from gateway.session_contract import Principal
    from gateway.session_kanban import run_task
    if enforce is None:
        monkeypatch.delenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', raising=False)
    else:
        monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', enforce)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    monkeypatch.setattr('gateway.session_local.create_local_session',
                        lambda *a, **k: SimpleNamespace(session_id=session_id, profile_id=str(tmp_path)))
    monkeypatch.setattr('gateway.session_kanban.asdict', lambda receipt: {})
    calls = []

    async def submit(actor, submission, **kwargs):
        calls.append(('submit', submission))
        return SimpleNamespace(admission_id='a')

    async def resume(ref, params):
        calls.append(('resume', ref))

    # The session layer is mocked: ``get_session`` must resolve a REAL session
    # (the gate's ``verify_session`` checks exactly that), and ``local_receipt``
    # must not return a saved claim or the retry branch would replace the fresh
    # admission under test.
    monkeypatch.setattr('hermes_state_local.local_receipt', lambda db, sid: None)
    monkeypatch.setattr(_slr, 'local_receipt', lambda db, sid: None)
    db = SimpleNamespace(get_session=lambda sid: True, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board=BOARD, task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    return run_task, connection, params, calls


def _pre_own(tmp_path, scope, session_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        return claim_ownership(conn, scope, session_id, ttl_seconds=900)


def _board_rows(tmp_path, task_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        events = [dict(r) for r in conn.execute(
            'SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id', (task_id,))]
        has_ownership = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_ownership'").fetchone()
        owners = [dict(r) for r in conn.execute('SELECT * FROM task_ownership')] if has_ownership else []
        run = conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1',
                           (task_id,)).fetchone()
    return task, events, owners, run


def _reclaim_card(tmp_path, task_id):
    """Re-claim a released card so a retry can be admitted (new run/lock)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        kb.recompute_ready(conn)
        return kb.claim_task(conn, task_id)


# --------------------------------------------------------------------------
# 01 — Direct <-> Direct
# --------------------------------------------------------------------------
def test_01_direct_vs_direct_same_issue_exactly_one_winner(tmp_path):
    conn = _ledger(tmp_path)
    a = claim_for_dispatch(conn, mode='direct', session_id='session-A', scope=SCOPE)
    b = claim_for_dispatch(conn, mode='direct', session_id='session-B', scope=SCOPE)
    _assert_one_winner([a, b], expected_owner='session-A')
    assert b['verdict'] == VERDICT_OWNED and b['owner'] == 'session-A'
    _assert_single_owner(conn, SCOPE, 'session-A')
    # the loser did not write a second row / a second dispatch
    assert [r['ok'] for r in (a, b)].count(True) == 1
    assert len(list_ownership(conn, active_only=False)) == 1


# --------------------------------------------------------------------------
# 02 — Kanban <-> Kanban
# --------------------------------------------------------------------------
def test_02_kanban_vs_kanban_same_source_issue_exactly_one_winner(tmp_path):
    conn = _ledger(tmp_path)
    card_a = _gh_card('t_a', 33)
    card_b = _gh_card('t_b', 33)
    # two distinct cards, one logical work item -> one canonical key
    assert canonical_key_from_kanban(card_a) == canonical_key_from_kanban(card_b) == SCOPE
    a = claim_for_dispatch(conn, mode='kanban', session_id='session-A', task=card_a)
    b = claim_for_dispatch(conn, mode='kanban', session_id='session-B', task=card_b)
    _assert_one_winner([a, b], expected_owner='session-A')
    assert b['verdict'] == VERDICT_OWNED and b['owner'] == 'session-A'
    _assert_single_owner(conn, SCOPE, 'session-A')
    # C7: a third card carrying ONLY a repo-less idempotency_key GH-33 must NOT
    # resolve to the shared GH key. Mapping it to an assumed DEFAULT_REPO made
    # GH-33 collide across repos (two repos' GH-33 -> one ledger row). It now
    # falls back to its own card id, fail-closed, with a diagnostic.
    card_c = {'id': 't_c', 'title': 'unrelated title', 'body': 'no source fields',
              'idempotency_key': 'GH-33'}
    diag: list[str] = []
    assert canonical_key_from_kanban(card_c, diagnostics=diag) == 'kanban:t_c'
    # the diagnostic chain records WHY it fell back and WHICH branch produced the key
    assert diag == ['fail_closed:gh_number_without_repo', 'card_id'], diag
    c = claim_for_dispatch(conn, mode='kanban', session_id='session-C', task=card_c)
    # It claims its OWN card scope: it neither conflicts with nor steals #33.
    assert c['ok'] is True and c['owner'] == 'session-C'
    assert len(list_ownership(conn, active_only=False)) == 2


def test_02b_repo_less_gh_number_with_explicit_repo_is_honoured(tmp_path):
    """C7 pins the explicit-only contract: a caller that genuinely knows the
    repo may pass ``default_repo``; the DEFAULT (no repo) never assumes one."""
    card = {'id': 't_d', 'title': 'unrelated title', 'body': 'no source fields',
            'idempotency_key': 'GH-33'}
    diag: list[str] = []
    assert canonical_key_from_kanban(card, default_repo='SomeOwner/some-repo',
                                     diagnostics=diag) == 'gh:someowner/some-repo#33'
    assert diag == ['gh_number_with_explicit_repo'], diag
    # same card, no explicit repo -> the card's own id, not an assumed repo
    assert canonical_key_from_kanban(card) == 'kanban:t_d'
    # a card WITH source_repo still derives the GH key from its own repo
    with_repo = {'id': 't_e', 'title': 'unrelated', 'body': 'source_repo: Other/repo-x',
                 'idempotency_key': 'GH-33'}
    assert canonical_key_from_kanban(with_repo) == 'gh:other/repo-x#33'


# --------------------------------------------------------------------------
# 03 — Direct <-> Kanban (both directions)
# --------------------------------------------------------------------------
def test_03_direct_vs_kanban_conflict_both_directions(tmp_path):
    # Direct first, then the Kanban card for the same issue is blocked.
    conn = _ledger(tmp_path, 'direct_first.db')
    d = claim_for_dispatch(conn, mode='direct', session_id='session-D', scope=SCOPE)
    k = claim_for_dispatch(conn, mode='kanban', session_id='session-K', task=_gh_card('t_k', 33))
    _assert_one_winner([d, k], expected_owner='session-D')
    assert k['verdict'] == VERDICT_OWNED and k['owner'] == 'session-D'
    _assert_single_owner(conn, SCOPE, 'session-D')

    # Kanban first, then the Direct dispatch for the same issue is blocked.
    conn2 = _ledger(tmp_path, 'kanban_first.db')
    k2 = claim_for_dispatch(conn2, mode='kanban', session_id='session-K2', task=_gh_card('t_k2', 33))
    d2 = claim_for_dispatch(conn2, mode='direct', session_id='session-D2', scope=SCOPE)
    _assert_one_winner([k2, d2], expected_owner='session-K2')
    assert d2['verdict'] == VERDICT_OWNED and d2['owner'] == 'session-K2'
    _assert_single_owner(conn2, SCOPE, 'session-K2')


# --------------------------------------------------------------------------
# 04 — Concurrent claim races
# --------------------------------------------------------------------------
def _race_claim(db_path, scope, n, *, mode='direct', task=None):
    """n threads claim the same scope at once; returns (results, winners)."""
    barrier = threading.Barrier(n)
    results = [None] * n
    lock = threading.Lock()
    winners = []

    def worker(i):
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            barrier.wait()
            if mode == 'kanban':
                r = claim_for_dispatch(conn, mode='kanban', session_id=f'session-{i}', task=task)
            else:
                r = claim_for_dispatch(conn, mode='direct', session_id=f'session-{i}', scope=scope)
            results[i] = r
            if r['ok']:
                with lock:
                    winners.append(i)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, winners


def test_04_concurrent_claims_exactly_one_winner(tmp_path):
    db = tmp_path / 'race.db'
    seed = _ledger(tmp_path, 'race.db')
    seed.close()
    results, winners = _race_claim(db, SCOPE, 8)
    _assert_one_winner(results)
    assert len(winners) == 1, f'double dispatch: {winners}'
    conn = _ledger_ro(db)
    _assert_single_owner(conn, SCOPE, f'session-{winners[0]}')
    # contention is visible in the audit trail (SAGE condition)
    found = [c for c in detect_contention(conn, window_seconds=300) if c['scope_key'] == SCOPE]
    assert found and found[0]['distinct_sessions'] >= 2, found
    conflicts = conn.execute(
        "SELECT COUNT(*) FROM task_ownership_events WHERE scope_key=? AND event='claim_conflict'",
        (SCOPE,)).fetchone()[0]
    assert conflicts >= 1


# --------------------------------------------------------------------------
# 05 — Batch nested-goal (F2 bypass guard)
# --------------------------------------------------------------------------
_ISSUE_REF_RE = re.compile(r'(?i)\b(?:gh[-_]?(\d+)|#(\d+))\b')


def _scopes_from_text(text):
    """Issue refs in a goal/context string -> canonical gh: scopes."""
    out = set()
    for m in _ISSUE_REF_RE.finditer(text or ''):
        number = m.group(1) or m.group(2)
        if number:
            out.add(github_scope(REPO_OWNER, REPO_NAME, int(number)))
    return out


def _scan_batch(batch, *, include_tasks=True):
    """RFC v2 §2: the UNION of top-level goal + context + every tasks[].goal/context.

    ``include_tasks=False`` reproduces the v1 defect (F2): a conflicting issue
    hidden inside ``tasks[]`` is never inspected.
    """
    texts = [batch.get('goal') or '', batch.get('context') or '']
    if include_tasks:
        for task in batch.get('tasks') or []:
            if isinstance(task, dict):
                texts.append(task.get('goal') or '')
                texts.append(task.get('context') or '')
    scopes = set()
    for text in texts:
        scopes |= _scopes_from_text(text)
    return sorted(scopes)


def _dispatch_batch(conn, session_id, batch, *, include_tasks=True):
    """All-or-nothing batch claim: inspect the union first, then claim.

    Any scope already owned by another live session blocks the whole batch
    before a single ledger write, so a refused batch cannot leak a partial
    claim.
    """
    scopes = _scan_batch(batch, include_tasks=include_tasks)
    conflicts = []
    for scope in scopes:
        ins = inspect_ownership(conn, scope)
        if ins['verdict'] == VERDICT_OWNED and ins['owner'] != session_id:
            conflicts.append({'scope': scope, 'owner': ins['owner']})
    if conflicts:
        return {'ok': False, 'reason': 'batch_scope_conflict', 'conflicts': conflicts,
                'scopes': scopes}
    for scope in scopes:
        res = claim_ownership(conn, scope, session_id)
        if not res['ok']:
            return {'ok': False, 'reason': res['reason'], 'scope': scope, 'scopes': scopes}
    return {'ok': True, 'scopes': scopes}


def test_05_batch_nested_goal_is_scanned_not_bypassed(tmp_path):
    batch = {
        'goal': 'Work Andrew288369/ANDREW-GRP#47 per directive',   # top-level: free
        'context': '',
        'tasks': [
            {'goal': 'Review Andrew288369/ANDREW-GRP#39', 'context': ''},
            {'goal': 'Implement Andrew288369/ANDREW-GRP#33', 'context': ''},  # nested conflict
        ],
    }
    # (i) prove the v1 top-level-only scan really does bypass the nested #33.
    naive_conn = _ledger(tmp_path, 'naive.db')
    assert claim_ownership(naive_conn, SCOPE, 'session-other', ttl_seconds=900)['ok']
    naive = _dispatch_batch(naive_conn, 'session-batch', batch, include_tasks=False)
    assert naive['ok'] is True, naive            # <-- F2 bypass, reproduced
    assert naive['scopes'] == ['gh:andrew288369/andrew-grp#47']

    # (ii) the RFC v2 scan of the SAME batch catches the nested conflict and
    #      writes nothing (all-or-nothing).
    full_conn = _ledger(tmp_path, 'full.db')
    assert claim_ownership(full_conn, SCOPE, 'session-other', ttl_seconds=900)['ok']
    full = _dispatch_batch(full_conn, 'session-batch', batch, include_tasks=True)
    assert full['ok'] is False and full['reason'] == 'batch_scope_conflict', full
    assert any(c['scope'] == SCOPE and c['owner'] == 'session-other' for c in full['conflicts'])
    # no partial claim leaked from the blocked batch
    rows = list_ownership(full_conn, active_only=True)
    assert len(rows) == 1 and rows[0]['owner_session'] == 'session-other'
    assert len(list_ownership(full_conn, active_only=False)) == 1


def test_05b_batch_without_nested_conflict_claims_every_scope(tmp_path):
    conn = _ledger(tmp_path)
    batch = {
        'goal': 'Work Andrew288369/ANDREW-GRP#33 per directive',
        'tasks': [{'goal': 'Also cover Andrew288369/ANDREW-GRP#39', 'context': ''}],
    }
    res = _dispatch_batch(conn, 'session-batch', batch, include_tasks=True)
    assert res['ok'] is True, res
    assert res['scopes'] == ['gh:andrew288369/andrew-grp#33', 'gh:andrew288369/andrew-grp#39']
    assert len(list_ownership(conn, active_only=True)) == 2


# --------------------------------------------------------------------------
# 06 — Canonical GH-n key on REAL production cards
# --------------------------------------------------------------------------
@pytest.mark.parametrize('task_id,expected', sorted(PRODUCTION_EXPECTED.items()))
def test_06_canonical_key_on_production_card(task_id, expected):
    card = PRODUCTION_SNAPSHOT[task_id]
    assert canonical_key_from_kanban(card) == expected


def test_06b_production_snapshot_matches_live_db_readonly():
    """The frozen snapshot must still match the live production cards, and the
    production DB must be byte-identical after the read."""
    if not PROD_DB.exists():
        pytest.skip('production kanban.db not present')
    before = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()
    conn = sqlite3.connect(f'file:{PROD_DB}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for task_id, expected in sorted(PRODUCTION_EXPECTED.items()):
            row = conn.execute(
                'SELECT id, title, body, idempotency_key FROM tasks WHERE id=?',
                (task_id,)).fetchone()
            if row is None:
                pytest.skip(f'production card {task_id} no longer present')
            live = {'id': row['id'], 'title': row['title'], 'body': row['body'],
                    'idempotency_key': row['idempotency_key']}
            assert canonical_key_from_kanban(live) == expected, task_id
            assert live['title'] == PRODUCTION_SNAPSHOT[task_id]['title'], task_id
            assert live['idempotency_key'] == PRODUCTION_SNAPSHOT[task_id]['idempotency_key'], task_id
    finally:
        conn.close()
    after = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()
    assert before == after, 'production kanban.db changed during the test'


def test_06c_direct_claim_collides_with_real_production_card(tmp_path):
    """The Direct key for #33 is the SAME key the real GH-33 card derives."""
    conn = _ledger(tmp_path)
    card = PRODUCTION_SNAPSHOT['t_97a8c77f']           # real production GH-33 card
    card_key = canonical_key_from_kanban(card)
    direct = claim_for_dispatch(conn, mode='direct', session_id='session-D', scope=card_key)
    assert direct['ok'] is True
    blocked = claim_for_dispatch(conn, mode='kanban', session_id='session-K', task=card)
    assert blocked['ok'] is False and blocked['owner'] == 'session-D'
    _assert_single_owner(conn, card_key, 'session-D')


# --------------------------------------------------------------------------
# 07 — Still-live child after a stale lease
# --------------------------------------------------------------------------
def test_07_still_live_child_after_stale_lease_takeover(tmp_path):
    conn = _ledger(tmp_path)
    t0 = 1_000_000
    assert claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0)['ok']
    # The child process is still running (liveness says ALIVE) but the lease is
    # gone: expiry is checked BEFORE liveness, so the verdict is STALE.
    alive = lambda _sid: 'ALIVE'
    ins = inspect_ownership(conn, SCOPE, liveness=alive, now=t0 + 61)
    assert ins['verdict'] == VERDICT_STALE, ins
    takeover = claim_ownership(conn, SCOPE, 'session-B', ttl_seconds=300,
                               liveness=alive, now=t0 + 61)
    assert takeover['ok'] is True and takeover['owner'] == 'session-B'
    assert takeover['reason'] == 'taken_over_from:session-A'
    # Fencing: the still-running old child can no longer heartbeat or release.
    assert heartbeat_ownership(conn, SCOPE, 'session-A', now=t0 + 62)['ok'] is False
    assert release_ownership(conn, SCOPE, 'session-A', now=t0 + 62)['ok'] is False
    # No double execute: exactly one session passes the execute-gate check.
    def _may_execute(session_id, now):
        v = inspect_ownership(conn, SCOPE, now=now)
        return v['verdict'] == VERDICT_OWNED and v['owner'] == session_id
    assert _may_execute('session-A', t0 + 62) is False
    assert _may_execute('session-B', t0 + 62) is True
    _assert_single_owner(conn, SCOPE, 'session-B')


# --------------------------------------------------------------------------
# 08 — Claim-lock fencing / takeover
# --------------------------------------------------------------------------
def test_08_claim_lock_fencing_after_takeover(tmp_path):
    conn = _ledger(tmp_path)
    t0 = 1_000_000
    assert claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0)['ok']
    lock_a = conn.execute('SELECT claim_lock FROM task_ownership WHERE scope_key=?',
                          (SCOPE,)).fetchone()[0]
    assert claim_ownership(conn, SCOPE, 'session-B', ttl_seconds=300, now=t0 + 61)['ok']
    lock_b = conn.execute('SELECT claim_lock FROM task_ownership WHERE scope_key=?',
                          (SCOPE,)).fetchone()[0]
    assert lock_a != lock_b, 'takeover must rotate the claim lock'
    # The old owner is refused by owner check AND by its stale lock reference.
    assert heartbeat_ownership(conn, SCOPE, 'session-A', expected_lock=lock_a,
                               now=t0 + 62)['ok'] is False
    assert release_ownership(conn, SCOPE, 'session-A', expected_lock=lock_a,
                             now=t0 + 62)['ok'] is False
    # The new owner's own lock works.
    assert heartbeat_ownership(conn, SCOPE, 'session-B', expected_lock=lock_b,
                               now=t0 + 62)['ok'] is True
    assert release_ownership(conn, SCOPE, 'session-B', expected_lock=lock_b,
                             now=t0 + 63)['ok'] is True
    row = conn.execute('SELECT * FROM task_ownership WHERE scope_key=?', (SCOPE,)).fetchone()
    assert row['state'] == STATE_RELEASED and row['owner_session'] == 'session-B'


# --------------------------------------------------------------------------
# 09 — Refused-claim release (Kanban gate E2E)
# --------------------------------------------------------------------------
def test_09_refused_claim_releases_card_and_does_not_leak(tmp_path, monkeypatch):
    _workspace, task = _prepare_board(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == [], 'a refused admission must not reach resume/submit'
    card, events, owners, run = _board_rows(tmp_path, task.id)
    # S2: the card claim is released — not left running, no leaked lock.
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None and card['current_run_id'] is None
    assert card['last_failure_error'] == 'ownership_refused'
    assert run['status'] == 'refused' and run['ended_at'] is not None
    assert [e['kind'] for e in events].count('owner_admitted') == 0
    # no leak: exactly one ledger row, still the foreign owner, still ACTIVE.
    assert len(owners) == 1 and owners[0]['owner_session'] == OTHER
    assert owners[0]['state'] == STATE_ACTIVE
    # a third session is still refused: the foreign owner was not disturbed.
    third = claim_ownership(_ledger_ro(_board_db(tmp_path)), SCOPE, 'session-C')
    # (separate connection, same file — the foreign owner row is intact)
    assert third['ok'] is False and third['owner'] == OTHER
    # the released card can be re-claimed and admitted by the OWNER session:
    # proves the refusal left no stale claim_lock behind.
    task2 = _reclaim_card(tmp_path, task.id)
    run_task2, connection2, params2, calls2 = _run_task(tmp_path, task2, monkeypatch,
                                                        session_id=OTHER)
    result = asyncio.run(run_task2(connection2, params2))
    assert result['session_id'] == OTHER
    assert [c[0] for c in calls2] == ['resume', 'submit']
    card2, events2, owners2, _ = _board_rows(tmp_path, task.id)
    assert card2['status'] == 'running'
    assert [e['kind'] for e in events2].count('owner_admitted') == 1
    assert len(owners2) == 1 and owners2[0]['owner_session'] == OTHER


def _board_db(tmp_path):
    """Resolve the private board DB path under the monkeypatched HERMES_KANBAN_HOME."""
    from hermes_cli.kanban_db import kanban_db_path
    return kanban_db_path(board=BOARD)


# --------------------------------------------------------------------------
# 10 — Nested transaction failure
# --------------------------------------------------------------------------
def test_10_nested_transaction_rollback_and_commit(tmp_path):
    # (a) join_txn=True without a caller transaction is fail-closed.
    conn = _ledger(tmp_path)
    with pytest.raises(OwnershipError):
        claim_ownership(conn, SCOPE, 'session-A', join_txn=True)
    assert list_ownership(conn, active_only=False) == []

    # (b) a caller transaction that raises after a joined claim rolls the claim back.
    conn2 = _ledger(tmp_path, 'ledger2.db')
    conn2.execute('BEGIN IMMEDIATE')
    assert claim_ownership(conn2, SCOPE, 'session-A', join_txn=True)['ok']
    conn2.execute('ROLLBACK')
    assert list_ownership(conn2, active_only=False) == []

    # (c) a caller transaction that commits keeps the claim.
    conn3 = _ledger(tmp_path, 'ledger3.db')
    conn3.execute('BEGIN IMMEDIATE')
    assert claim_ownership(conn3, SCOPE, 'session-A', join_txn=True)['ok']
    conn3.execute('COMMIT')
    _assert_single_owner(conn3, SCOPE, 'session-A')


def test_10b_kanban_gate_ownership_error_refusal_persists(tmp_path, monkeypatch):
    _workspace, task = _prepare_board(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise OwnershipError('ledger unavailable')

    monkeypatch.setattr('cross_session_ownership.claim_ownership', boom)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, run = _board_rows(tmp_path, task.id)
    # The refusal persists (it is not rolled away with the failed claim).
    assert card['status'] == 'ready' and card['claim_lock'] is None
    assert card['current_run_id'] is None
    assert run['status'] == 'refused'
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'ownership_txn_error'
    # No partial ledger row was left behind by the failed claim.
    assert owners == []


# --------------------------------------------------------------------------
# 11 — Restart / retry / 429
# --------------------------------------------------------------------------
def test_11_restart_retry_429_idempotent(tmp_path):
    conn = _ledger(tmp_path)
    t0 = 1_000_000
    assert claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0)['ok']
    # restart while ACTIVE: same session re-asserts, no second owner.
    again = claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0 + 10)
    assert again['ok'] is True and again['reason'] == 'already_owner'
    assert len(list_ownership(conn, active_only=False)) == 1
    # restart after expiry: same session takes over its own expired lease.
    # t0+70 is still inside the lease (claim_expires == now, and expiry is
    # `expires < now`), so the takeover must be asserted one second later.
    restarted = claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0 + 71)
    assert restarted['ok'] is True and restarted['reason'].startswith('taken_over_from:session-A')
    assert len(list_ownership(conn, active_only=False)) == 1
    # 429 burst: every attempt claims then releases; no duplicate rows.
    for i in range(5):
        assert claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60,
                               now=t0 + 100 + i)['ok']
        assert release_ownership(conn, SCOPE, 'session-A', reason='429',
                                 now=t0 + 100 + i)['ok']
    rows = list_ownership(conn, active_only=False)
    assert len(rows) == 1 and rows[0]['state'] == STATE_RELEASED
    # a fresh retry after the burst succeeds, still one row.
    assert claim_ownership(conn, SCOPE, 'session-A', ttl_seconds=60, now=t0 + 200)['ok']
    assert len(list_ownership(conn, active_only=False)) == 1


# --------------------------------------------------------------------------
# 12 — Lock contention
# --------------------------------------------------------------------------
def test_12_lock_contention_many_claimants(tmp_path):
    db = tmp_path / 'contention.db'
    _ledger(tmp_path, 'contention.db').close()
    results, winners = _race_claim(db, SCOPE, 6)
    _assert_one_winner(results)
    assert len(winners) == 1
    conn = _ledger_ro(db)
    _assert_single_owner(conn, SCOPE, f'session-{winners[0]}')
    found = [c for c in detect_contention(conn, window_seconds=300) if c['scope_key'] == SCOPE]
    assert found and found[0]['distinct_sessions'] >= 2, found
    assert found[0]['actors'] == sorted(found[0]['actors'])


# --------------------------------------------------------------------------
# 13 — Duplicate issue events
# --------------------------------------------------------------------------
def test_13_duplicate_issue_events_no_duplicate_claim(tmp_path):
    conn = _ledger(tmp_path)
    # the same GitHub issue event delivered twice (webhook retry)
    events = [{'id': 'evt-1', 'issue': 33}, {'id': 'evt-1', 'issue': 33}]
    dispatches = []
    for event in events:
        scope = github_scope(REPO_OWNER, REPO_NAME, event['issue'])
        res = claim_for_dispatch(conn, mode='direct', session_id='session-A', scope=scope)
        # A duplicate delivery re-asserts the SAME owner (reason='already_owner',
        # ok=True by design) — only a fresh claim is a dispatch.
        if res.get('reason') == 'claimed':
            dispatches.append(event['id'])
    assert dispatches == ['evt-1'], f'duplicate dispatch: {dispatches}'
    assert len(list_ownership(conn, active_only=True)) == 1
    # a genuinely different issue is not false-deduped.
    other = claim_for_dispatch(conn, mode='direct', session_id='session-A',
                               scope=github_scope(REPO_OWNER, REPO_NAME, 39))
    assert other['ok'] is True
    assert len(list_ownership(conn, active_only=True)) == 2


# --------------------------------------------------------------------------
# 14 — Broken / missing ledger
# --------------------------------------------------------------------------
def test_14_missing_ledger_fails_closed(tmp_path):
    db = tmp_path / 'empty.db'
    sqlite3.connect(db).close()          # no ledger tables at all
    conn = sqlite3.connect(db, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises((OwnershipError, sqlite3.OperationalError)):
            claim_for_dispatch(conn, mode='direct', session_id='session-A', scope=SCOPE)
        # fail-closed: nothing was claimed, no table was created by the caller.
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='task_ownership'").fetchone()[0] == 0
    finally:
        conn.close()


def test_14b_kanban_gate_broken_ledger_releases_card(tmp_path, monkeypatch):
    """A ledger whose schema is broken must fail closed AND release the card
    claim (S2): the refusal path runs inside the same transaction."""
    _workspace, task = _prepare_board(tmp_path, monkeypatch)
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        # incompatible pre-existing table: init_schema's IF NOT EXISTS is a no-op,
        # the claim then raises a non-OwnershipError sqlite3 error.
        conn.execute('CREATE TABLE task_ownership (scope_key TEXT)')
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(Exception):
        asyncio.run(run_task(connection, params))
    assert calls == [], 'a broken ledger must not dispatch'
    card, events, _owners, run = _board_rows(tmp_path, task.id)
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['status'] != 'running'
    assert card['current_run_id'] is None
    assert card['last_failure_error'] == 'ownership_refused'
    assert run['status'] == 'refused'
    assert [e['kind'] for e in events].count('owner_admitted') == 0


# --------------------------------------------------------------------------
# 15 — Independent REVIEW and distinct phases
# --------------------------------------------------------------------------
def test_15_independent_review_and_distinct_phases_allowed(tmp_path):
    conn = _ledger(tmp_path)
    assert claim_ownership(conn, SCOPE, 'session-exec', ttl_seconds=900)['ok']
    # An independent REVIEW of the same issue is a distinct phase scope and
    # must not be blocked by the execution owner.
    review = scope_key('gh', 'andrew288369/andrew-grp#33:review')
    reviewer = claim_ownership(conn, review, 'session-reviewer', ttl_seconds=900)
    assert reviewer['ok'] is True and reviewer['owner'] == 'session-reviewer'
    # Distinct phases of the same issue do not false-lock each other.
    phase_a = scope_key('gh', 'andrew288369/andrew-grp#33:phase-10a')
    phase_b = scope_key('gh', 'andrew288369/andrew-grp#33:phase-10b')
    assert claim_ownership(conn, phase_a, 'session-phase-a')['ok'] is True
    assert claim_ownership(conn, phase_b, 'session-phase-b')['ok'] is True
    # A different issue is untouched.
    assert claim_ownership(conn, github_scope(REPO_OWNER, REPO_NAME, 39),
                           'session-other-issue')['ok'] is True
    rows = list_ownership(conn, active_only=True)
    # SCOPE, REVIEW, phase_a, phase_b and the different issue #39 -> 5 owners.
    assert len(rows) == 5, rows
    owners = {r['scope_key']: r['owner_session'] for r in rows}
    assert owners[SCOPE] == 'session-exec'
    assert owners[review] == 'session-reviewer'
    # exactly one owner per scope, no false global lock
    for scope in (SCOPE, review, phase_a, phase_b):
        _assert_single_owner(conn, scope)
