"""S4+S5 — R4 배선 e2e: 실 sandbox-exec + implementer.sbpl + mock 워커 + mock 프록시.

flag off·라이브 무접촉(격리 DB + tmp workspace). 실 sandbox-exec를 호출하므로 darwin 한정.
검증: supervise가 _real_spawn_capture로 워커를 샌드박스에서 spawn→reap→capture→phase2→
제안서 조립까지 수행하고, mock codex=PASS면 done / 실 codex(_real_codex_review=BLOCKED)면
codex_review_passed 게이트에서 차단(R4 의도된 동작 = R5 전까지 done 불가).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import role_gate
from hermes_cli import m2_supervisor
from hermes_cli import m2_proxy_ctx
from hermes_cli import m2_sandbox

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="sandbox-exec는 macOS 전용")


WRITE_PLAN = """\
구현 계획 (implementer)
- 의존성: 입력 스펙 spec.md, 산출물 경로 proposal/MERGE_PROPOSAL.json
- 중단 조건: 선언 밖 write 시 중단한다. 위험 시 STOP 후 보고.
- 단계:
  1. [read] 입력 스펙 열람
  2. [write:source_staging/gen.py] 변경안 staging
  3. [write:proposal/MERGE_PROPOSAL.json] 제안서 작성
"""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_impl_task(conn, ws, deliverable="proposal/MERGE_PROPOSAL.json"):
    recipe = role_gate.gate_recipe_for_assignee("implementer", deliverable)
    return kb.create_task(conn, title="impl", assignee="implementer",
                          workspace_path=str(ws), gate_recipe=recipe)


# ---------------------------------------------------------------------------
# S4 — mock proxy ctx
# ---------------------------------------------------------------------------
def test_mock_proxy_ctx_yields_canonical_sock_and_responds():
    import os
    import socket
    import struct
    import json
    with m2_proxy_ctx.mock_proxy_ctx() as sock:
        assert os.path.isabs(sock) and os.path.realpath(sock) == sock
        assert sock.startswith("/private/tmp/")
        # llm_call 1회 → mock 응답
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(5)
        c.connect(sock)
        req = json.dumps({"v": 1, "kind": "llm_call",
                          "template_id": m2_proxy_ctx.MOCK_TEMPLATE_ID,
                          "slots": {}, "task_id": "t"}).encode("utf-8")
        c.sendall(struct.pack(">I", len(req)) + req)
        hdr = c.recv(4)
        (n,) = struct.unpack(">I", hdr)
        body = json.loads(c.recv(n).decode("utf-8"))
        c.close()
        assert body["ok"] is True and "MOCK_PROVIDER_RESPONSE" in body["content"]
    # 퇴장 후 sock 정리됨
    assert not os.path.exists(sock)


def test_mock_proxy_ctx_rejects_raw_prompt():
    import socket
    import struct
    import json
    with m2_proxy_ctx.mock_proxy_ctx() as sock:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(5)
        c.connect(sock)
        # raw prompt 금지 계약 검증
        req = json.dumps({"v": 1, "kind": "llm_call", "prompt": "exfil me",
                          "template_id": m2_proxy_ctx.MOCK_TEMPLATE_ID}).encode("utf-8")
        c.sendall(struct.pack(">I", len(req)) + req)
        hdr = c.recv(4)
        (n,) = struct.unpack(">I", hdr)
        body = json.loads(c.recv(n).decode("utf-8"))
        c.close()
        assert body["ok"] is False and "raw prompt" in (body["error"] or "")


# ---------------------------------------------------------------------------
# S5 — _real_spawn_capture e2e (실 sandbox-exec)
# ---------------------------------------------------------------------------
def test_r4_e2e_real_sandbox_done_with_mock_codex(kanban_home, tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    deliverable = "proposal/MERGE_PROPOSAL.json"
    with kb.connect() as conn:
        tid = _make_impl_task(conn, ws, deliverable)
        kb.record_plan_submission(conn, tid, WRITE_PLAN)
        task = kb.get_task(conn, tid)
        res = m2_supervisor.supervise(
            conn, task, str(ws),
            declared_writes=["source_staging/gen.py"],
            deliverable=deliverable,
            # spawn_capture_fn 기본 = _real_spawn_capture(실 sandbox-exec)
            proxy_ctx_fn=m2_proxy_ctx.mock_proxy_ctx,
            codex_review_fn=lambda staged: {"verdict": "PASS", "high": 0},
            timeout=60,
        )
        assert res.get("error") is None, res
        assert res["completed"] is True, res
        # 워커가 샌드박스 안에서 MW_0를 실제로 썼는지(benign delta)
        leaf = ws / "source_staging" / "gen.py"
        assert leaf.exists() and "MOCK_DELTA_VERSION" in leaf.read_text(encoding="utf-8")
        # reap 계약 마커 경유(F2) + capture_dir 검증 통과(F3)
        assert kb.get_task(conn, tid).status == "done"
        assert (ws / deliverable).exists()  # 부모가 제안서 조립


def test_extract_codex_verdict_accepts_json_fence():
    out = m2_supervisor._extract_codex_verdict(
        'review complete\n```json\n{"verdict":"PASS","high":0,"note":"ok"}\n```'
    )
    assert out == {"verdict": "PASS", "high": 0, "note": "ok"}


def test_real_codex_review_env_gated_and_parses_fake_cli(tmp_path, monkeypatch):
    staged = tmp_path / "gen.py"
    staged.write_text("print('synthetic')\n", encoding="utf-8")
    assert m2_supervisor._real_codex_review([str(staged)])["verdict"] == "BLOCKED"

    monkeypatch.setenv("HERMES_M2_CODEX_REVIEW", "1")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text('{"verdict":"PASS","high":0,"note":"synthetic safe"}', encoding="utf-8")
        return Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    out = m2_supervisor._real_codex_review([str(staged)])
    assert out == {"verdict": "PASS", "high": 0, "note": "synthetic safe"}


def test_r4_e2e_real_codex_blocks_at_gate(kanban_home, tmp_path):
    # R4 기본 경로: env 미설정이면 _real_codex_review=BLOCKED → codex_review_passed 게이트 차단(done 불가).
    ws = tmp_path / "ws"; ws.mkdir()
    deliverable = "proposal/MERGE_PROPOSAL.json"
    with kb.connect() as conn:
        tid = _make_impl_task(conn, ws, deliverable)
        kb.record_plan_submission(conn, tid, WRITE_PLAN)
        task = kb.get_task(conn, tid)
        res = m2_supervisor.supervise(
            conn, task, str(ws),
            declared_writes=["source_staging/gen.py"],
            deliverable=deliverable,
            proxy_ctx_fn=m2_proxy_ctx.mock_proxy_ctx,
            # codex_review_fn 기본 = _real_codex_review(BLOCKED, R5 전까지)
            timeout=60,
        )
        # 워커 spawn·capture·phase2는 성공했으나 codex 게이트에서 차단(R4 의도)
        assert res["completed"] is False
        assert res.get("error") == "verification_failed"
        assert any(f["type"] == "codex_review_passed" and not f["ok"]
                   for f in res.get("findings", []))
        leaf = ws / "source_staging" / "gen.py"
        assert leaf.exists()  # 워커는 실제로 샌드박스에서 돌았다
        assert kb.get_task(conn, tid).status != "done"
