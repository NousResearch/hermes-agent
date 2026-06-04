"""Tests for the ② role gate (hermes_cli.role_gate).

두 층 검증:
  1. check_plan() — read-only 계획 텍스트 결정적 grep (kill switch·범위·의존성).
  2. build_readonly_gate_recipe() — 라이브 _parse_gate_recipe / _evaluate_gate_recipe
     엔진이 실제로 받아들이고, 산출물 노트/child 카드 조건을 결정적으로 차단·통과.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import role_gate


# ---------------------------------------------------------------------------
# 1) check_plan — 사전(pre-execution) 결정적 게이트
# ---------------------------------------------------------------------------

COMPLIANT_PLAN = """\
조사 계획 (news-curator)
- 의존성: 입력 키워드 'AI 에이전트', 기간 최근 7일. 산출물 경로 workspace/note.md
- 중단 조건: 출처 확보 실패 시 중단한다. 쓰기 필요 판단 시 STOP 후 보고.
- 단계:
  1. [read] 웹 검색으로 후보 기사 수집
  2. [read] 상위 기사 본문 열람·요약
  3. [write:조사노트] workspace/note.md 에 요약·출처 정리
"""


def test_check_plan_compliant_passes():
    res = role_gate.check_plan(COMPLIANT_PLAN)
    assert res["passed"] is True, res["findings"]
    assert {f["type"] for f in res["findings"]} == {"kill_switch", "scope_readonly", "dependencies"}
    assert all(f["ok"] for f in res["findings"])


def test_check_plan_missing_kill_switch_fails():
    plan = COMPLIANT_PLAN.replace("- 중단 조건: 출처 확보 실패 시 중단한다. 쓰기 필요 판단 시 STOP 후 보고.\n", "")
    res = role_gate.check_plan(plan)
    assert res["passed"] is False
    ks = next(f for f in res["findings"] if f["type"] == "kill_switch")
    assert ks["ok"] is False and ks["reason"]


def test_check_plan_illegal_write_target_fails():
    # read-only 역할이 코드 파일에 쓰는 단계를 계획 → scope 위반
    plan = COMPLIANT_PLAN + "  4. [write:hermes_cli/config.py] 설정 수정\n"
    res = role_gate.check_plan(plan)
    assert res["passed"] is False
    scope = next(f for f in res["findings"] if f["type"] == "scope_readonly")
    assert scope["ok"] is False
    assert "config.py" in scope["reason"]


def test_check_plan_no_scope_tags_fails():
    plan = "의존성: 입력 키워드. 중단 조건: 실패 시 중단한다.\n자유 서술 단계만 있고 태그 없음"
    res = role_gate.check_plan(plan)
    assert res["passed"] is False
    scope = next(f for f in res["findings"] if f["type"] == "scope_readonly")
    assert scope["ok"] is False


def test_check_plan_missing_deps_fails():
    plan = "중단 조건: 실패 시 중단한다.\n1. [read] 검색\n2. [write:조사노트] 정리"
    res = role_gate.check_plan(plan)
    assert res["passed"] is False
    deps = next(f for f in res["findings"] if f["type"] == "dependencies")
    assert deps["ok"] is False


def test_check_plan_unsupported_policy():
    res = role_gate.check_plan(COMPLIANT_PLAN, policy="read-write")
    assert res["passed"] is False
    assert res["findings"][0]["type"] == "policy"


def test_build_recipe_requires_path():
    with pytest.raises(ValueError):
        role_gate.build_readonly_gate_recipe("")


# ---------------------------------------------------------------------------
# 2) build_readonly_gate_recipe — 라이브 엔진 통합
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_recipe_passes_live_parser():
    recipe = role_gate.build_readonly_gate_recipe("note.md")
    import json
    parsed, errors = kb._parse_gate_recipe(json.dumps(recipe))
    assert errors == [], errors
    assert parsed is not None
    assert {c["type"] for c in parsed["checks"]} == {"plan_gate", "artifact_exists", "no_child_cards"}


def _make_news_task(conn, ws):
    recipe = role_gate.build_readonly_gate_recipe("note.md")
    return kb.create_task(
        conn, title="news job", assignee="news-curator",
        workspace_path=str(ws), gate_recipe=recipe,
    )


def test_live_gate_blocks_when_note_absent(kanban_home, tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with kb.connect() as conn:
        tid = _make_news_task(conn, ws)
        kb.record_plan_submission(conn, tid, COMPLIANT_PLAN)  # ② 통과시켜도
        # ① 산출물 노트가 없으니 완료 게이트가 막아야 한다.
        with pytest.raises(kb.VerificationFailedError):
            kb.complete_task(conn, tid, result="done")


def test_live_gate_passes_with_plan_and_note(kanban_home, tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "note.md").write_text("# 조사노트\n요약\n", encoding="utf-8")
    with kb.connect() as conn:
        tid = _make_news_task(conn, ws)
        res = kb.record_plan_submission(conn, tid, COMPLIANT_PLAN)
        assert res["passed"] is True
        # ②계획 통과 + ①노트 존재 → 이중게이트 통과.
        assert kb.complete_task(conn, tid, result="done") is True


def test_plan_gate_blocks_when_no_plan_submitted(kanban_home, tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "note.md").write_text("# 조사노트\n", encoding="utf-8")  # ① 충족
    with kb.connect() as conn:
        tid = _make_news_task(conn, ws)
        # 계획 미제출 → ②가 막아야 한다(노트가 있어도).
        with pytest.raises(kb.VerificationFailedError) as ei:
            kb.complete_task(conn, tid, result="done")
        types = {f["type"] for f in ei.value.findings} if hasattr(ei.value, "findings") else set()
        # plan_gate 실패가 포함돼야 한다.
        assert "plan_gate" in str(ei.value) or "plan_gate" in types


def test_plan_gate_blocks_when_plan_noncompliant(kanban_home, tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "note.md").write_text("# 조사노트\n", encoding="utf-8")
    bad_plan = COMPLIANT_PLAN + "  9. [write:hermes_cli/x.py] 코드 수정\n"  # 범위 위반
    with kb.connect() as conn:
        tid = _make_news_task(conn, ws)
        res = kb.record_plan_submission(conn, tid, bad_plan)
        assert res["passed"] is False  # 즉시 피드백도 거절
        # 완료 시 재실행에서도 막혀야 한다(무신뢰).
        with pytest.raises(kb.VerificationFailedError):
            kb.complete_task(conn, tid, result="done")


def _gate_types(recipe_json):
    import json
    return {c["type"] for c in json.loads(recipe_json)["checks"]}


def test_gate_recipe_for_assignee_known_and_unknown():
    r = role_gate.gate_recipe_for_assignee("news-curator")
    assert r is not None
    assert {c["type"] for c in r["checks"]} == {"plan_gate", "artifact_exists", "no_child_cards"}
    assert role_gate.gate_recipe_for_assignee("default") is None
    assert role_gate.gate_recipe_for_assignee(None) is None
    assert role_gate.gate_recipe_for_assignee("") is None


def test_create_task_auto_attaches_for_role(kanban_home):
    with kb.connect() as conn:
        # 역할 카드: gate_recipe 명시 안 해도 자동 부착.
        tid = kb.create_task(conn, title="research", assignee="news-curator")
        t = kb.get_task(conn, tid)
        assert t.gate_recipe is not None
        assert _gate_types(t.gate_recipe) == {"plan_gate", "artifact_exists", "no_child_cards"}
        # default 카드: 게이트 없음(기존 동작 보존).
        tid2 = kb.create_task(conn, title="other", assignee="default")
        assert kb.get_task(conn, tid2).gate_recipe is None


def test_create_task_explicit_recipe_respected(kanban_home):
    with kb.connect() as conn:
        explicit = {"checks": [{"type": "no_child_cards"}]}
        tid = kb.create_task(conn, title="r", assignee="news-curator", gate_recipe=explicit)
        # 호출자가 명시하면 자동부착이 덮어쓰지 않는다.
        assert _gate_types(kb.get_task(conn, tid).gate_recipe) == {"no_child_cards"}


def test_decompose_auto_attaches_for_role_child(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="research request", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, root, root_assignee="default",
            children=[
                {"title": "조사: AI 동향", "assignee": "news-curator", "parents": []},
                {"title": "정리", "assignee": "default", "parents": [0]},
            ],
        )
        assert child_ids and len(child_ids) == 2
        nc = kb.get_task(conn, child_ids[0])
        assert nc.assignee == "news-curator"
        assert nc.gate_recipe is not None
        assert _gate_types(nc.gate_recipe) == {"plan_gate", "artifact_exists", "no_child_cards"}
        # default child 는 게이트 없음.
        assert kb.get_task(conn, child_ids[1]).gate_recipe is None


def test_specify_triage_auto_attaches_for_role(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="raw research req", triage=True)
        ok = kb.specify_triage_task(conn, tid, assignee="news-curator", author="orchestrator")
        assert ok
        t = kb.get_task(conn, tid)
        assert t.assignee == "news-curator"
        assert t.gate_recipe is not None
        assert _gate_types(t.gate_recipe) == {"plan_gate", "artifact_exists", "no_child_cards"}


def test_specify_triage_no_recipe_for_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)
        kb.specify_triage_task(conn, tid, assignee="default", author="orchestrator")
        assert kb.get_task(conn, tid).gate_recipe is None


def test_plan_gate_reruns_latest_plan_at_completion(kanban_home, tmp_path):
    """제출 시 통과해도, 이후 나쁜 계획으로 바꿔치기하면 완료 시 재실행이 잡는다."""
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "note.md").write_text("# 조사노트\n", encoding="utf-8")
    bad_plan = "자유서술. 태그도 의존성도 중단조건도 없음."
    with kb.connect() as conn:
        tid = _make_news_task(conn, ws)
        kb.record_plan_submission(conn, tid, COMPLIANT_PLAN)   # 먼저 통과
        kb.record_plan_submission(conn, tid, bad_plan)          # 나중에 나쁜 계획
        # 최신 계획(나쁜 것)으로 재실행 → 차단.
        with pytest.raises(kb.VerificationFailedError):
            kb.complete_task(conn, tid, result="done")
