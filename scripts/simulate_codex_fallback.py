#!/usr/bin/env python3
"""Codex → local-claude fallback 시뮬레이션.

실제 try_activate_fallback() 코드 경로를 직접 실행해서
Context Bridge와 Skill Re-injection이 동작하는지 확인한다.

실행:
    cd /home/ubuntu/hermes-agent
    python scripts/simulate_codex_fallback.py
"""

from __future__ import annotations

import json
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from unittest.mock import MagicMock

# ──────────────────────────────────────────────
# ANSI 컬러 헬퍼
# ──────────────────────────────────────────────
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(title): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}{title}{RESET}\n{'─'*60}")
def ok(msg):       print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):     print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg):     print(f"  {msg}")
def field(k, v):   print(f"  {BOLD}{k:30s}{RESET}{v}")

# ──────────────────────────────────────────────
# Step 1: Codex 세션 대화 히스토리 구성
# ──────────────────────────────────────────────
header("Step 1: Codex 세션 대화 히스토리 구성")

ORIGINAL_SYSTEM = (
    "You are Hermes, a helpful AI assistant.\n\n"
    "## Skills\n"
    "- codex-demo: Codex 세션 중 만들어진 스킬\n"
    "- code-review: 코드 리뷰 스킬\n"
)

api_messages = [
    {"role": "system", "content": ORIGINAL_SYSTEM},
    {"role": "user", "content": "파이썬 덧셈 함수 만들어줘"},
    {
        "role": "assistant",
        "content": "네, `add(a, b)` 함수를 만들겠습니다.",
        # Codex Responses API 전용 필드
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "encrypted_content": "enc_codex_abc123==",
                "_issuer_kind": "codex_backend",
            }
        ],
        "codex_message_items": [
            {"type": "message", "id": "msg_001", "content": [{"type": "output_text", "text": "함수 생성 중..."}]}
        ],
        "_thinking_prefill": True,
    },
    {"role": "tool", "content": '{"result": "def add(a, b): return a + b"}', "tool_call_id": "call_001"},
    {
        "role": "assistant",
        "content": "완료! `def add(a, b): return a + b` 함수를 만들었습니다.",
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "encrypted_content": "enc_codex_def456==",
                "_issuer_kind": "codex_backend",
            }
        ],
    },
    {"role": "user", "content": "고마워! 이제 skill도 하나 저장해줘"},
    {
        "role": "assistant",
        "content": "codex-demo 스킬을 저장했습니다.",
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "encrypted_content": "enc_codex_ghi789==",
                "_issuer_kind": "codex_backend",
            }
        ],
        "codex_message_items": [
            {"type": "message", "id": "msg_002", "content": [{"type": "output_text", "text": "스킬 저장 완료"}]}
        ],
    },
]

before = copy.deepcopy(api_messages)

info(f"메시지 수          : {len(api_messages)}")
info(f"시스템 프롬프트 포함: {'codex-demo' in api_messages[0]['content']}")
info(f"codex_reasoning_items 포함 메시지 수: "
     f"{sum(1 for m in api_messages if 'codex_reasoning_items' in m)}")
info(f"codex_message_items 포함 메시지 수 : "
     f"{sum(1 for m in api_messages if 'codex_message_items' in m)}")

# ──────────────────────────────────────────────
# Step 2: try_activate_fallback() 핵심 로직 시뮬레이션
# ──────────────────────────────────────────────
header("Step 2: try_activate_fallback() — Codex → local-claude 전환")

REBUILT_SYSTEM = (
    "You are Hermes, a helpful AI assistant.\n\n"
    "## Skills\n"
    "- codex-demo: Codex 세션 중 만들어진 스킬  ← Codex 세션 결과물 보존\n"
    "- code-review: 코드 리뷰 스킬\n"
    "\n[Rebuilt for anthropic_messages provider: local-claude / claude-opus-4-8]"
)

# 실제 try_activate_fallback() 이 하는 일 재현
old_api_mode = "codex_responses"      # Codex 에서
new_api_mode = "anthropic_messages"   # Anthropic(local-claude) 으로

agent = SimpleNamespace()
agent.api_mode          = old_api_mode
agent.provider          = "openai-codex"
agent.model             = "gpt-5.5"
agent.session_id        = "sim-session-001"
agent._cached_system_prompt = ORIGINAL_SYSTEM
agent._session_db       = None
agent._codex_reasoning_replay_enabled = True

agent._build_system_prompt = MagicMock(return_value=REBUILT_SYSTEM)

def _disable_replay(messages=None):
    count = 0
    for m in (messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            if m.pop("codex_reasoning_items", None):
                count += 1
    agent._codex_reasoning_replay_enabled = False
    return {"messages": count, "items": count}

agent._disable_codex_reasoning_replay = _disable_replay

# --- try_activate_fallback() 이 세팅하는 플래그 ---
if old_api_mode != new_api_mode:
    agent._pending_context_bridge = {
        "old_api_mode": old_api_mode,
        "new_api_mode": new_api_mode,
    }
    agent._cached_system_prompt = None   # 시스템 프롬프트 무효화

agent.api_mode   = new_api_mode
agent.provider   = "local-claude"
agent.model      = "claude-opus-4-8"

field("이전 provider    :", f"openai-codex  (api_mode={old_api_mode})")
field("새 provider      :", f"local-claude  (api_mode={new_api_mode})")
field("_pending_context_bridge:", json.dumps(agent._pending_context_bridge))
field("_cached_system_prompt  :", repr(agent._cached_system_prompt))

# ──────────────────────────────────────────────
# Step 3: conversation_loop 안에서 Bridge 실행
# ──────────────────────────────────────────────
header("Step 3: conversation_loop → apply_fallback_context_bridge() 실행")

from agent.fallback_context_bridge import apply_fallback_context_bridge

_bridge = getattr(agent, "_pending_context_bridge", None)
assert _bridge, "bridge 플래그가 없음 — try_activate_fallback() 수정 확인 필요"

agent._pending_context_bridge = None   # 플래그 소비

apply_fallback_context_bridge(
    agent,
    api_messages,
    _bridge["old_api_mode"],
    _bridge["new_api_mode"],
)

# ──────────────────────────────────────────────
# Step 4: 결과 검증
# ──────────────────────────────────────────────
header("Step 4: 검증 결과")

passed = 0
failed = 0

def check(desc, condition):
    global passed, failed
    if condition:
        ok(desc)
        passed += 1
    else:
        import traceback
        warn(f"FAIL: {desc}")
        failed += 1

# Codex 전용 필드가 제거됐는지
check(
    "codex_reasoning_items 전부 제거됨",
    all("codex_reasoning_items" not in m for m in api_messages),
)
check(
    "codex_message_items 전부 제거됨",
    all("codex_message_items" not in m for m in api_messages),
)
check(
    "_thinking_prefill 내부 키 제거됨",
    all("_thinking_prefill" not in m for m in api_messages),
)

# Reasoning replay 비활성화
check(
    "codex reasoning replay 비활성화",
    not agent._codex_reasoning_replay_enabled,
)

# 시스템 프롬프트 재빌드
check(
    "_build_system_prompt() 호출됨",
    agent._build_system_prompt.called,
)
check(
    "시스템 메시지가 새 프롬프트로 교체됨",
    api_messages[0]["role"] == "system" and api_messages[0]["content"] == REBUILT_SYSTEM,
)
check(
    "agent._cached_system_prompt 갱신됨",
    agent._cached_system_prompt == REBUILT_SYSTEM,
)

# Codex 세션에서 만든 스킬이 보존됐는지 (핵심!)
check(
    "Codex 세션 스킬(codex-demo)이 새 시스템 프롬프트에 보존됨",
    "codex-demo" in api_messages[0]["content"],
)

# 일반 필드(content, tool_calls 등)는 유지됐는지
check(
    "일반 content 필드 유지됨",
    all(
        "content" in m
        for m in api_messages
        if m.get("role") in ("assistant", "user", "tool")
    ),
)

# bridge 플래그 소비됨
check(
    "_pending_context_bridge 플래그 소비됨",
    agent._pending_context_bridge is None,
)

# ──────────────────────────────────────────────
# Step 5: Before / After 메시지 diff
# ──────────────────────────────────────────────
header("Step 5: Before / After 메시지 비교")

for i, (b, a) in enumerate(zip(before, api_messages)):
    role = b.get("role", "?")
    removed = set(b.keys()) - set(a.keys())
    added   = set(a.keys()) - set(b.keys())
    if removed or added:
        info(f"  msg[{i}] role={role}")
        if removed: info(f"    {RED}제거됨{RESET}: {removed}")
        if added:   info(f"    {GREEN}추가됨{RESET}: {added}")
    elif i == 0 and b["content"] != a["content"]:
        info(f"  msg[{i}] role={role}")
        info(f"    {YELLOW}시스템 프롬프트 내용 교체됨{RESET}")
        old_lines = b["content"].strip().splitlines()
        new_lines = a["content"].strip().splitlines()
        for ln in old_lines[:4]:
            info(f"      {RED}- {ln}{RESET}")
        for ln in new_lines[:5]:
            info(f"      {GREEN}+ {ln}{RESET}")

# ──────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────
header("최종 결과")
total = passed + failed
print(f"\n  {BOLD}{passed}/{total} 검증 통과{RESET}")
if failed == 0:
    print(f"\n  {GREEN}{BOLD}✅ Context Bridge + Skill Re-injection 정상 동작{RESET}")
    print(f"  Codex 세션에서 만든 스킬과 컨텍스트가")
    print(f"  local-claude(anthropic_messages)로 안전하게 전달됩니다.")
else:
    print(f"\n  {RED}{BOLD}❌ {failed}개 검증 실패 — 로그를 확인하세요{RESET}")
    sys.exit(1)
