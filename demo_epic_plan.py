#!/usr/bin/env python3
import json
import logging
import sys

# 터미널 로깅 셋팅
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from tools.epic_manager_tool import epic_delegate

print("== 🚀 Epic Manager Kanban Board Demo ==")

goal_text = "간단한 터미널용 To-Do CLI 앱 (파이썬, SQLite 연동, 추가/삭제 기능)"
print(f"Goal: {goal_text}\n")

class DummyAgent:
    name = "DummyDemoAgent"
    model = "accounts/fireworks/routers/kimi-k2p5-turbo"
    base_url = None
    _active_children = []
    platform = "linux"
    providers_allowed = []
    providers_ignored = []
    providers_order = []
    provider_sort = "performance"
dummy = DummyAgent()

# plan_only 모드로 실행
result_json = epic_delegate(goal_text, mode="plan_only", parent_agent=dummy)

print("\n== 🟢 Result Output from epic_delegate ==")
try:
    data = json.loads(result_json)
    if data["success"]:
        # message 항목에 칸반보드가 들어있음
        print(data["message"])
    else:
        print(f"Error: {data.get('error')}")
except Exception as e:
    print(f"Failed to parse result: {e}\nRaw={result_json}")
