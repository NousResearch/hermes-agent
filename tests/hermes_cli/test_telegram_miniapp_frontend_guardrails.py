"""Static frontend guardrails for Telegram Mini App action-surface safety."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "apps" / "telegram-miniapp" / "src"
APP_TSX = FRONTEND_SRC / "App.tsx"
API_TS = FRONTEND_SRC / "api.ts"
MOCK_DATA_TS = FRONTEND_SRC / "mockData.ts"

FORBIDDEN_FRONTEND_ENDPOINTS = [
    "/api/actions",
    "/api/restart",
    "/api/execute",
    "/api/tool",
    "/api/command",
    "/api/process",
    "/api/config",
    "/api/model/switch",
]


def read_frontend(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_frontend_has_no_action_endpoint_strings_or_fetch_helpers():
    combined = "\n".join(read_frontend(path) for path in (APP_TSX, API_TS, MOCK_DATA_TS))

    for endpoint in FORBIDDEN_FRONTEND_ENDPOINTS:
        assert endpoint not in combined
    assert not re.search(r"/api/approvals/[^\n\"'`]+/(approve|reject|decision)", combined)

    api_text = read_frontend(API_TS)
    forbidden_helper_names = [
        "approve",
        "reject",
        "restart",
        "execute",
        "command",
        "process",
        "modelSwitch",
        "config",
    ]
    for name in forbidden_helper_names:
        assert not re.search(rf"export\s+async\s+function\s+\w*{name}\w*", api_text, re.IGNORECASE)


def test_decision_strip_buttons_are_disabled_and_handler_free():
    app_text = read_frontend(APP_TSX)
    match = re.search(r'<div className="decision-strip"[\s\S]*?</div>', app_text)
    assert match, "decision-strip block must exist as a visible locked affordance"
    block = match.group(0)

    assert "disabled>Одобрить позже" in block
    assert "disabled>Отклонить позже" in block
    assert "onClick" not in block
    assert "fetch" not in block


def test_command_palette_route_map_is_navigation_only():
    app_text = read_frontend(APP_TSX)
    match = re.search(r"const routeMap:.*?= \{([\s\S]*?)\};", app_text)
    assert match, "Command palette routeMap should be explicit and reviewable"
    route_map = match.group(1)

    assert set(re.findall(r"(\w+):\s*\"(\w+)\"", route_map)) == {
        ("status", "status"),
        ("sessions", "sessions"),
        ("approvals", "approvals"),
        ("logs", "logs"),
    }
    for forbidden in ("restart", "approve", "reject", "actions", "execute", "command", "process", "config"):
        assert forbidden not in route_map.lower()


def test_local_storage_keys_remain_harmless_ui_state_only():
    app_text = read_frontend(APP_TSX)
    match = re.search(r"const STORAGE_KEYS = \{([\s\S]*?)\} as const;", app_text)
    assert match, "STORAGE_KEYS must stay centralized and reviewable"
    storage_block = match.group(1)
    keys = set(re.findall(r"(\w+):\s*\"([^\"]+)\"", storage_block))

    assert keys == {
        ("activeTab", "hermes-miniapp:active-tab"),
        ("selectedApprovalId", "hermes-miniapp:selected-approval-id"),
    }
    for forbidden in ("action", "decision", "payload", "initData", "token", "command", "approvalPayload"):
        assert forbidden.lower() not in storage_block.lower()


def test_mock_quick_actions_remain_read_only_navigation():
    mock_text = read_frontend(MOCK_DATA_TS)
    ids = set(re.findall(r'id: "([^"]+)"', mock_text[mock_text.index("export const quickActions"):mock_text.index("export const recentLogs")]))
    risks = set(re.findall(r'risk: "([^"]+)"', mock_text[mock_text.index("export const quickActions"):mock_text.index("export const recentLogs")]))

    assert ids == {"status", "sessions", "approvals", "logs"}
    assert risks == {"read_only"}
    for forbidden in ("restart", "approve", "reject", "execute", "command", "process", "config"):
        assert forbidden not in ids
