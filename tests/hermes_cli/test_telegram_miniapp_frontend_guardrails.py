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


EXPECTED_ALLOWED_API_PATHS = {
    "/api/auth/telegram",
    "/api/logout",
    "/api/me",
    "/api/status",
    "/api/capabilities",
    "/api/approvals",
    "/api/sessions",
    "/api/logs",
}


def read_frontend(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def all_frontend_sources() -> dict[Path, str]:
    paths = sorted(FRONTEND_SRC.rglob("*.ts")) + sorted(FRONTEND_SRC.rglob("*.tsx"))
    assert APP_TSX in paths and API_TS in paths and MOCK_DATA_TS in paths
    return {path: read_frontend(path) for path in paths}


def all_shipped_frontend_assets() -> dict[Path, str]:
    # Everything Vite ships to the browser: TS/TSX sources plus CSS and the
    # HTML entry point — an endpoint string in any of them is a violation.
    app_dir = FRONTEND_SRC.parent
    paths = (
        sorted(FRONTEND_SRC.rglob("*.ts"))
        + sorted(FRONTEND_SRC.rglob("*.tsx"))
        + sorted(FRONTEND_SRC.rglob("*.css"))
        + [app_dir / "index.html"]
    )
    return {path: read_frontend(path) for path in paths if path.exists()}


def test_frontend_has_no_action_endpoint_strings_or_fetch_helpers():
    assets = all_shipped_frontend_assets()
    combined = "\n".join(assets.values())

    for path, text in assets.items():
        for endpoint in FORBIDDEN_FRONTEND_ENDPOINTS:
            assert endpoint not in text, f"{path.name} references forbidden endpoint {endpoint}"
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


def test_api_client_enforces_runtime_path_allowlist():
    api_text = read_frontend(API_TS)

    match = re.search(r"const ALLOWED_API_PATHS = new Set\(\[([\s\S]*?)\]\)", api_text)
    assert match, "api.ts must declare a runtime allowlist of API paths"
    declared = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert declared == EXPECTED_ALLOWED_API_PATHS

    request_fn = re.search(r"async function requestJson[\s\S]*?\n\}", api_text)
    assert request_fn, "requestJson must exist as the single fetch entry point"
    body = request_fn.group(0)
    guard = re.search(
        r"if\s*\(\s*!ALLOWED_API_PATHS\.has\(path\)\s*\)\s*\{\s*\n\s*throw new Error",
        body,
    )
    assert guard, (
        "requestJson must contain the rejecting guard branch: "
        "if (!ALLOWED_API_PATHS.has(path)) { throw new Error(...) }"
    )
    assert guard.start() < body.index("fetch("), (
        "requestJson must reject non-allowlisted paths before any fetch"
    )

    assert api_text.count("fetch(") == 1, "all API calls must go through requestJson"


# Word-boundary match catches aliased/member-access variants too:
# fetch (, globalThis.fetch(, navigator['sendBeacon'](, new window.WebSocket(.
NETWORK_SINK_RE = re.compile(r"\b(fetch|XMLHttpRequest|sendBeacon|WebSocket|EventSource)\b")


def test_all_network_sinks_are_confined_to_api_client():
    for path, text in all_frontend_sources().items():
        if path == API_TS:
            continue
        match = NETWORK_SINK_RE.search(text)
        assert match is None, (
            f"{path.name} references network sink '{match.group(0) if match else ''}'; "
            "all network calls go through requestJson in api.ts, which enforces the path allowlist"
        )


def test_api_client_contains_single_guarded_fetch_sink():
    api_text = read_frontend(API_TS)

    # Word-level count (not just call syntax) also catches aliasing such as
    # `const rawFetch = fetch` or `window["fetch"]` inside api.ts itself.
    assert len(re.findall(r"\bfetch\b", api_text)) == 1, (
        "api.ts must reference fetch exactly once — the guarded call inside requestJson"
    )
    assert len(re.findall(r"\bfetch\s*\(", api_text)) == 1
    for sink in ("XMLHttpRequest", "sendBeacon", "WebSocket", "EventSource"):
        assert not re.search(rf"\b{sink}\b", api_text), (
            f"api.ts must not use {sink}; requestJson's guarded fetch is the only sink"
        )


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
