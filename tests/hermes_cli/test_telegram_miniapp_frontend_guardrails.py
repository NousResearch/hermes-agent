"""Static frontend guardrails for Telegram Mini App action-surface safety."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "apps" / "telegram-miniapp" / "src"
APP_TSX = FRONTEND_SRC / "App.tsx"
API_TS = FRONTEND_SRC / "api.ts"
MOCK_DATA_TS = FRONTEND_SRC / "mockData.ts"
APP_MODEL_TS = FRONTEND_SRC / "appModel.ts"
APPROVALS_SECTION_TSX = FRONTEND_SRC / "components" / "ApprovalsSection.tsx"
COMMAND_PALETTE_TSX = FRONTEND_SRC / "components" / "CommandPalette.tsx"
SNAPSHOTS_HOOK_TS = FRONTEND_SRC / "useMiniAppSnapshots.ts"

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


def test_frontend_has_no_forbidden_action_endpoint_strings():
    assets = all_shipped_frontend_assets()
    combined = "\n".join(assets.values())

    for path, text in assets.items():
        for endpoint in FORBIDDEN_FRONTEND_ENDPOINTS:
            assert endpoint not in text, f"{path.name} references forbidden endpoint {endpoint}"
    # The Phase 1 gate exposes exactly one mutating path: /api/approvals/{id}/decision.
    # Direct approve/reject sub-paths remain forbidden (decisions go through the
    # single decision endpoint), and restart/execute/etc are covered above.
    assert not re.search(r"/api/approvals/[^\n\"'`]+/(approve|reject)\b", combined)

    api_text = read_frontend(API_TS)
    # No approve/reject/restart/... helper — the only mutating helper is
    # postApprovalDecision, which does not contain any forbidden token.
    forbidden_helper_names = [
        "restart",
        "execute",
        "process",
        "modelSwitch",
    ]
    for name in forbidden_helper_names:
        assert not re.search(rf"export\s+async\s+function\s+\w*{name}\w*", api_text, re.IGNORECASE)


def test_single_mutating_helper_is_decision_with_proof_header():
    api_text = read_frontend(API_TS)
    # Exactly one POST-to-decision helper, and it attaches the Telegram proof
    # header the backend requires (the session cookie alone is refused).
    assert "export async function postApprovalDecision" in api_text
    match = re.search(r"export async function postApprovalDecision[\s\S]*?\n\}", api_text)
    assert match, "postApprovalDecision must exist"
    body = match.group(0)
    assert "/api/approvals/" in body and "/decision" in body
    assert '"x-telegram-init-data"' in body, "decision must carry the fresh Telegram proof header"
    # No other exported helper posts to a mutating path.
    posts = re.findall(r'requestJson<[^>]*>\(([^,]+),\s*\{\s*[\s\S]*?method:\s*"POST"', api_text)
    mutating = [p for p in posts if "/decision" in p or "/auth/telegram" in p or "/logout" in p]
    assert len(mutating) == len(posts), f"unexpected mutating POST helper: {posts}"


def test_api_client_enforces_runtime_path_allowlist():
    api_text = read_frontend(API_TS)

    match = re.search(r"const ALLOWED_API_PATHS = new Set\(\[([\s\S]*?)\]\)", api_text)
    assert match, "api.ts must declare a runtime allowlist of API paths"
    declared = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert declared == EXPECTED_ALLOWED_API_PATHS

    # The one dynamic mutating path is allowed only via a strict anchored
    # pattern — the exact decision endpoint and nothing else.
    decision_pat = re.search(r"const DECISION_PATH = /(.+?)/;", api_text)
    assert decision_pat, "api.ts must declare a strict DECISION_PATH pattern"
    assert decision_pat.group(1).startswith("^") and decision_pat.group(1).endswith("decision$"), (
        "DECISION_PATH must be anchored to exactly /api/approvals/<id>/decision"
    )
    assert "isAllowedApiPath" in api_text
    guard_fn = re.search(r"function isAllowedApiPath[\s\S]*?\n\}", api_text)
    assert guard_fn, "isAllowedApiPath must exist"
    assert "ALLOWED_API_PATHS.has(path)" in guard_fn.group(0)
    assert "DECISION_PATH.test(path)" in guard_fn.group(0)

    request_fn = re.search(r"async function requestJson[\s\S]*?\n\}", api_text)
    assert request_fn, "requestJson must exist as the single fetch entry point"
    body = request_fn.group(0)
    guard = re.search(
        r"if\s*\(\s*!isAllowedApiPath\(path\)\s*\)\s*\{\s*\n\s*throw new Error",
        body,
    )
    assert guard, (
        "requestJson must contain the rejecting guard branch: "
        "if (!isAllowedApiPath(path)) { throw new Error(...) }"
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


def test_decision_buttons_are_gated_and_go_through_confirm():
    approvals_text = read_frontend(APPROVALS_SECTION_TSX)

    # Decisions are actionable only when the backend capability is live, the
    # user is the authenticated owner, and the approval advertises the decision.
    assert "actionsEnabled && isOwner && Boolean(onDecision)" in approvals_text
    # Buttons are disabled unless canDecide (and while a decision is pending),
    # AND unless the approval advertises that specific decision.
    assert 'disabled={!canDecide || pending || !allowed.includes("approve_once")}' in approvals_text
    assert 'disabled={!canDecide || pending || !allowed.includes("reject_once")}' in approvals_text
    # canDecide requires the server-authenticated owner flag, not just initData.
    assert "actionsEnabled && isOwner && Boolean(onDecision)" in approvals_text
    # The confirm freezes the exact target approval (no redirect on refresh).
    assert "confirm.approval" in approvals_text and "runDecision(confirm.approval" in approvals_text
    # Tapping a decision opens the two-step confirm sheet with a FROZEN target
    # (approval + decision); it does NOT fetch or call the handler directly.
    assert 'setConfirm({ approval: selected, decision: "approve_once" })' in approvals_text
    assert 'setConfirm({ approval: selected, decision: "reject_once" })' in approvals_text
    # The network call only happens from the confirm step, via the injected
    # onDecision prop (which routes through api.ts) — never a direct fetch here.
    assert "fetch" not in approvals_text
    assert "ConfirmSheet" in approvals_text and "ШАГ 2" in approvals_text
    # The raw command is never displayed; the sheet states so explicitly.
    assert "Сырая команда не раскрывается" in approvals_text


def test_action_gate_fails_closed_on_any_degraded_action_read():
    hook_text = read_frontend(SNAPSHOTS_HOOK_TS)

    # The capability that opens the action gate may be stored ONLY when every
    # action-critical read of the poll is fresh and successful. A tripwire that
    # breaks if someone weakens the invariant back to capabilities-only.
    assert "const actionReadsHealthy =" in hook_text
    for cond in (
        'status.status === "fulfilled"',
        'capabilities.status === "fulfilled"',
        'approvals.status === "fulfilled"',
    ):
        assert cond in hook_text
    assert "setServerCapabilities(actionReadsHealthy ? capabilities.value.items : [])" in hook_text
    # The catch path also fails closed.
    assert hook_text.count("setServerCapabilities([])") >= 1
    # The stale approvals version is cleared on any degraded/errored poll, and a
    # fresh version is only written back when the whole poll is healthy.
    assert 'if (!actionReadsHealthy) setApprovalsVersion("")' in hook_text
    assert "if (actionReadsHealthy) setApprovalsVersion(approvals.value.snapshot_version" in hook_text
    # The effective gate also requires the live status flag, so a status that
    # reads safe/blocked can never sit over enabled action buttons.
    assert "snapshot?.miniapp.actions_enabled === true" in hook_text


def test_command_palette_route_map_is_navigation_only():
    palette_text = read_frontend(COMMAND_PALETTE_TSX)
    match = re.search(r"const routeMap:.*?= \{([\s\S]*?)\};", palette_text)
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
    app_text = read_frontend(APP_MODEL_TS)
    match = re.search(r"const STORAGE_KEYS = \{([\s\S]*?)\} as const;", app_text)
    assert match, "STORAGE_KEYS must stay centralized and reviewable"
    storage_block = match.group(1)
    keys = set(re.findall(r"(\w+):\s*\"([^\"]+)\"", storage_block))

    assert keys == {
        ("activeTab", "hermes-miniapp:active-tab"),
        ("selectedApprovalId", "hermes-miniapp:selected-approval-id"),
        ("theme", "hermes-miniapp:theme"),
    }
    for forbidden in ("action", "decision", "payload", "initData", "token", "command", "approvalPayload"):
        assert forbidden.lower() not in storage_block.lower()


def test_approvals_badge_is_derived_live_not_hardcoded():
    # Truthfulness tripwire: the nav badge must never carry a static count.
    mock_text = read_frontend(MOCK_DATA_TS)
    nav_block = re.search(r"export const navItems[\s\S]*?\];", mock_text)
    assert nav_block, "navItems must be reviewable"
    assert "badge:" not in nav_block.group(0), "navItems must NOT hardcode a badge; derive it live"

    app_text = read_frontend(APP_TSX)
    # The approvals badge is computed from the real queue length AND suppressed
    # in live mode unless the approvals endpoint is fresh (no stale count).
    assert "item.key === \"approvals\" && approvalCount > 0 && approvalsTrustworthy" in app_text
    assert 'endpointHealth.approvals.state === "ok"' in app_text


def test_mock_quick_actions_remain_read_only_navigation():
    mock_text = read_frontend(MOCK_DATA_TS)
    ids = set(re.findall(r'id: "([^"]+)"', mock_text[mock_text.index("export const quickActions"):mock_text.index("export const recentLogs")]))
    risks = set(re.findall(r'risk: "([^"]+)"', mock_text[mock_text.index("export const quickActions"):mock_text.index("export const recentLogs")]))

    assert ids == {"status", "sessions", "approvals", "logs"}
    assert risks == {"read_only"}
    for forbidden in ("restart", "approve", "reject", "execute", "command", "process", "config"):
        assert forbidden not in ids
