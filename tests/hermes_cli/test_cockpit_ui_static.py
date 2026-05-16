"""Static regression checks for the production Cockpit dashboard UI (BIF-498/BIF-500/BIF-517)."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "web" / "src" / "App.tsx"
API = ROOT / "web" / "src" / "lib" / "api.ts"
COCKPIT_PAGE = ROOT / "web" / "src" / "pages" / "CockpitPage.tsx"
CHAT_PAGE = ROOT / "web" / "src" / "pages" / "ChatPage.tsx"
INDEX_HTML = ROOT / "web" / "index.html"
VITE_CONFIG = ROOT / "web" / "vite.config.ts"
MANIFEST = ROOT / "web" / "public" / "cockpit.webmanifest"
ICON_SVG = ROOT / "web" / "public" / "cockpit-icon.svg"
INDEX_CSS = ROOT / "web" / "src" / "index.css"


FORBIDDEN_PAGE_MARKERS = (
    "method: \"POST\"",
    "method: 'POST'",
    "method: `POST`",
    "method: \"PUT\"",
    "method: 'PUT'",
    "method: `PUT`",
    "method: \"PATCH\"",
    "method: 'PATCH'",
    "method: `PATCH`",
    "method: \"DELETE\"",
    "method: 'DELETE'",
    "method: `DELETE`",
    "sendMessage",
    "EventSource(",
)


def test_cockpit_page_route_and_nav_are_registered():
    app = APP.read_text(encoding="utf-8")

    assert 'const CockpitPage = lazy(() => import("@/pages/CockpitPage"));' in app
    assert '"/cockpit": () => <CockpitPage />' in app
    assert 'path: "/cockpit"' in app
    assert 'label: "Cockpit"' in app
    assert 'icon: Activity' in app


def test_builtin_dashboard_routes_are_lazy_loaded_but_persistent_chat_stays_static():
    app = APP.read_text(encoding="utf-8")

    assert "lazy," in app
    assert "Suspense," in app
    assert "function RouteLoadingFallback" in app
    assert 'const SessionsPage = lazy(() => import("@/pages/SessionsPage"));' in app
    assert 'const DocsPage = lazy(() => import("@/pages/DocsPage"));' in app
    assert 'import ChatPage from "@/pages/ChatPage";' in app
    assert 'lazy(() => import("@/pages/ChatPage"))' not in app
    assert "<ChatPage isActive={isChatRoute} />" in app
    assert "data-chat-active" in app


def test_vite_manual_chunks_split_shared_vendor_without_touching_routes():
    config = VITE_CONFIG.read_text(encoding="utf-8")

    assert "function manualChunks(id: string): string | undefined" in config
    assert "if (!id.includes(\"/node_modules/\")) return undefined;" in config
    assert "manualChunks," in config
    assert "vendor-react-runtime" in config
    assert 'packages: ["react", "react-dom", "react-router-dom", "scheduler"]' in config
    assert "vendor-noui" in config
    assert '"@nous-research/ui"' in config
    assert "vendor-icons" in config
    assert 'packages: ["lucide-react"]' in config
    assert "vendor-terminal" in config
    assert 'packages: ["@xterm"]' in config
    assert "vendor-visualization" in config
    assert '"@observablehq/plot"' in config
    assert '"@react-three/fiber"' in config
    assert '"d3-*"' in config
    assert '"three"' in config
    assert 'return "vendor-misc";' in config

    # Route/component code should remain under Vite's default route chunking.
    assert '"@/pages/CockpitPage"' not in config
    assert '"@/pages/ChatPage"' not in config


def test_cockpit_route_uses_standalone_shell_not_admin_sidebar():
    app = APP.read_text(encoding="utf-8")

    assert "normalizedPath === \"/cockpit\" || normalizedPath === \"/biff/cockpit\"" in app
    assert "<CockpitPage standalone />" in app
    assert "<PageHeaderProvider pluginTabs={pluginTabMeta}>" in app
    assert "<Suspense fallback={<CockpitRouteLoadingFallback />}>" in app
    assert "if (isCockpitRoute && !cockpitOverriddenByPlugin)" in app
    assert "data-cockpit-shell" in app


def test_cockpit_plugin_override_is_not_bypassed_by_standalone_return():
    app = APP.read_text(encoding="utf-8")

    assert 'm.tab.override === "/cockpit"' in app
    assert "const cockpitOverriddenByPlugin = useMemo(" in app
    assert "if (isCockpitRoute && pluginsLoading)" in app
    assert "<CockpitRouteLoadingFallback />" in app
    assert "if (isCockpitRoute && !cockpitOverriddenByPlugin)" in app
    assert "buildRoutes(builtinRoutes, manifests)" in app
    assert "element: <PluginPage name={om.name} />" in app

def test_cockpit_route_unlocks_document_scrolling_for_ios_pwa():
    app = APP.read_text(encoding="utf-8")

    assert "cockpitScrollable" in app
    assert "document.documentElement" in app
    assert "document.body" in app
    assert "root.style.overflow" in app
    assert "overflowY = \"auto\"" in app
    assert "maxHeight = \"none\"" in app


def test_cockpit_pwa_metadata_and_ios_tags_are_declared():
    html = INDEX_HTML.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert '<link rel="manifest" href="/cockpit.webmanifest" />' in html
    assert '<meta name="theme-color" content="#050403" />' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes" />' in html
    assert '<meta name="apple-mobile-web-app-title" content="Biff Cockpit" />' in html
    assert '<link rel="apple-touch-icon" href="/cockpit-apple-touch-icon.png" />' in html
    assert '"name": "Biff Cockpit"' in manifest
    assert '"start_url": "/cockpit"' in manifest
    assert '"display": "standalone"' in manifest
    assert '"theme_color": "#050403"' in manifest
    assert '"purpose": "any maskable"' in manifest
    assert ICON_SVG.exists()


def test_cockpit_dark_executive_visual_system_uses_semantic_tokens_not_default_palette():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    css = INDEX_CSS.read_text(encoding="utf-8")

    required_tokens = (
        "--cockpit-shell: #050403",
        "--cockpit-card: #17130f",
        "--cockpit-card-raised: #1d1813",
        "--cockpit-text: #f2ece2",
        "--cockpit-muted: #9a9187",
        "--cockpit-border: rgba(242, 236, 226, 0.12)",
        "--cockpit-active: #79d8c4",
        "--cockpit-healthy: #78c7ac",
        "--cockpit-warning: #d6a451",
        "--cockpit-risk: #d87474",
        "--cockpit-secondary: #8d86c9",
    )
    for token in required_tokens:
        assert token in css

    assert "bg-[var(--cockpit-shell)]" in page
    assert "bg-[var(--cockpit-card)]" in page
    assert "bg-[var(--cockpit-card-raised)]" in page
    assert "text-[var(--cockpit-text)]" in page
    assert "text-[var(--cockpit-muted)]" in page
    assert "border-[var(--cockpit-border)]" in page
    assert "#050812" not in page
    assert "rgba(56,189,248" not in page
    for default_marker in ("sky-", "blue-", "slate-", "emerald-", "green-", "bg-white/", "border-white/", "bg-black/"):
        assert default_marker not in page
    assert "backdrop-blur" not in page
    assert "text-[var(--cockpit-text)]0" not in page


def test_cockpit_api_client_exposes_only_read_endpoints():
    api = API.read_text(encoding="utf-8")

    assert "getCockpitCapabilities" in api
    assert "getCockpitLanes" in api
    assert "getCockpitSignals" in api
    assert "getCockpitAgentActivity" in api
    assert "getCockpitLaneMessages" in api
    assert "getCockpitN8nChecks" in api
    assert "getCockpitSelfWorkHandoff" in api
    assert 'fetchJSON<CockpitCapabilitiesResponse>("/api/cockpit/capabilities")' in api
    assert "fetchJSON<CockpitLanesResponse>(`/api/cockpit/lanes?" in api
    assert "fetchJSON<CockpitSignalsResponse>(`/api/cockpit/signals?" in api
    assert "fetchJSON<CockpitAgentActivityResponse>(`/api/cockpit/agent-activity?" in api
    assert "fetchJSON<CockpitLaneMessagesResponse>(" in api
    assert 'fetchJSON<CockpitN8nChecksResponse>("/api/cockpit/n8n-checks")' in api
    assert 'fetchJSON<CockpitSelfWorkHandoffResponse>("/api/cockpit/self-work-handoff")' in api
    assert "/api/cockpit/events" not in api, "browser SSE must not use unauthenticated EventSource"


def test_cockpit_agent_activity_panel_uses_read_only_projection_endpoint():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "type CockpitAgentActivityResponse" in page
    assert "api.getCockpitAgentActivity({ limit: LANE_LIMIT, message_limit: 8 })" in page
    assert "<AgentActivityPanel response={agentActivity}" in page
    assert "No interrupt, restart, send, deploy, workflow mutation, raw channel IDs, credentials, or full transcripts" in page
    assert "completed ${formatClockLabel(item.completed_at)}" in page
    assert "CockpitAgentActivityItem" in api
    assert 'status: "running" | "completed" | "failed" | "blocked" | "waiting" | "stale" | string' in api


def test_cockpit_api_types_expose_bounded_recent_transcript_window_metadata():
    api = API.read_text(encoding="utf-8")

    assert "CockpitTranscriptWindow" in api
    assert "transcript_window?: CockpitTranscriptWindow" in api
    assert "window?: CockpitTranscriptWindow" in api
    assert "bounded?: boolean" in api
    assert "window_limit?: number" in api
    assert "total_scope?: \"bounded_recent_window\" | string" in api
    assert "external_send_enabled?: boolean" in api
    assert "local_chat?:" in api


def test_cockpit_page_uses_local_chat_and_avoids_external_control_ui_markers():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    lowered = page.lower()

    assert "export default function CockpitPage" in page
    assert "api.getCockpitCapabilities" in page
    assert "api.getCockpitLanes" in page
    assert "api.getCockpitSignals" in page
    assert "api.getCockpitLaneMessages" in page
    assert "api.getCockpitN8nChecks" in page
    assert 'import ChatPage from "@/pages/ChatPage";' in page
    assert "<ChatPage isActive={activeSection === \"local-chat\"} sessionQuotaRecommendation={dashboardStatus?.session_quota_recommendation ?? null} />" in page
    assert 'hidden={activeSection !== "local-chat"}' in page
    assert "Dominant local PTY-backed Hermes chat" in page
    assert "External delivery, new routing targets, attachments, and audio/voice paths are unavailable here" in page
    assert "Risky external send" in page
    assert "Send externally" in page
    assert "Attachments disabled" in page
    assert "Voice disabled" in page

    for marker in FORBIDDEN_PAGE_MARKERS:
        assert marker.lower() not in lowered


def test_session_quota_warning_primary_control_lives_in_chat_section_and_uses_local_pty_path():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    chat = CHAT_PAGE.read_text(encoding="utf-8")

    chat_section = page[page.index('data-testid="cockpit-section-local-chat"') : page.index('data-testid="cockpit-section-automation-health"')]
    brief_section = page[page.index('data-testid="cockpit-section-overview"') : page.index('data-testid="cockpit-section-local-chat"')]

    assert "sessionQuotaRecommendation={dashboardStatus?.session_quota_recommendation ?? null}" in chat_section
    assert "cockpit-session-quota-recommendation" in brief_section
    assert "chat-session-quota-warning-card" not in brief_section
    assert 'data-testid="chat-session-quota-warning-card"' in chat
    assert "Not now" in chat
    assert "Yes — start fresh" in chat
    assert "window.localStorage.setItem(quotaDismissStorageKey(quotaDedupeKey), \"1\")" in chat
    assert "sendSlashCommand(\"/new\")" in chat
    assert "ws.send(command)" in chat
    assert "current.send(\"\\r\")" in chat
    assert "chat-session-quota-command-error" in chat
    assert "Local Chat is not connected yet" in chat
    assert "deleteSession" not in chat
    assert "session files" not in chat.lower()


def test_chat_session_quota_warning_stays_adjacent_to_composer_without_remounting_pty():
    chat = CHAT_PAGE.read_text(encoding="utf-8")

    card_index = chat.index('data-testid="chat-session-quota-warning-card"')
    terminal_index = chat.index('data-testid="chat-local-pty-terminal-pane"')
    host_index = chat.index('data-testid="chat-local-pty-host"')

    assert card_index < terminal_index < host_index
    assert 'data-testid="chat-session-quota-not-now"' in chat
    assert 'data-testid="chat-session-quota-start-fresh"' in chat
    assert 'aria-label="Dismiss session quota recommendation for this threshold"' in chat
    assert 'aria-label="Start a fresh local Hermes session through Local Chat"' in chat
    start_fresh_block = chat[
        chat.index("const startFreshFromQuotaWarning") : chat.index(
            "useEffect(() => {",
            chat.index("const startFreshFromQuotaWarning"),
        )
    ]
    assert 'if (sendSlashCommand("/new")) {' in start_fresh_block
    assert "dismissQuotaWarning();" in start_fresh_block
    assert "}, [dismissQuotaWarning, sendSlashCommand]);" in start_fresh_block
    assert "setReconnectKey" not in start_fresh_block
    assert "ws.close()" not in start_fresh_block


def test_cockpit_ops_checks_section_surfaces_n8n_daily_checks_read_only():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")

    assert '"ops-checks"' not in page
    assert "Daily n8n checks" in page
    assert "Read-only n8n daily check outputs" in page
    assert "function N8nChecksPanel" in page
    assert "data-testid=\"cockpit-health-n8n-details\"" in page
    assert "data-testid=\"cockpit-section-ops-checks\"" not in page
    assert "Morning Briefing" in page
    assert "Workflow Health Daily Report" in page
    assert "Auto-Remediation Monitor" in page
    assert "Immich Nightly Sync Monitor" in page
    assert "Obsidian Inbox Processor" in page
    assert "Alexa Bring Sync" in page
    assert "n8n Nightly Workflow Backup" in page
    assert "No workflow triggers, Discord sends, retries, repairs, voice, attachments, or routing are available here" in page
    assert "actions_enabled" in page
    assert "external_delivery_enabled" in page
    assert "grid-cols-7" in page
    assert "min-w-[960px]" in page
    assert "live n8n" in page
    assert "fallback/stale" in page
    assert "Output summary" in page


def test_cockpit_home_prioritizes_now_attention_activity_and_local_chat():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    lowered = page.lower()

    assert "standalone" in page
    assert "Command brief" in page
    assert "What needs Marco attention now" in page
    assert "Recent changes" in page
    assert "Active work" in page
    assert "System health" in page
    assert "Safe gated action" in page
    assert "PTY command surface" in page
    assert "Local Chat" in page
    assert "No active work is visible" in page
    assert "Command surface offline" not in page
    assert "Chat placeholder" not in page
    assert "aria-disabled=\"true\"" not in page
    assert "safe-area-inset-top" in page
    assert "safe-area-inset-bottom" in page
    assert "near-black" in lowered or "var(--cockpit-shell)" in page
    assert "<details" in page and "Diagnostics" in page
    assert "You normally do not need this for daily reading" in page
    assert "Recent window" in page
    assert "bounded recent cockpit window" in page
    assert "recent bounded window" in page
    assert "ChevronDown" in page
    assert "group-open:rotate-180" in page
    assert "Nothing is asking for Marco right now" in page
    assert "Automation Health needs review" in page
    assert "automationHealth?.summary?.headline" in page
    assert "totalAttentionCount" in page
    assert "Open Actions / read-only upgrade review" in page
    assert "Actions that need confirmation" in page
    assert "Ask Biff to investigate" in page
    assert "Health action ready" in page
    assert "Upgrade brief" in page
    assert "Major improvements" in page
    assert "cockpit-upgrade-brief" in page
    assert "why_this_matters" in page
    assert "Should we upgrade?" in page
    assert "cockpit-upgrade-recommendation" in page
    assert "Do not upgrade blindly" in page
    assert "Wait; radar is not trustworthy enough for an upgrade call" in page
    assert "Last radar basis" in page
    assert "Freshness:" in page
    assert "Basis signals" in page
    assert "formatClockLabel" in page
    assert "formatFreshness" in page
    assert "Last refreshed" in page
    assert "Not available" in page
    assert "Archive/Context" in page


def test_cockpit_overview_reads_as_command_brief_not_telemetry():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    overview = page[page.index('data-testid="cockpit-section-overview"') : page.index('data-testid="cockpit-section-local-chat"')]
    overview_lower = overview.lower()

    for section in ("Attention", "Recent changes", "Active work", "System health", "Safe gated action"):
        assert section in overview
    assert "Command brief" in overview
    assert "What needs Marco attention now" in overview
    assert "What changed recently" in overview
    assert "What is running and healthy" in overview
    assert "Safe gated action" in overview

    assert "Operational now" not in overview
    assert "Operational attention" not in overview
    assert "Signal activity" not in overview
    assert "Real signal model" not in overview
    assert "<SignalSummary" not in overview
    assert "Math.round" not in overview
    assert "confidence" not in overview_lower
    assert "sessiondb" not in overview_lower
    assert "lane_message" not in overview
    assert "provenance" not in overview_lower


def test_cockpit_overview_surfaces_session_quota_recommendation():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "session_quota_recommendation" in api
    assert "SessionQuotaRecommendation" in api
    assert 'data-testid="cockpit-session-quota-recommendation"' in page
    assert "Session quota:" in page
    assert "dashboardStatus.session_quota_recommendation.text" in page


def test_cockpit_surfaces_active_biff_operating_mode():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "biff_operating_mode" in api
    assert "BiffOperatingMode" in api
    assert 'data-testid="cockpit-biff-mode-badge"' in page
    assert 'data-testid="cockpit-biff-mode-card"' in page
    assert "Biff mode ·" in page


def test_cockpit_overview_surfaces_restart_safe_self_work_handoff():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "CockpitSelfWorkHandoffResponse" in api
    assert "CockpitSelfWorkHandoffRecord" in api
    assert "SelfWorkHandoffPanel" in page
    assert "setSelfWorkHandoff" in page
    assert "api.getCockpitSelfWorkHandoff" in page
    assert "Where Biff was before restart" in page
    assert 'data-testid="cockpit-self-work-handoff"' in page
    assert "Last action:" in page
    assert "Next safe step:" in page
    assert "Pending:" in page
    assert "Known failures:" in page


def test_cockpit_selected_lane_detail_loads_bounded_display_safe_messages():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")

    assert "function LaneDetailPanel" in page
    assert "Selected lane" in page
    assert "selectedLaneMessages" in page
    assert "selectedLaneLoading" in page
    assert "selectedLaneError" in page
    assert "api.getCockpitLaneMessages(laneId, { limit: LANE_MESSAGE_LIMIT, offset: 0 })" in page
    assert "}, [selectedLane?.lane_id]);" in page
    assert "Showing display-safe recent messages only" in page
    assert "No display-safe recent messages are visible for this lane" in page
    assert "Could not load selected lane messages" in page
    assert "Loading selected lane messages" in page
    assert "raw identifiers, keys, and media payloads are redacted or hidden" in page


def test_cockpit_selected_lane_detail_hides_sensitive_message_fields():
    page = COCKPIT_PAGE.read_text(encoding="utf-8")
    lowered = page.lower()

    assert "function safeText" in page
    assert "function boundedCopy" in page
    assert "function messageBody" in page
    assert "[redacted key]" in page
    assert "[redacted id]" in page
    assert "Internal execution details are hidden" in page
    assert "role === \"tool\" || role === \"system\"" in page
    assert "{lane.lane_id}</" not in page
    assert "{lane.session_id}</" not in page
    assert "reasoning" not in lowered
    assert "tool args" not in lowered
