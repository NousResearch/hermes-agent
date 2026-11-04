"""Closed desktop action ids shared by collection and export validation."""

# Button presses per day. Action ids are the Desktop keybinding registry (KEYBIND_ACTIONS in
# apps/desktop/src/lib/keybinds/actions.ts) plus the typed button table DESKTOP_BUTTON_ACTIONS in
# apps/desktop/src/store/desktop-metrics.ts; contributed/plugin actions collapse to `other`.
DESKTOP_ACTION_USE_MARK = DESKTOP_ACTION_USE_METRIC = "hermes.desktop.action_use"
DESKTOP_KEYBIND_ACTION_IDS = frozenset({
    "appearance.toggleMode", "composer.cancel", "composer.dictate", "composer.focus", "composer.help",
    "composer.history", "composer.mention", "composer.modelPicker", "composer.newline", "composer.queue",
    "composer.reasoningDown", "composer.reasoningUp", "composer.send", "composer.sendQueued", "composer.slash",
    "composer.steer", "composer.voice", "conversation.scrollPageDown", "conversation.scrollPageUp",
    "hud.snapToPointer", "keybinds.openPanel", "nav.agents", "nav.artifacts", "nav.capabilities",
    "nav.commandCenter", "nav.commandPalette", "nav.cron", "nav.messaging", "nav.profiles", "nav.settings",
    "profile.create", "profile.default", "profile.next", "profile.prev", "profile.toggleAll",
    "session.archive", "session.focusSearch", "session.new", "session.newTab", "session.newWindow",
    "session.next", "session.prev", "session.togglePin", "view.closeTab", "view.closeTerminal",
    "view.cycleSidebarGrouping", "view.findInPage", "view.findNext", "view.findPrevious", "view.flipPanes",
    "view.newTerminal", "view.nextTerminal", "view.prevTerminal", "view.reopenTab", "view.selectionToComposer",
    "view.showBrowser", "view.showFiles", "view.showTerminal", "view.terminalCopy", "view.terminalPaste",
    "view.toggleHud", "view.toggleProfileRail", "view.toggleReview", "view.toggleRightSidebar",
    "view.toggleSidebar", "view.toggleSimpleMode", "view.toggleStatusbar", "view.toggleTabStrip",
    "workspace.newWorktree", "workspace.openFolder",
})
DESKTOP_BUTTON_ACTION_IDS = frozenset({
    "composer.attach", "message.copy", "message.retry", "sidebar.simplified.enable_offer",
    "sidebar.simplified.enable_settings", "sidebar.simplified.decline", "sidebar.simplified.dismiss",
    "sidebar.simplified.disable",
})
# Content-free IA counters; mirrored by store/ia-metrics-contract.ts in Desktop.
DESKTOP_IA_ACTIONS = {
    "rail": ('chat', 'profiles', 'capabilities', 'artifacts', 'cron', 'plugin'),
    "drawer": ('opened', 'settings', 'account', 'credit-card', 'remote', 'key', 'paintcan', 'mic', 'keyboard', 'shield', 'info', 'question', 'tools', 'documentation'),
    "session": ('opened_pinned', 'opened_agent', 'opened_search'),
    "section": ('pinned_expanded', 'pinned_collapsed', 'agent_expanded', 'agent_collapsed', 'gateway_expanded', 'gateway_collapsed'),
    "search": ('opened', 'used', 'results', 'empty', 'partial_failure', 'selected', 'cleared', 'early'),
    "pin": ('pinned_request', 'agent_request', 'search_request', 'pinned', 'unpinned', 'failed'),
    "new_session": ('sidebar', 'command', 'agent_row', 'picker_opened', 'picker_on_new', 'specific', 'default', 'dismissed'),
    "gateway": ('popover_open', 'popover_connecting', 'popover_offline', 'system', 'logs', 'reconnect', 'offline_agent'),
}
DESKTOP_IA_AREAS = (
    "chat", "profiles", "capabilities", "artifacts", "cron", "settings", "command-center", "plugin", "other",
)
DESKTOP_IA_ACTION_IDS = frozenset(
    f"ia.{group}.{event}" for group, events in DESKTOP_IA_ACTIONS.items() for event in events
) | frozenset(
    f"ia.{group}.{area}" for group in ("view", "backtrack") for area in DESKTOP_IA_AREAS
) | frozenset(
    f"ia.transition.{area}_to_{other}" for area in DESKTOP_IA_AREAS for other in DESKTOP_IA_AREAS if area != other
) | frozenset(
    f"ia.dwell.{area}_{bucket}" for area in DESKTOP_IA_AREAS
    for bucket in ("lt5s", "5s_to30s", "30s_to5m", "gte5m")
)
DESKTOP_ACTION_IDS = DESKTOP_KEYBIND_ACTION_IDS | DESKTOP_BUTTON_ACTION_IDS | DESKTOP_IA_ACTION_IDS | {"other"}
DESKTOP_ACTION_VIAS = frozenset({"click", "menu", "palette", "shortcut", "state"})
