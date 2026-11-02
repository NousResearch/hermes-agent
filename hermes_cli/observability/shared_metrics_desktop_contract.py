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
DESKTOP_ACTION_IDS = DESKTOP_KEYBIND_ACTION_IDS | DESKTOP_BUTTON_ACTION_IDS | {"other"}
DESKTOP_ACTION_VIAS = frozenset({"click", "menu", "palette", "shortcut"})
