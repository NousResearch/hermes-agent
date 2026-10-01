export type Shortcut = { id: string; label: string; keys: string[]; group: string; note?: string };

/** Single source of truth — drives the ⌘/ overlay, Settings → Keyboard and the status bar. */
export const SHORTCUTS: Shortcut[] = [
  { id: "palette", label: "Command palette", keys: ["⌘", "K"], group: "Global" },
  { id: "shortcuts", label: "Keyboard shortcuts", keys: ["⌘", "/"], group: "Global" },
  { id: "settings", label: "Settings (toggle)", keys: ["⌘", ","], group: "Global" },
  { id: "close-panel", label: "Close panel · back to chat", keys: ["Esc"], group: "Global", note: "Or click the same sidebar item again" },
  { id: "switch-product", label: "Switch Code / Agent", keys: ["⌘", "E"], group: "Global" },
  { id: "new", label: "New session", keys: ["⌘", "N"], group: "Global" },
  { id: "todo", label: "Todo list", keys: ["⌘", "T"], group: "Global" },
  { id: "voice", label: "Dictate", keys: ["⌘", "⇧", "V"], group: "Composer", note: "Speech → text in the composer" },

  { id: "v-threads", label: "Threads", keys: ["⌘", "1"], group: "Navigate" },
  { id: "v-tasks", label: "Tasks", keys: ["⌘", "2"], group: "Navigate" },
  { id: "v-skills", label: "Skills", keys: ["⌘", "3"], group: "Navigate" },
  { id: "v-connectors", label: "Connectors", keys: ["⌘", "4"], group: "Navigate" },
  { id: "v-projects", label: "Projects", keys: ["⌘", "5"], group: "Navigate" },
  { id: "v-pane", label: "Cycle context pane section", keys: ["⌘", "["], group: "Navigate", note: "Plan · Changes · Context · Fleet · Git · Browser · Editors · Terminal" },

  { id: "mode", label: "Cycle permission mode", keys: ["⇧", "Tab"], group: "Composer", note: "Plan · Read only · Agent · Full access" },
  { id: "send", label: "Send", keys: ["⏎"], group: "Composer" },
  { id: "newline", label: "New line", keys: ["⇧", "⏎"], group: "Composer" },
  { id: "commands", label: "Slash commands", keys: ["/"], group: "Composer", note: "Type at an empty prompt" },
  { id: "mention", label: "Add context", keys: ["@"], group: "Composer", note: "File · diff · editor · browser · terminal" },
  { id: "compact", label: "Compact context", keys: ["⌘", "⇧", "K"], group: "Composer" },

  { id: "stop", label: "Stop the agent", keys: ["Esc"], group: "Agent" },
  { id: "rewind", label: "Rewind to checkpoint", keys: ["Esc", "Esc"], group: "Agent" },
  { id: "approve", label: "Allow pending approval", keys: ["⌘", "⏎"], group: "Agent" },
  { id: "fork", label: "Fork into parallel run", keys: ["⌘", "D"], group: "Agent" },

  { id: "layout-sidebar", label: "Sidebar", keys: ["⌘", "⇧", "B"], group: "Layout" },
  { id: "sidebar-nav", label: "Move through threads", keys: ["↑", "↓"], group: "Layout", note: "⏎ open · ⌫ archive · Home/End jump" },
  { id: "layout-terminal", label: "Terminal", keys: ["⌘", "J"], group: "Layout" },
  { id: "layout-panel", label: "Side panel / Agent panel", keys: ["⌘", "L"], group: "Layout" },
  { id: "layout-inspector", label: "Inspector (in session)", keys: ["⌘", "\\"], group: "Layout" },
  { id: "layout", label: "Layout control", keys: ["⌘", "⇧", "L"], group: "Layout" },
  { id: "layout-focus", label: "Focus mode", keys: ["⌘", "⇧", "F"], group: "Layout", note: "Hide every panel" },
];

export const SHORTCUT_GROUPS = ["Global", "Navigate", "Composer", "Agent", "Layout"] as const;
export const byId = (id: string) => SHORTCUTS.find((s) => s.id === id);
