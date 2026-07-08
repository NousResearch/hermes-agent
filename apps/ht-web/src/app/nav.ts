// Primary navigation registry. Management pages register here and are routed
// in AppShell; keeping the list in one place makes porting a page a two-line
// change (add the entry + the <Route>).
export interface NavItem {
  to: string;
  label: string;
  glyph: string;
  /** "chat" is the agent conversation; "manage" are REST-backed admin pages. */
  group: "chat" | "manage";
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Chat", glyph: "◆", group: "chat" },
  { to: "/sessions", label: "Sessions", glyph: "≡", group: "manage" },
  { to: "/models", label: "Models", glyph: "◇", group: "manage" },
  { to: "/config", label: "Config", glyph: "⚙", group: "manage" },
  { to: "/logs", label: "Logs", glyph: "▤", group: "manage" },
  { to: "/system", label: "System", glyph: "◉", group: "manage" },
];
