// The single source of truth for rebindable desktop hotkeys.
//
// Each entry is pure metadata: an id, a category, and the default combo(s).
// Handlers are wired separately in `use-keybinds.ts` (they need React context
// like navigate / theme); labels come from i18n (`t.keybinds.actions[id]`). To
// add a hotkey, add a row here and a handler there — nothing else.

import { IS_MAC } from './combo'

export type KeybindCategory = 'composer' | 'profiles' | 'session' | 'navigation' | 'view'

// The self-referential opener — bound + dispatched like any action, but shown in
// the panel subtitle (not as its own row).
export const KEYBIND_PANEL_ACTION = 'keybinds.openPanel'

// `composer` is read-only; the rest are rebindable. `view` is the catch-all for
// layout, appearance, and the panel-opener.
export const KEYBIND_CATEGORIES: readonly KeybindCategory[] = ['composer', 'profiles', 'session', 'navigation', 'view']

export interface KeybindActionMeta {
  id: string
  category: KeybindCategory
  /** Default combos. Empty = shipped unbound (user can assign one). */
  defaults: readonly string[]
}

// Positional switch slots for *named* profiles: ⌘1…⌘9 for profiles 1-9, then
// ⌘⌥1…⌘⌥9 for 10-18. The default profile gets the two-key mnemonic ⌘D (see
// `profile.default`) — ⌘` is macOS-reserved (window cycling) and ⌘0 is reset-zoom.
export const PROFILE_SLOT_COUNT = 18

function comboForSlot(slot: number): string {
  return slot <= 9 ? `mod+${slot}` : `mod+alt+${slot - 9}`
}

const PROFILE_SWITCH_ACTIONS: KeybindActionMeta[] = Array.from({ length: PROFILE_SLOT_COUNT }, (_, i) => ({
  id: `profile.switch.${i + 1}`,
  category: 'profiles' as const,
  defaults: [comboForSlot(i + 1)]
}))

// ⌘` on macOS / Ctrl+` elsewhere (the `~` key), plus the Shift/tilde variant.
// `mod` keeps one binding cross-platform; on macOS this shadows the system
// window-cycler, which is fine for a single-window app.
const TERMINAL_TOGGLE_DEFAULTS = ['mod+`', 'mod+shift+`']

// Positional jumps — ^1…^9, mirroring profiles' ⌘1…⌘9.
export const SESSION_SLOT_COUNT = 9

const SESSION_SLOT_ACTIONS: KeybindActionMeta[] = Array.from({ length: SESSION_SLOT_COUNT }, (_, i) => ({
  id: `session.slot.${i + 1}`,
  category: 'session' as const,
  defaults: [`ctrl+${i + 1}`]
}))

export const KEYBIND_ACTIONS: readonly KeybindActionMeta[] = [
  // ── Composer ─────────────────────────────────────────────────────────────
  { id: 'composer.focus', category: 'composer', defaults: [] },
  { id: 'composer.modelPicker', category: 'composer', defaults: [] },
  // Voice conversation toggle. Matches the documented `voice.record_key`
  // (Ctrl+B). On macOS that's literally ⌃B — distinct from the ⌘B sidebar
  // toggle. Off macOS `ctrl` folds to `mod`, which IS the ⌘B/Ctrl+B sidebar
  // chord, so ship it unbound there (rebindable in the panel) rather than
  // stealing the long-standing sidebar binding.
  { id: 'composer.voice', category: 'composer', defaults: IS_MAC ? ['ctrl+b'] : [] },

  // ── Profiles ─────────────────────────────────────────────────────────────
  { id: 'profile.default', category: 'profiles', defaults: ['mod+d'] },
  ...PROFILE_SWITCH_ACTIONS,
  { id: 'profile.next', category: 'profiles', defaults: ['mod+shift+]'] },
  { id: 'profile.prev', category: 'profiles', defaults: ['mod+shift+['] },
  { id: 'profile.toggleAll', category: 'profiles', defaults: ['mod+shift+0'] },
  { id: 'profile.create', category: 'profiles', defaults: [] },

  // ── Session ──────────────────────────────────────────────────────────────
  { id: 'session.new', category: 'session', defaults: ['mod+n', 'shift+n'] },
  { id: 'session.newWindow', category: 'session', defaults: ['mod+shift+n'] },
  // ⌃Tab / ⌃⇧Tab — the universal tab-cycle chord. Literally Control, not Cmd
  // (macOS reserves Cmd+Tab for app switching); see `ctrl` in combo.ts.
  { id: 'session.next', category: 'session', defaults: ['ctrl+tab'] },
  { id: 'session.prev', category: 'session', defaults: ['ctrl+shift+tab'] },
  ...SESSION_SLOT_ACTIONS,
  { id: 'session.focusSearch', category: 'session', defaults: ['mod+shift+f'] },
  { id: 'session.togglePin', category: 'session', defaults: [] },
  // ⌘⇧B — "b" for branch: spin up a new git worktree from the active repo.
  { id: 'workspace.newWorktree', category: 'session', defaults: ['mod+shift+b'] },

  // ── Navigation ───────────────────────────────────────────────────────────
  { id: 'nav.commandPalette', category: 'navigation', defaults: ['mod+k', 'mod+p'] },
  { id: 'nav.commandCenter', category: 'navigation', defaults: ['mod+.'] },
  { id: 'nav.settings', category: 'navigation', defaults: ['mod+,'] },
  { id: 'nav.profiles', category: 'navigation', defaults: [] },
  { id: 'nav.skills', category: 'navigation', defaults: [] },
  { id: 'nav.messaging', category: 'navigation', defaults: [] },
  { id: 'nav.artifacts', category: 'navigation', defaults: [] },
  { id: 'nav.cron', category: 'navigation', defaults: [] },
  { id: 'nav.agents', category: 'navigation', defaults: [] },

  // ── View (layout + appearance + the shortcuts panel itself) ───────────────
  { id: 'view.toggleSidebar', category: 'view', defaults: ['mod+b'] },
  { id: 'view.toggleRightSidebar', category: 'view', defaults: ['mod+j'] },
  // ⌘G — "g" for git; the review pane is the source-control view.
  { id: 'view.toggleReview', category: 'view', defaults: ['mod+g'] },
  { id: 'view.showFiles', category: 'view', defaults: [] },
  { id: 'view.showTerminal', category: 'view', defaults: TERMINAL_TOGGLE_DEFAULTS },
  // Embedded browser pane. Avoids ⌘⇧B/Ctrl+Shift+B because Hermes already
  // ships that chord for new worktrees/branches.
  { id: 'view.openBrowser', category: 'view', defaults: ['mod+alt+b'] },
  // ⌘\ — the backslash reads like a mirror line flipping the layout.
  { id: 'view.flipPanes', category: 'view', defaults: ['mod+\\'] },
  { id: 'appearance.toggleMode', category: 'view', defaults: ['shift+x'] },
  { id: 'keybinds.openPanel', category: 'view', defaults: ['mod+/'] }
]

export const KEYBIND_ACTION_IDS: readonly string[] = KEYBIND_ACTIONS.map(action => action.id)

const ACTION_BY_ID = new Map(KEYBIND_ACTIONS.map(action => [action.id, action]))

export function keybindAction(id: string): KeybindActionMeta | undefined {
  return ACTION_BY_ID.get(id)
}

export type KeybindBindings = Record<string, string[]>

export function defaultBindings(): KeybindBindings {
  return Object.fromEntries(KEYBIND_ACTIONS.map(action => [action.id, [...action.defaults]]))
}

// Fixed, non-rebindable shortcuts surfaced read-only in the panel so the map is
// complete. `keys` are canonical tokens run through `formatCombo` for display
// (single symbols like "@" / "/" pass through unchanged). Categories listed here
// render after the rebindable ones.
export interface KeybindReadonly {
  id: string
  category: KeybindCategory
  keys: readonly string[]
}

export const KEYBIND_READONLY: readonly KeybindReadonly[] = [
  { id: 'composer.send', category: 'composer', keys: ['enter'] },
  { id: 'composer.newline', category: 'composer', keys: ['shift+enter'] },
  { id: 'composer.steer', category: 'composer', keys: ['mod+enter'] },
  { id: 'composer.sendQueued', category: 'composer', keys: ['mod+shift+k'] },
  { id: 'composer.mention', category: 'composer', keys: ['@'] },
  { id: 'composer.slash', category: 'composer', keys: ['/'] },
  { id: 'composer.help', category: 'composer', keys: ['?'] },
  { id: 'composer.history', category: 'composer', keys: ['up', 'down'] },
  { id: 'composer.cancel', category: 'composer', keys: ['escape'] },
  // Fixed, context-local shortcuts surfaced for discoverability.
  { id: 'view.terminalSelection', category: 'view', keys: ['mod+l'] },
  { id: 'view.closePreviewTab', category: 'view', keys: ['mod+w'] }
]
