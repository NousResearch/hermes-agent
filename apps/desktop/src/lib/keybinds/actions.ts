// The single source of truth for rebindable desktop hotkeys.
//
// Each entry is pure metadata: an id, a category, and the default combo(s).
// Handlers are wired separately in `use-keybinds.ts` (they need React context
// like navigate / theme); labels come from i18n (`t.keybinds.actions[id]`). To
// add a hotkey, add a row here and a handler there — nothing else.

import { registry } from '@/contrib/registry'

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
  /** Display label for CONTRIBUTED actions (built-ins use i18n). */
  label?: string
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

// Positional jumps — ^1…^9, mirroring profiles' ⌘1…⌘9.
export const SESSION_SLOT_COUNT = 9

const SESSION_SLOT_ACTIONS: KeybindActionMeta[] = Array.from({ length: SESSION_SLOT_COUNT }, (_, i) => ({
  id: `session.slot.${i + 1}`,
  category: 'session' as const,
  defaults: [`ctrl+${i + 1}`]
}))

export const KEYBIND_ACTIONS: readonly KeybindActionMeta[] = [
  // ── Composer ─────────────────────────────────────────────────────────────
  // Soft `/` / Enter focus (gated); other printables type-to-focus unbound.
  { id: 'composer.focus', category: 'composer', defaults: ['/', 'enter'] },
  // ⌘⇧M — "m" for model; the convention chat apps converged on (LibreChat,
  // Open WebUI, and Cherry Studio all ship the same chord). Opens the pill's
  // live dropdown on the pane under the pointer, else the active composer.
  { id: 'composer.modelPicker', category: 'composer', defaults: ['mod+shift+m'] },
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
  // `shift+n` was dropped from the defaults (#76185): a bare shifted letter
  // hijacked normal typing — pressing uppercase N outside an input (or via an
  // IME) created a new session unexpectedly. The deliberate ⌘/Ctrl+N chord
  // stays; users who liked ⇧N can rebind it in the panel.
  { id: 'session.new', category: 'session', defaults: ['mod+n'] },
  { id: 'session.newTab', category: 'session', defaults: ['mod+t'] },
  { id: 'session.newWindow', category: 'session', defaults: ['mod+shift+n'] },
  // ⌃Tab / ⌃⇧Tab — the universal tab-cycle chord. Literally Control, not Cmd
  // (macOS reserves Cmd+Tab for app switching); see `ctrl` in combo.ts.
  { id: 'session.next', category: 'session', defaults: ['ctrl+tab', 'ctrl+pagedown'] },
  { id: 'session.prev', category: 'session', defaults: ['ctrl+shift+tab', 'ctrl+pageup'] },
  ...SESSION_SLOT_ACTIONS,
  { id: 'session.focusSearch', category: 'session', defaults: ['mod+shift+f'] },
  { id: 'session.togglePin', category: 'session', defaults: [] },
  // Archive the active session. Ships unbound (like `session.togglePin`) so an
  // irreversible-feeling, mouse-only action doesn't silently claim a chord for
  // every user — surfaced in the panel for opt-in binding (the issue suggests
  // ⌘⇧⌫ / Ctrl+Shift+⌫).
  { id: 'session.archive', category: 'session', defaults: [] },
  // ⌘⇧B — "b" for branch: spin up a new git worktree from the active repo.
  { id: 'workspace.newWorktree', category: 'session', defaults: ['mod+shift+b'] },
  // ⌘O — the editor-standard "open folder" chord (VS Code ⌘O, Zed's
  // workspace::Open). Picks a folder and opens it as a project (upsert:
  // enters the owning project when one exists, else creates one), landing on
  // a fresh session anchored there.
  { id: 'workspace.openFolder', category: 'session', defaults: ['mod+o'] },

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
  // ⌘⇧S — "s" for status bar. VS Code ships
  // `workbench.action.toggleStatusbarVisibility` unbound (it's a chord-free
  // gap in their View family) and Hermes has no chord dispatcher, so this
  // takes the nearest free single combo instead of a ⌘K ⌘S two-stroke.
  { id: 'view.toggleStatusbar', category: 'view', defaults: ['mod+shift+s'] },
  // ⌥⌘T — "t" for tabs, reaching past ⇧ because ⌘⇧T is reopen-closed-tab
  // everywhere. Ships BOUND, unlike VS Code's settings-only tab-bar switch:
  // here the hide can take away every other affordance the zone had, so the
  // way back has to already exist. (⌥+letter emits a symbol on macOS; the
  // binding resolves through KeyT via comboFromEvent's `event.code` fallback.)
  { id: 'view.toggleTabStrip', category: 'view', defaults: ['mod+alt+t'] },
  // ⌘G — "g" for git; the review pane is the source-control view.
  { id: 'view.toggleReview', category: 'view', defaults: ['mod+g'] },
  { id: 'view.showFiles', category: 'view', defaults: [] },
  // ⌘⇧L — "L" for location, the address-bar chord every browser shares. Plain
  // ⌘L is the terminal's selection shortcut, hence the shift.
  { id: 'view.showBrowser', category: 'view', defaults: ['mod+shift+l'] },
  // ⌘⇧H — "h" for HUD. Enters/leaves the chrome-free floating chat: the app
  // window steps aside and a composer + live reply float over whatever the
  // user is working in. Ships bound because the whole point is leaving the app
  // without reaching for it — but the titlebar button is the discoverable door.
  { id: 'view.toggleHud', category: 'view', defaults: ['mod+shift+h'] },
  // ⌥⌘F — "f" for follow (⌘⇧F is session search). Lazy follow-the-pointer
  // for the HUD bar; a pref main owns, so it works from the app window too.
  { id: 'hud.toggleFollow', category: 'view', defaults: ['mod+alt+f'] },
  // Control+` everywhere (literal `ctrl`, NOT `mod`): ⌘` is macOS-reserved for
  // cycling app windows, so VS Code/Cursor/Zed bind the terminal to Ctrl+` on
  // every platform. Off macOS `ctrl` folds to `mod` (= Ctrl), so it's unchanged.
  // Toggle reveals the terminal (opening one if none exist); Shift spawns a new one.
  { id: 'view.showTerminal', category: 'view', defaults: ['ctrl+`'] },
  { id: 'view.newTerminal', category: 'view', defaults: ['ctrl+shift+`'] },
  // Same Ctrl(+Shift) terminal family: arrows walk the (vertical) tab rail, W
  // kills the active one. ⌘W is taken (close preview tab) and ⌘⇧[ ] are profiles,
  // so these stay on `ctrl` — distinct on macOS, folding to Ctrl elsewhere.
  { id: 'view.nextTerminal', category: 'view', defaults: ['ctrl+shift+down'] },
  { id: 'view.prevTerminal', category: 'view', defaults: ['ctrl+shift+up'] },
  { id: 'view.closeTerminal', category: 'view', defaults: ['ctrl+shift+w'] },
  // ⌘\ — the backslash reads like a mirror line flipping the layout.
  { id: 'view.flipPanes', category: 'view', defaults: ['mod+\\'] },
  // ⌘W closes the focused zone's active tab — its own tab strip (preview) or
  // the tree tab (session tiles, files, terminal). The uncloseable workspace
  // is a no-op. ⌘⇧T reopens the last closed tab where it was.
  { id: 'view.closeTab', category: 'view', defaults: ['mod+w'] },
  { id: 'view.reopenTab', category: 'view', defaults: ['mod+shift+t'] },
  // ⌘F — open the find-in-page bar. `actionAllowedInInput` lets this action
  // fire from inside a textarea / contenteditable (matches browser behavior
  // so typing in the composer and pressing ⌘F focuses find, not 'f').
  { id: 'view.findInPage', category: 'view', defaults: ['mod+f'] },
  // ⌘G / ⌘⇧G step matches — the platform-standard find-next/find-previous
  // pair (Chrome, Safari, VS Code, and Claude Desktop all ship it). No
  // `defaults` here on purpose: ⌘G already belongs to `view.toggleReview`,
  // and shipping a duplicate default would flag a permanent conflict in the
  // keybinds panel. While the find bar is OPEN, its capture-phase listener
  // claims ⌘G/⌘⇧G and stops propagation (see components/find-bar.tsx), so
  // stepping works out of the box and the review toggle keeps the key the
  // rest of the time. These entries exist so the panel documents the pair
  // and a user who prefers a dedicated chord can bind one.
  { id: 'view.findNext', category: 'view', defaults: [] },
  { id: 'view.findPrevious', category: 'view', defaults: [] },
  { id: 'appearance.toggleMode', category: 'view', defaults: ['shift+x'] },
  { id: 'keybinds.openPanel', category: 'view', defaults: ['mod+/'] }
]

export const KEYBIND_ACTION_IDS: readonly string[] = KEYBIND_ACTIONS.map(action => action.id)

const ACTION_BY_ID = new Map(KEYBIND_ACTIONS.map(action => [action.id, action]))

// ── Contributed actions — the `keybinds` registry area ──────────────────────
// Same declarative schema as every other surface: a data contribution carries
// the action's metadata AND its handler. Contributed actions are first-class:
// they dispatch, appear in the panel, are rebindable, and their overrides
// persist exactly like built-ins. Built-in ids can't be shadowed.

export const KEYBINDS_AREA = 'keybinds'

/** Payload of a `keybinds` data contribution. */
export interface KeybindContribution {
  id: string
  /** Panel section. Defaults to `view`. */
  category?: KeybindCategory
  /** Default combos (canonical form, e.g. `mod+shift+\\`). Empty = unbound. */
  defaults?: readonly string[]
  label: string
  run: () => void
}

export function contributedKeybinds(): KeybindContribution[] {
  return registry
    .getArea(KEYBINDS_AREA)
    .map(c => c.data as KeybindContribution)
    .filter(k => Boolean(k?.id && k.label) && typeof k?.run === 'function' && !ACTION_BY_ID.has(k.id))
}

/** Built-ins + contributed, one metadata list (panel, bindings, conflicts). */
export function allKeybindActions(): KeybindActionMeta[] {
  return [
    ...KEYBIND_ACTIONS,
    ...contributedKeybinds().map(k => ({
      id: k.id,
      category: k.category ?? ('view' as const),
      defaults: k.defaults ?? [],
      label: k.label
    }))
  ]
}

export function keybindAction(id: string): KeybindActionMeta | undefined {
  return ACTION_BY_ID.get(id) ?? allKeybindActions().find(action => action.id === id)
}

/** The contributed handler for an action id (built-ins wire theirs in use-keybinds). */
export function contributedKeybindHandler(id: string): (() => void) | undefined {
  return contributedKeybinds().find(k => k.id === id)?.run
}

export type KeybindBindings = Record<string, string[]>

export function defaultBindings(): KeybindBindings {
  return Object.fromEntries(allKeybindActions().map(action => [action.id, [...action.defaults]]))
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
  { id: 'composer.steer', category: 'composer', keys: ['enter'] },
  { id: 'composer.queue', category: 'composer', keys: ['mod+enter'] },
  { id: 'composer.sendQueued', category: 'composer', keys: ['mod+shift+k'] },
  { id: 'composer.mention', category: 'composer', keys: ['@'] },
  { id: 'composer.slash', category: 'composer', keys: ['/'] },
  { id: 'composer.help', category: 'composer', keys: ['?'] },
  { id: 'composer.history', category: 'composer', keys: ['up', 'down'] },
  { id: 'composer.cancel', category: 'composer', keys: ['escape'] },
  // ⌘/Ctrl+L moves focus to the composer from anywhere, like the address-bar
  // chord in a browser. The row reuses the id of the rebindable soft-focus
  // action above. As a result, the panel shows one "Focus composer" label
  // for both. The row is fixed because the selection shortcut below uses the
  // same chord. Who claims a contested press: see the priority ladder in
  // app/chat/composer/focus-chord.ts.
  { id: 'composer.focus', category: 'composer', keys: ['mod+l'] },
  // Fixed, context-local shortcuts, listed so users can find them. This row
  // uses the same ⌘/Ctrl+L chord as `composer.focus` above. It is the
  // selection half of the chord: the selected text (terminal text, preview
  // lines) goes into the composer as context.
  { id: 'view.selectionToComposer', category: 'view', keys: ['mod+l'] },
  // Terminal clipboard. ⌘C/⌘V on macOS, Ctrl+Shift+C/V elsewhere — matching VS
  // Code. Plain Ctrl+C also copies when text is selected (Windows Terminal /
  // Tabby behavior); with no selection it stays SIGINT, so it isn't listed.
  { id: 'view.terminalCopy', category: 'view', keys: IS_MAC ? ['mod+c'] : ['mod+shift+c'] },
  { id: 'view.terminalPaste', category: 'view', keys: IS_MAC ? ['mod+v'] : ['mod+shift+v'] },
  // Global OS chord registered in main while HUD mode is up.
  { id: 'hud.snapToPointer', category: 'view', keys: ['mod+shift+g'] }
]
