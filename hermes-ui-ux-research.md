# Hermes Agent — UI/UX Research Reference
### Desktop Electron app (`apps/desktop/`), Web Dashboard (`web/`), Ink TUI (`ui-tui/`)

**Sources.** Primary source is a local checkout of the public repo at
`~/AppData/Local/hermes/hermes-agent` @ `f42f579` (Wed 2026-09-30). Doc claims are
cross-checked against <https://hermes-agent.nousresearch.com/docs/> (`/user-guide/desktop`,
`/user-guide/tui`, `/user-guide/features/web-dashboard`, `/user-guide/bot-mode`,
`/user-guide/features/kanban`, `/user-guide/features/bot-screen`).

Anything not directly read out of source or docs is tagged **[INFERRED]**.

---

## 0. Stack at a glance (sourced: `apps/desktop/package.json`)

| Concern | Choice |
|---|---|
| Shell | Electron (`apps/desktop/`), custom frameless titlebar (`app/shell/titlebar.ts`) |
| Renderer | React **19.2.7**, React Compiler enabled |
| State | **nanostores 1.4.2** (`@nanostores/react`), heavily derived-atom memoized |
| Server state | **@tanstack/react-query 5** — one client (`lib/query-client.ts`) |
| Routing | `react-router 8` |
| Styling | **Tailwind v4** (`@tailwindcss/vite`) + one 3,784-line hand-authored `src/styles.css` token layer; `--dt-*` aliases shadcn-style primitives |
| Primitives | `radix-ui`, `cmdk 1.1.1`, `class-variance-authority`, `lucide-react`, `@tabler/icons-react`, `@vscode/codicons` |
| Chat primitives | **`@assistant-ui/react` 0.14.24 + `@assistant-ui/react-streamdown`** |
| Markdown | `streamdown`, `remend`, `shiki` via `react-shiki`, `katex`, `remark-math`, `mermaid` |
| Editor | CodeMirror 6 (`code-editor.tsx`, `json-document-editor.tsx`) |
| Diff | custom Shiki-transformer diff (`components/chat/diff-lines.tsx`, `syntax-diff.tsx`) |
| Terminal | `node-pty` + `@xterm/xterm` 6 (WebGL addon, serialize, unicode11) |
| Drag/drop | `@dnd-kit/*` + `react-arborist` (file tree) + `dnd-core` |
| Virtualization | `@tanstack/react-virtual` |
| Guided tour | `driver.js` |
| Graph (Starmap) | `d3-force` |
| Shared client | `@hermes/shared` (`file:../shared`) — see §7 |

**The single most important architectural fact:** `apps/desktop/src/app/index.tsx` is six lines:

```ts
export { ContribController as default } from './contrib'
```

> "The app root is the contribution-driven shell: panes, titlebar/statusbar items,
> keybinds, palette commands, routes, and themes all register through the
> contribution registry (`src/contrib`) — core surfaces use the same calls plugins
> do."

**Every core surface is a contribution to the same registry a plugin writes to.** That
is the reusable seam for an overlay product: don't fork the app, register areas.

---

## 1. Desktop app shell, layout & IA

### 1.1 App shell

Layout is a **binary tree of split nodes** (`components/pane-shell/tree/model.ts`):

```ts
type LayoutNode = SplitNode | GroupNode
type Orientation  = 'row' | 'column'
type TabStripMode = 'always' | 'never'
type DropPosition = 'center' | 'left' | 'right' | 'top' | 'bottom'
interface GroupNode { type:'group'; panes: string[]; active: string; tabStrip?: TabStripMode; minimized?: boolean }
```

Panels inside a `GroupNode` render as **tabs**; siblings in a `SplitNode` render as
adjacent columns/rows with a draggable sash. Model API is `split/`, `group/`,
`insertAtGroup`, `movePane(s)`, `mergeZonesWithPane`, `removePane`,
`setGroupMinimized`, `setSplitWeights`, `mirrorTreeHorizontal`, `normalize`,
`migratePersistedTree`.

### 1.2 Layout presets — the concrete default geometry

`app/contrib/layout-presets.ts` (sourced verbatim). Five preset trees ship:

**`DEFAULT_TREE` (order 0, tier `advanced`)** — "sessions left, chat main, right
sidebars in column order main | … | review | file-browser (files outermost)":

```
split('row', [
  group(['sessions']),                    // 1
  group(['workspace']),                   // 3.4   ← the chat
  split('column', [                        // 1.25
    split('row', [
      group(['review']), group(['files'])  // weights [1, 1.2]
    ], 'spl-rail'),
    group(['terminal'])                    // weights [1.6, 1]
  ], 'spl-right')
], [1, 3.4, 1.25], 'spl-root')
```

**`BASIC_TREE` (tier `simple`, and the Advanced "Basic" preset)** — same weights but
files/review become a right column *resting closed*, terminal a collapsed rail:

```
BASIC_TREE  = split('row', [ group(['sessions']),
                             split('column',[group(['workspace']), group(['terminal'])],[3,1]),
                             split('row',[group(['review']), group(['files'])],[1,1.2]) ],
                          [1, 3.4, 1.25])
BASIC_RESTING = ['terminal', 'files', 'review']   // closed but PRESENT in the tree
```

**`FOCUS_TREE`** — files/review as tabs *behind* the chat, terminal a rail:
`[sessions(1) | split('column',[group(['workspace','files','review']), group(['terminal'])],[3,1])] [1, 4.6]`

**`TERMINAL_DECK`** — `[3,1]` vertical; top row `[sessions(1) | workspace(3.2) | (files,review)(1.2)]`, terminal full-width bottom.

**`QUAD`** — `[3,1]` vertical; `[sessions+files(1) | workspace(3)]` over `[terminal(1.4) | review(1)]`.

**Simple-mode shelf**: `sidebar-left` (order 0) and `sidebar-right` (order 1, `mirrorTreeHorizontal(BASIC_TREE)`).

The comment on `BASIC_TREE` is a design lesson worth stealing:

> "A tree that simply omitted them was a lie — applying it adopts every missing pane
> back in as workspace tabs, which is Focus."

**Presets are themselves contributions** (`area: 'layouts'`, `data: LayoutNode`).
Bundled register as `source: 'core'`, user-saved presets round-trip through
`localStorage['hermes.desktop.layoutPresets.v2']` and re-register as `source: 'user'`.
Each preset carries `resting: string[]` (panes the tree places but leaves CLOSED) and
`tier: 'simple' | 'advanced'`.

### 1.3 Pane geometry & window-control awareness

`components/pane-shell/tree/grid-model.ts` is a **faithful port of Microsoft
PowerToys FancyZones** (`GridLayoutModel.cs` + `GridData.cs`, MIT) — names and
algorithms track the C# source. Coordinates live in a **0..10000 space**
(`MULTIPLIER = 10000`), rows/columns hold percent tracks summing to 10000, and a
`cellChildMap[row][col] = zoneIndex` assigns cells to zones (a zone spanning cells
repeats its index). `MIN_ZONE_SIZE = 500`. Resizers are derived from
`cellChildMap` discontinuities; dragging one moves **every zone touching that edge**.
Merging computes the rectangular **closure** of the selection.

`components/pane-shell/geometry.ts` handles the native-window-controls collision:

```ts
const CONTROLS_BAND_HEIGHT = 34
const MACOS_LIGHTS_WIDTH   = 58
const MACOS_FALLBACK_BUTTON_X = 24
```

`windowControlsRect()` returns an AABB in viewport pixels — `{x: vw-overlayWidth, …}`
for Windows/WSLg's top-right native overlay, `{x:0, width: x+58, height:34}` for macOS
traffic lights, `null` in fullscreen or a plain browser. Any layout region intersecting
it reserves the space and exposes a drag strip. The workspace zone also publishes its
edges as CSS vars `--workspace-left` / `--workspace-right` so plain CSS can align to
the main pane.

**Per-pane state** is a flat persisted map (`store/panes.ts`):

```ts
interface PaneStateSnapshot { open: boolean; widthOverride?: number; heightOverride?: number }
```

Height exists because of the bottom-row terminal. All selectors are memoized per pane id
so `useStore` subscriptions stay referentially stable.

### 1.4 Responsive collapse

`app/layout-constants.ts` (sourced verbatim):

```ts
export const PAGE_INSET_X        = 'px-[clamp(1.25rem,4vw,4rem)]'
export const PAGE_INSET_NEG_X    = '-mx-[clamp(1.25rem,4vw,4rem)]'
export const PAGE_MAX_W          = 'max-w-[75rem]'
export const SIDEBAR_DOCK_MIN_WIDTH_PX = 640
export const SIDEBAR_COLLAPSE_MEDIA_QUERY = `(max-width: ${640 - 0.02}px)`
```

> "A rail costs 237px (`SIDEBAR_DEFAULT_WIDTH`) and the chat beside it wants roughly
> what a popped-out session window enforces on itself (420px), so docking stops paying
> for itself around here — while still leaving an overlay band down to the window's own
> 400px minimum. Expressed as a **dock floor** rather than a collapse ceiling so
> half-screen splits stay docked on common laptop widths (1280 → 640)."

Two reusable geometry constants (rail = 237px, pop-out min = 420px) and a clamp-based
gutter are directly liftable. **[INFERRED]** the two literals named in the comment
(`237`, `420`) are not themselves exported constants in that file — they live wherever
the sidebar width and pop-out window min-width are declared.

### 1.5 Sidebar (left)

Sidebar is a **pane in the tree** (id `sessions`), not a fixed column — so it can be
mirrored, dragged, or hidden by any preset. Contents (`app/chat/sidebar/`):

- **Nav rows** — typed as `SidebarNavId = 'artifacts' | 'capabilities' | 'command-center' | 'cron' | 'messaging' | 'new-session' | 'settings'`, each with optional `route`, optional `action:'new-session'`, optional `keybindActionId` for a tooltip hint.
- **Gateway / connection groups** — `gateway-groups.tsx`, `gateway-group-model.ts`, `fleet-gateway-menu-group.tsx`; each group = one gateway + profile, expanded to its session rows.
- **Profile rail** — `profile-rail*.tsx` (status dots, fleet roster, connect).
- **Projects tree** — `projects/`, `workspace-groups`, `project-filter.tsx`, `project-dialog.tsx`, worktree lanes.
- **Sessions sections** — `sessions-section.tsx`, `virtual-session-list.tsx` (`@tanstack/react-virtual`), `session-row.tsx` + `session-row-gesture.tsx` + `session-row-details.tsx` + `session-row-actions.tsx` + `session-row-slots.tsx`, `load-more-row.tsx`.
- **Cron jobs section** (`cron-jobs-section.tsx`), filter menu, archive, split submenu, storage-corrupt notice.

**Navbar prefs are a contribution area too** (`store/sidebar-nav.ts`,
`area: 'sidebarNav.prefs'`). The rationale is a good pattern to copy verbatim:

> "A contribution is attributed, merged with a stated rule, and dropped by the
> loader's per-plugin disposer on disable/reload — the rows come back on their own.
> The USER's choices persist in the plugin's own `ctx.storage`; the plugin
> re-contributes them on register."

Merge rule is explicit and pure: `hidden = UNION(hide) − NEVER_HIDDEN`, then first
contribution's `order` wins for the ids it names; unnamed rows keep default relative
order; unknown ids are inert. `NEVER_HIDDEN = ['capabilities']` because that row hosts
the Plugins tab — "the user's only path to a plugin's own off-switch."

### 1.6 Right sidebar — terminal, files, review

Two physical edges, three logical panes. `app/contrib/layout-sides.ts` derives the
side mapping **from the tree**, not from a stored flag:

```ts
const sessionsOnRight = () => order.indexOf('sessions') > order.indexOf('workspace')
```

`⌘\` (`view.flipPanes`) mirrors the tree; the toggle buttons follow automatically.
Terminal/file toggles are bound to whichever edge currently holds them
(`bindTreeSideVisibility`).

- **Terminal** (`app/right-sidebar/terminal/`) — real PTY via `node-pty` + xterm WebGL. Multiple terminals stack in a vertical tab rail (`ctrl+shift+↑/↓`, `ctrl+shift+w` close). "Shells persist while hidden" — every terminal stays mounted with scrollback and running processes until explicitly closed. **Add to chat** sends a selection into the composer.
- **Files** — `RightSidebarPane`, a file browser feeding the preview rail; `react-arborist` tree.
- **Review** (`app/right-sidebar/review/`) — git working-tree surface: branch + ahead/behind, changed files (list or tree), diffs scoped to **Uncommitted / Branch / Last turn**, stage/unstage, revert, commit message (or **Generate commit message**), **Commit**, **Commit & Push**, **Create PR** via `gh`, or **Ask Hermes to open PR**. Also branch create/switch.
- **Preview rail** (`app/chat/right-rail/`) — renders web pages, files, tool outputs side by side. Sub-surfaces: `preview-browser-bar.tsx`, `preview-console*.ts`, `preview-reader.ts`, `preview-file.ts`, `preview-artifact.tsx`, `preview-nav.ts`, `preview-drive.ts`, plus `preview-annotate-card.tsx` (comment mode).

**Preview tiles are dynamic panes.** `DockPaneBeside` lands each new preview as its own
pane next to the file tree *wherever the tree currently lives* — so a file double-click
never stacks a preview into the files sidebar's tab group.

### 1.7 Floating HUD

`⌘/Ctrl+Shift+H` detaches the chat into a chrome-free always-on-top bar over
whatever you're working in (`app/hud/`, `app/floating-hud.ts`). The **position is
context**: the bar's location tells the agent which app/screen you mean, so "this" and
"that page" resolve. Move = press-and-hold the composer then drag (macOS/Windows);
Ctrl+drag on X11; compositor drag handle on native Wayland. `⌘/Ctrl+Shift+G` snaps it to
the pointer. Hyprland/COSMIC need compositor cooperation (floated+pinned over IPC on
Hyprland; `desktop.ozone_platform_hint: x11` on COSMIC). Exiting returns the caret to
the same session in the main window.

### 1.8 Tabs & windows

Session tabs are panes in the tree (`session-tile:*`), so "tabs" and "panes" are the
same primitive. `⌘T` new tab, `⌘W` close focused zone's active tab, `⌘⇧T` reopen.
`⌘⇧N` new window; any session pops out to a single-chat window **without the global
sidebar**, and live output streams into every window showing that session.

### 1.9 Keyboard system (sourced: `lib/keybinds/actions.ts`)

`KeybindActionMeta = { id, category, defaults, label?, passthrough?, editableTargetPolicy? }`.
Categories: `composer | profiles | session | navigation | view`. **61 declared
actions.** Highlights:

| id | default | note |
|---|---|---|
| `nav.commandPalette` | `mod+k`, `mod+p` | |
| `nav.commandCenter` | `mod+.` | |
| `nav.settings` | `mod+,` | |
| `session.new` | `mod+n` | `shift+n` deliberately dropped (#76185 — hijacked normal typing) |
| `session.newTab` | `mod+t` | |
| `session.newWindow` | `mod+shift+n` | |
| `session.next` / `.prev` | `ctrl+tab`+`ctrl+pagedown` / `ctrl+shift+tab`+`ctrl+pageup` | **literal Control**, macOS reserves ⌘Tab |
| `session.slot.1…9` | `ctrl+1…9` | |
| `session.focusSearch` | `mod+shift+f` | |
| `workspace.newWorktree` | `mod+shift+b` | |
| `workspace.openFolder` | `mod+o` | editor-standard open-folder |
| `view.toggleSidebar` / `.toggleRightSidebar` | `mod+b` / `mod+j` | |
| `view.toggleStatusbar` | `mod+shift+s` | |
| `view.toggleTabStrip` | `mod+alt+t` | ⌥⌘T because ⌘⇧T is reopen-tab everywhere; **ships bound** where VS Code leaves it unbound, "here the hide can take away every other affordance" |
| `view.toggleReview` | `mod+g` | ⌘G |
| `view.showBrowser` | `mod+shift+l` | "L for location"; plain ⌘L is terminal-select |
| `view.toggleHud` | `mod+shift+h` | |
| `view.showTerminal` / `.newTerminal` | `` ctrl+` `` / `` ctrl+shift+` `` | literal `` ctrl `` everywhere (⌘` is macOS window cycling) |
| `view.nextTerminal`/`.prevTerminal`/`.closeTerminal` | `ctrl+shift+↓`/`↑`/`w` | |
| `view.flipPanes` | `mod+\` | "the backslash reads like a mirror line" |
| `view.closeTab` / `.reopenTab` | `mod+w` / `mod+shift+t` | |
| `view.findInPage` | `mod+f` | fires inside textareas (browser parity) |
| `view.findNext` / `.findPrevious` | *(unbound)* | ⌘G/⌘⇧G claimed by the find bar while open; unbound so the panel doesn't report a permanent conflict |
| `appearance.toggleMode` | `shift+x` | |
| `keybinds.openPanel` | `mod+/` | |
| `composer.focus` | `/`, `enter` | read-only soft focus; `/`/bare keys type-to-focus |
| `composer.modelPicker` | `mod+shift+m` | "m for model — the convention chat apps converged on (LibreChat, Open WebUI, Cherry Studio)" |
| `composer.voice` | `ctrl+b` (macOS) / `mod+alt+v` | |
| `composer.dictate`, `.reasoningUp/Down` | *(unbound)* | "too personal to claim by default" |
| `profile.default` | `mod+d` | ⌘` macOS-reserved, ⌘0 is reset-zoom |
| `profile.switch.1…18` | `mod+1…9`, then `mod+alt+1…9` | `PROFILE_SLOT_COUNT = 18` |
| `view.tabSlot.1…9` | `mod+1…9` | **`passthrough: true`** — shares ⌘1…9 with profiles; claims it only when a tab strip is eligible |

`KEYBIND_READONLY` lists non-rebindable chords for completeness: composer
`enter / shift+enter / mod+enter / mod+shift+k / @ / / / ? / up / down / escape /
mod+l`, plus `view.selectionToComposer` (also `mod+l`), terminal
copy/paste (`mod+c`/`mod+v` on macOS, `mod+shift+c/v` elsewhere), `hud.snapToPointer`.

Three mechanisms worth stealing wholesale:

1. **`passthrough`** — a handler returning `false` hands the chord to the next action
   bound to it. Chord sharing becomes *layering*, not conflict.
2. **`editableTargetPolicy: 'modified'`** — only fires in a focused editable if the combo
   carries a modifier beyond Shift. "a bare or shift-only rebind stays with the input so
   it can never hijack typing" (#71627).
3. **Diff-only persistence** (`store/keybinds.ts`): only combos that *differ* from the
   shipped default are written to `localStorage['hermes.desktop.keybinds']`, so changing
   a default in a future release is never shadowed by an old snapshot. Late-registered
   contributed actions carry their overrides forward by re-reading storage, not the
   module-init snapshot. Conflict detection (`conflictsFor`) ignores the pair when the
   earlier action is `passthrough`.

### 1.10 ⌘K command palette

`app/command-palette/index.tsx` (1,683 lines), built on `cmdk`
(`Command/CommandInput/CommandList/CommandGroup/CommandItem`). Supporting files:
`contrib.ts` (`PALETTE_AREA = 'palette'`), `marketplace-theme-page.tsx`,
`pet-palette-page.tsx`, `status-row.tsx`, `highlight-watcher.tsx`.

The palette is **two-mode** (`$commandPaletteOpen`, `$commandPalettePage`,
`$commandPaletteSeed`) — a command surface *and* a **Command Center** page
(`⌘.` → `/command-center`) with grouped sections. Group headings visible in source:
`branches`, `projects`, `goTo`, `commands`, `commandCenter`, `appearance`, `settings`,
`settingsFields`, `apiKeys`, `mcpServers`, `archivedChats`, `pinned`, `sessions`,
`plugins`, `themeTitle`, `colorMode`, plus `capLabel` (capabilities).

Search is **hand-rolled ranking on top of cmdk** — worth copying:

```ts
scoreItem(item, needle): number
  label === needle                      → 1
  label.startsWith(needle)              → 0.9
  words.includes(needle)                → 0.85
  words.some(w => w.startsWith(needle)) → 0.8
  label.includes(needle)                → 0.7
  every term in label                   → 0.6
  matched only via keywords             → 0.4
  any term missing from label AND keys   → 0 (dropped)
```

AND semantics on every typed word; **items sorted by score within a group, groups
sorted by their best item**, stable sort so curated ordering breaks ties. The comment
explains why: cmdk auto-selects the first DOM item on every search change, so
"rendering best-match-first is what puts the highlight on the best match." Headings are
written into `data-value` but excluded from matching so groups keep source order.
Selections are namespaced by id because a settings field and a session can share a title.

A **contributed** palette area lets a plugin add commands; a palette command calls
`host.navigate(path)` to reach a plugin page. `⌘K` also has *pages* (theme marketplace,
pet palette).

---

## 2. Design system

### 2.1 Token architecture (sourced: `apps/desktop/src/styles.css`, 3,784 lines)

Three layers, deliberately:

1. **Shadcn-compatible alias layer** (`--color-background`, `--color-foreground`,
   `--color-card`, `--color-popover`, `--color-primary`, `--color-muted`,
   `--color-destructive`, `--color-ring`, `--color-sidebar*`, …) each pointing at a
   `--dt-*` "desktop theme" variable.
2. **`--dt-*` layer** (`--dt-background`, `--dt-foreground`, `--dt-primary-solid`,
   `--dt-input-inset`, `--dt-composer-ring`, `--dt-scrollbar-thumb`,
   `--dt-user-bubble`, `--dt-sidebar-bg`, …) — the app's own semantics.
3. **`--ui-*` layer — the one plugin authors should use.** Tokens are named by
   *semantic role and ordinal*, not by component:

```
--ui-text-primary       = mix(--ui-base 94%)
--ui-text-secondary     = mix(--ui-base 74%)
--ui-text-tertiary      = mix(--ui-base 54%)
--ui-text-quaternary    = mix(--ui-base 36%)

--ui-stroke-primary / -secondary / -tertiary / -quaternary

--ui-bg-chrome / -sidebar / -editor / -elevated / -card / -input / -primary
--ui-bg-secondary / -tertiary / -quaternary / -quinary
--ui-row-hover-background / --ui-row-active-background / --ui-row-open-background
--ui-control-hover-background / --ui-control-active-background

--ui-chat-surface-background  --ui-editor-surface-background
--ui-sidebar-surface-background  --ui-terminal-surface-background
--ui-widget-surface-background  --ui-surface-background

--ui-accent  --ui-accent-secondary  --ui-warm  --ui-base
--ui-chat-bubble-background  --ui-chat-bubble-opaque-background
--ui-inline-code-background / -foreground  --ui-selection-background
```

There is **no spacing or radius scale of its own** — density rides Tailwind, radius
rides a scalar:

```css
--radius-xs:  calc(var(--radius-scalar) * 0.125rem)
--radius-sm:  calc(var(--radius-scalar) * 0.5rem)
--radius-md:  calc(var(--radius-scalar) * 0.625rem)
--radius-lg:  calc(var(--radius-scalar) * 0.75rem)
--radius-xl:  calc(var(--radius-scalar) * 1rem)
--radius-2xl: calc(var(--radius-scalar) * 1.5rem)
--radius-3xl: calc(var(--radius-scalar) * 2rem)
--radius-4xl: calc(var(--radius-scalar) * 2.5rem)
--radius: 0.75rem   --radius-scalar: 0.6  --dt-spacing-mul: 1
```

### 2.2 The seed-and-mix theme algorithm

A theme supplies a handful of **seeds**, and `color-mix()` derives everything. Light
mode seeds:

```css
--theme-foreground:        #17171a
--theme-primary:           #0053fd
--theme-midground:         #0053fd
--theme-warm:              #cf806d
--theme-background-seed:   #f8faff
--theme-sidebar-seed:      #f3f7ff
--theme-card-seed:         #ffffff
--theme-elevated-seed:     #ffffff
--theme-bubble-seed:       color-mix(in srgb, #0053fd 6%, #ffffff)
--theme-neutral-chrome:    #f3f3f3
--theme-neutral-card:      #fcfcfc
```

…and eleven **per-surface mix percentages** that decide how much accent tints each
surface: `--theme-mix-chrome: 92%`, `-sidebar: 100%`, `-card: 22%`, `-elevated: 28%`,
`-bubble: 0%`, plus seven fill percentages (`--theme-fill-primary-accent-mix: 16%` down
to `--theme-fill-quinary-accent-mix: 3%`), four stroke percentages (`-stroke-primary
24%` … `-stroke-quaternary 6%`), `--theme-row-hover-accent-mix: 4%`,
`-row-active 8%`, `-control-hover 6%`, `-control-active 8%`.

**Dark mode flips the mix percentages, not the algorithm** (`:root.dark`, line 587):

```css
--theme-mix-chrome: 74%;  --theme-mix-card: 38%;  --theme-mix-elevated: 46%;  --theme-mix-bubble: 46%;
--theme-neutral-chrome: #0d0d0e;  --theme-neutral-sidebar: #0a0a0b;  --theme-neutral-card: #161618;
--ui-red: #e75e78; --ui-green: #55a583; --ui-cyan: #6f9ba6;
--dt-input-inset: inset 0 1px 1px color-mix(in srgb, #000 38%, transparent);
--ui-inline-code-background: color-mix(in srgb, #ffffff 7%, transparent);
--ui-selection-background:    color-mix(in srgb, #ffd24a 38%, transparent);
--ui-widget-surface-background: color-mix(in srgb, var(--ui-bg-editor) 88%, #000);
--composer-ring-strength: 1.3;  --dt-input-border: 4%;  --ui-tab-hover-darken: 6%;
```

Note the deliberate asymmetrys: dark needs a *stronger* black inset to show a recess,
a *lighter* resting input border (4% vs 7%), a *larger* selection tint share (38% of
`#ffd24a` vs 55%) because a dark canvas swallows alpha, and a *lower* widget-surface
value so an inline widget doesn't read as the brightest thing in the transcript.
**Mode toggle is class-based** — `@custom-variant dark (&:is(.dark *))` with
`:root.dark` overrides — and there is a `prefers-color-scheme` path too.

**Shipped presets** (`themes/presets.ts`): `github`, `nous`, `nous-alt`, `catppuccin`,
`everforest`, `solarized`, `midnight`, `ember`, `mono`, `cyberpunk`, `slate`. The
canonical `nous` theme is a **fork of the GitHub VS Code theme** (GitHub Light Default /
Dark Default) with a Nous-blue accent, "converted through the same path a Marketplace
install takes, so the palette here is byte-identical to importing the extension
yourself." `github` ships alongside precisely so nous's accent can move without
silently redefining what "GitHub" means.

### 2.3 Accent & status colors

Eight named accents as `--ui-*`, defined once in `:root` and overridden in `:root.dark`:

```css
--ui-red: #cf2d56;  --ui-orange: #db704b;  --ui-yellow: #c08532;  --ui-green: #1f8a65;
--ui-cyan:  #4c7f8c; --ui-blue:   #0053fd;  --ui-purple: #9e94d5;
```
(dark: red `#e75e78`, green `#55a583`, cyan `#6f9ba6`)

**The context-usage popover gets one color per token category** — a genuinely good
idea for any agent UI that must explain *why* a window is filling:

```css
--context-usage-system:        mix(--ui-base 55%, transparent)
--context-usage-tools:         var(--ui-purple)
--context-usage-rules:         var(--ui-green)
--context-usage-skills:        var(--ui-yellow)
--context-usage-mcp:           mix(--ui-red 72%, var(--ui-purple))
--context-usage-subagents:     mix(--ui-blue 70%, var(--ui-cyan))
--context-usage-memory:        mix(--ui-orange 80%, var(--ui-yellow))
--context-usage-conversation:  var(--ui-cyan)
```

Plus a five-stop "legendary memory" gradient family and a dedicated `StatusDot`,
`status-pulse.tsx`, `action-status.tsx`, `profile-dot-state.ts`, `session-dot-state.ts`,
`wake-indicator/`.

### 2.4 Diff tokens

```css
--ui-diff-add-border:      var(--ui-green)
--ui-diff-add-background:  mix(--ui-green 12%, transparent)
--ui-diff-add-foreground:  mix(--ui-green 70%, #000)      /* dark: mix(--ui-green 62%, #fff) */
--ui-diff-remove-border / -background / -foreground       /* same shape, --ui-red */
```

Rendering (`components/chat/diff-lines.tsx`): a **unified diff** parsed once, two paths
sharing it — `SyntaxDiff` (Shiki-highlight the change content in the file's language,
then a per-line transformer paints the add/remove tint on top) and `DiffLines` (color
only: no language, over budget, or while Shiki loads). Git file-headers and `@@` hunk
noise are **dropped**, as is the `+/-` gutter: "so changes read by color + a 2px gutter
accent, **the way Cursor does**." `chunkLines` + `useFixedRowWindow` virtualizes.

### 2.5 Code blocks & syntax highlighting

`components/chat/shiki-config.ts` (sourced verbatim):

```ts
export const SHIKI_THEME = { dark: 'github-dark-dimmed', light: 'github-light-default' }
export const SHIKI_COLOR_REPLACEMENTS = { 'github-light-default': { '#6e7781': '#57606a' } }
export const SHIKI_SHIGHLIGHT_SCOPE = `hermes-shiki-v1:${JSON.stringify({...})}`
```

- `github-dark-dimmed` over `github-dark-default`: "the vivid tokens read harsh at our small code size." Shared by the inline diff renderer "so code + diffs match."
- The color replacement is a **contrast audit with a stated reason**: `#6e7781` is ~4.2:1 on the code card — "borderline unreadable at our 11px code size, and worst of all for shell snippets where a single `#` turns the rest of the line into one long comment span." Bumped to `#57606a` (~6.4:1). Dark's `#8b949e` (~6.1:1) is left alone.
- **Content-addressed highlight cache** keyed by an explicit `SCOPE` string that must be bumped whenever rendering options change, "because keys are NOT allowed to silently produce a different DOM than the one they were computed with."
- Also `shiki-plain.tsx` (no highlight), `shiki-block.tsx` (lazy chunk), `code-card.tsx`, `code-editor.tsx` (CodeMirror), `json-document-editor.tsx`, `exceedsHighlightBudget` (a budget guard that falls back to plain), and `terminal-output.tsx`.
- Markdown fidelity is a first-class concern: "Code display and Copy preserve leading blank lines, trailing spaces, and terminal blank lines from the Markdown parser"; media/preview extraction preserves text outside removed attachment spans including first-line code indentation.

### 2.6 Typography

```css
--dt-base-size: 1rem;  --dt-line-height: 1.5;  --dt-letter-spacing: 0;
--dt-font-sans: 'Segoe WPC','Segoe UI',-apple-system,BlinkMacSystemFont,'SF Pro Text',system-ui,sans-serif, <emoji stack>
--dt-font-kbd:  -apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',system-ui,sans-serif
--dt-font-mono: 'JetBrains Mono','Cascadia Code','Cascadia Mono','DejaVu Sans Mono','Liberation Mono',
                'Noto Sans Mono','Noto Mono','SF Mono',ui-monospace,Menlo,Consolas,monospace, <emoji stack>
```

Three decisions worth naming: **Kbd always uses the native UI face** ("never theme
typography overrides"); JetBrains Mono is first because it is *bundled* via `@font-face`
and is the terminal's primary, so "code/diff match the terminal on every platform
instead of drifting to a system Cascadia Code where it's installed"; and Linux mono
fallbacks are kept for Latin Extended glyphs (#61392, Vietnamese diacritics).
`themes/chat-font.ts` and per-theme `typography.fontUrl` allow remote display faces
(e.g. GitHub's Courier Prime).

### 2.7 Shadows, borders, window effects

```css
--shadow-xs: 0 0.0625rem 0.125rem mix(#000 5%, transparent)
--shadow-composer: (same as xs)
--shadow-nous / --shadow-sm / --shadow-md / --shadow-lg
--stroke-nous: color-mix(in srgb, currentColor 3%, transparent)
--ui-sash-hover-border:    mix(--ui-accent 18%, --ui-stroke-tertiary)
--ui-sash-hover-background: mix(--ui-accent 6%, transparent)
```

Sashes get a **hover tint rather than a persistent border**. Dark mode raises the alpha
because "a dark card swallows shadow."

**Three window treatments** on `<html>`: default opaque; `data-hermes-glass` (macOS
vibrancy / Windows 11 acrylic-mica) and `data-hermes-clear`. The glass algorithm is
documented at length and is the cleanest transparency design I've read in an app
stylesheet:

> "**ONE PAINTER:** `<body>` paints the glass tint exactly once … and the field tokens
> (chat / sidebar / editor surface) go fully transparent. The field surfaces **NEST** —
> body > pane container > chat section > transcript wrapper all wear these tokens — so
> thinning the tokens themselves stacks the tint once per layer and a session pane ends
> up near-opaque (~0.93 at 60%) while the landing page, with fewer layers, reads far
> clearer. With a single painter the field alpha is the same number on every route."

Scope selectors `[data-glass-raised]`, `[data-glass-opaque]`, `data-translucency-peek-scope`
keep raised content legible over material. `store/translucency.ts` +
`translucency.win10.test.ts` handle the platform differences; a boot script in
`index.html` pins an opaque background before first paint and must be relaxed or it
sits behind the translucent body.

### 2.8 Z-index ladder

```css
--z-modal-backdrop: 120;  --z-modal: 130;  --z-modal-popover: 140;
--z-over-modal: 200;      --z-over-modal-content: 210;
--z-switcher-backdrop: 219; --z-switcher: 220;
--z-connecting: 1200; --z-onboarding: 1300; --z-onboarding-popover: 1310;
--z-setup: 1400;  --z-crash: 1500;
```

### 2.9 The component kit (`components/ui/` — 100+ files)

`Button`, `Input`, `Textarea`, `Select*`, `Switch`, `Checkbox`, `Slider`,
`SegmentedControl`, `SplitButton`, `Tabs`, `TextTab`, `PaneTab`, `TabDropdown`,
`Dialog`(+portal context), `ConfirmDialog`, `SetupFormDialog`, `Sheet`,
`DropdownMenu`, `ContextMenu`, `ActionsMenu`, `FanMenu`, `Menu`, `Popover`,
`Tooltip`/`Tip` + `tooltip-placement.ts`, `Badge`, `Kbd`/`KbdGroup`/`KbdCombo`,
`SearchField`, `ScrollArea`, `FadeScroll`, `Separator`, `Skeleton`, `Loader`,
`GlyphSpinner`(+`.css`), `EmptyState`, `ErrorState`, `CardStack`, `Masonry`, `Reel`,
`Alert`, `AvatarChip`, `ConnectorCard`, `Field`, `Progress`, `Pagination`,
`RowButton`, `CopyButton`, `FileTypeIcon`, `Favicon`, `Codicon`, `ToolIcon`,
`ProfileGlyph`, `DecodeText`, `HighlightMatches`, `DiffCount`, `DisclosureCaret`,
`StatusDot`, `StatusPulse`, `LogView`, `ColorSwatches`, `Zoomable`, `use-zoom-pan`,
`SandboxedFrame`, `SanitizedInput`, `Masonry`, `keyboard-first.ts`, `DropAffordance`,
`context` for portals.

**`Codicon`** (`@vscode/codicons`) is the icon system — VS Code's own glyph font. Any
plugin's sidebar row picks one by name (`{ codicon: 'project' }`).

The plugin SDK explicitly re-exports this kit and says:

> "Prefer these over hand-rolled elements so the plugin looks native; **style with theme
> vars, never hardcoded colors.**"

and the pitfall is stated without ambiguity:

> "NEVER hardcode colors or backgrounds (`#000`, `black`, `rgb(...)`). Panes already sit
> on the app's editor background — leave the background alone … For canvas drawing,
> resolve them once with
> `getComputedStyle(canvas).getPropertyValue('--ui-accent')`."

---

## 3. Chat UX

### 3.1 Thread architecture

`components/assistant-ui/` wraps `@assistant-ui/react`. The thread layer
(`components/assistant-ui/thread/`) is ~70 files:

- `thread/index.tsx`, `list.tsx`, `timeline.tsx` + `timeline-rail.tsx` + `timeline-data.ts` + `timeline-index.ts` + `use-timeline-history/reveal.ts` — the transcript window, its rail, virtualization, history.
- `transcript-window.tsx`, `windowing`/`retention` — bounded transcript.
- `assistant-message.tsx`, `user-message.tsx` + `user-message-text.tsx` + `user-message-edit*.tsx` (edit-and-resend with undo), `system-message.tsx`, `response-group.tsx`, `agent-delivery.tsx`.
- `status.tsx`, `streaming.test.tsx`, `status-tail-only.test.tsx`, `sealed-tool-parts.test.tsx`.
- `inter-agent-collapse.tsx`, `duplicate-activity-indicator.test.tsx`, `turn-gap-indicator`, `block-direction.test.tsx`, `use-messages-below.ts`, `use-sticky-prompt-clip.ts`.

**Timeline rail**: "long chats get a slim rail of markers along the edge of the
transcript, one per prompt. Hover it to pop open the list of prompts, click one to jump
straight to that point." Appears once the chat has a handful of turns.

**Reading-position memory**: returning to a session restores its saved distance from
the bottom instead of jumping to the latest message; sessions left at the bottom keep
following. Stored **in the Desktop installation's local storage, not synced through the
backend** — stated explicitly.

**Find in page**: `⌘F` opens `components/find-bar.tsx` over the *rendered* transcript;
Enter/Shift+Enter (or `⌘G`/`⌘⇧G` while open) step matches; Esc closes. The find bar's
capture-phase listener claims `⌘G`/`⌘⇧G` and stops propagation so stepping works
without stealing them from the review pane toggle.

### 3.2 Tool-call rendering — a *ticker*, not a wall

This is the most distinctive chat pattern in the app.

`components/assistant-ui/tool/run-ticker.tsx`:

> "A one-line window over a growing list of rows. Each new row slides the one before it
> up and out, so activity that would otherwise grow down the page reads as a single
> line ticking over in place. Rows are clipped to a uniform line box so the reel's offset
> stays exact whatever a row happens to contain."

```html
<div class="tool-ticker"><div class="tool-ticker__reel" style="--tool-ticker-index: N">
  <div class="tool-ticker__row">…</div>
</div></div>
```

Shared by the tool run and by each subagent's relayed stream in a delegation card —
"the same thing seen from two sides."

**Run summary** (`run-summary.ts`) turns calls into prose with a **fixed clause order**
so the same run always reads the same way:

```ts
type RunCategory = 'analyze'|'browse'|'delegate'|'edit'|'explore'|'interact'|'other'|'read'|'run'|'search'
const CATEGORY_ORDER = ['edit','explore','search','read','browse','interact','analyze','run','delegate','other']
const CATEGORY_COPY = { edit:{noun:['file','files'],past:'Edited',present:'Editing'}, … }
```

Tools are routed by name so categories stay honest: `browser_navigate→browse`,
`delegate_task→delegate`, `execute_code`/`terminal→run`, `list_files`/`read_file`/
`search_files→explore`, `web_search`/`session_search_recall→search`, `web_extract→read`,
`vision_analyze→analyze`, `list_files`→`explore`. Two comments capture the intent
precisely:

> "Routed by name so a web search never counts as an explored file (#123085). Browser
> tools other than navigation are interaction, not page loads: a screenshot or a click
> fetches nothing, so they must not be counted as pages."

Other tool components: `approval.tsx` + `approval-activity.ts` (interactive approve/deny
cards), `delegate.tsx` + `delegate-model.ts` (subagent cards with steer/stop),
`skill-activity.ts`, `connector-tool.tsx`, `mcp-setup-tool.tsx`, `catalog-install-tool.tsx`,
`fallback.tsx` + `fallback-model.ts` (generic renderer for unknown tools), and
`hide-code-diffs.test.tsx`.

**Output collapsing** — `components/chat/expandable-block.tsx`:

```ts
el.scrollHeight > 121                       // overflow detection
overflowing && !expanded → 'max-h-[7.5rem]'
expanded                     → 'max-h-[40dvh]'
```

plus a bottom fade (`pointer-events-none`, "a pure overflow cue" that spans the full
edge so making it clickable "killed both sideways scrolling and text selection") with a
compact toggle pinned clear of the scrollbar track. Measurement runs inside
`ResizeObserver` timing only — "a synchronous mount-time `scrollHeight` read forces a
reflow per instance, and a tool-heavy transcript mounts dozens of these on a session
switch."

### 3.3 Reasoning / thinking

`store/reasoning-disclosure.ts`:

```ts
$reasoningCollapsedByDefaultPref  // localStorage 'hermes.desktop.reasoning.collapsedByDefault', default false
$reasoningCollapsedByDefault      // modeBound — Simple mode rests it collapsed WITHOUT touching the pref
$showReasoning                    // mirrors display.show_reasoning; quoted "false" still means off
```

"Desktop-local presentation preference; **shared backend config must not be changed by a
single window**."

`message-parts.tsx` logic worth noting: `null` = no explicit user toggle yet, so **live
reasoning stays visible** by default; the preview follows new tokens "until the user
scrolls up to read earlier reasoning"; a reasoning group with **no actual text is dropped
entirely** — "encrypted/spinner-coerced reasoning … an empty header is never wanted."
There's a self-gating "Thinking…" header that reads its own reasoning parts.

### 3.4 Attachments, messages, composer

- **Drag-and-drop anywhere** in the chat area attaches to the next message (`chat-drop-overlay.tsx`, `drop-affordance.ts`, `large-paste.ts` collapses long pastes).
- **Directive chip actions**: hover an actionable reference (a URL) to reveal its action pill. "A short grace period lets you move from the chip to the pill before it dismisses… after leaving, unrelated pointer movement does not delay dismissal. Clicking its action **preserves the draft selection**."
- **Message actions**: copy (response-scoped), edit-and-resend (`user-message-edit*.tsx`), reactions (`message-reactions.tsx`, `double-click-reaction`, `reaction-picker-focus-follow`), double-click to react.
- **Composer** (`app/chat/composer/`, ~90 files): `rich-editor.tsx` (contenteditable with undo history `undo-history.ts` + `undo-cut-drag`), `model-pill.tsx`, `reasoning-pill.tsx`, `voice-*` (voice-activity, voice-menu, voice-fan, start-voice-button, voice-engine-rows), `attachments.tsx`, `completion-drawer.tsx`, `suggestion-pills.tsx`, `queue-panel.tsx`, `help-hint.tsx`, `inline-refs.tsx` / `slash-refs.tsx` / `path-refs.tsx` / `url-refs.tsx` / `at-folder-navigation`, `trigger-popover*.tsx`, `micro-actions.tsx`, `focus-chord.ts` (the ⌘L priority ladder), `status-stack/`.
- **Queue editing**: ⌘Enter queues while busy; pressing **Stop (or Esc)** while turns are queued pauses the queue and expands it above the composer — "resume it from there, or send, edit, and delete individual entries."
- **Composer history**: ↑/↓ in an empty composer recalls previous prompts.
- **Task progress above the composer**: expand the Tasks header to inspect each phase; "long lists stay bounded above the input; scroll inside the expanded list to reach the final tasks without moving the conversation."
- **Subagents frame** above the composer while delegated workers are live: count, task names, elapsed time, latest activity; previews up to three, expand for the full roster, select for detail + **Steer**/**Stop**. Steering "acknowledges that guidance is queued for a checkpoint, not that the child has already read it."
- **Status stack** (`components/chat/status-section.tsx`) — the shared collapsible chrome for queue / subagents / background. One card with dividers; `StatusSection` owns only its own collapse, supports `collapsedIndicator` and a compact `preview` that stays visible while the roster is collapsed.
- **Model pill** lives in the composer, left of the microphone, and is the one control that can give width back continuously (`shrink` + truncating label, **no `max-w` cap**): "the pill sizes to its label, so a long model name only truncates when the row is genuinely out of room (#49340) — not at an arbitrary 160px."

### 3.5 Sessions, Projects, Plugins as session filters

Sessions are scoped by `connectionId` + `profile` + project. `store/session.ts` holds
`$sessions`, `$cronSessions`, `$messagingSessions`, `$connection`, `$currentCwd`,
`$currentModelSource`. Scope atoms: `ALL_PROJECTS` / `$projectScope`,
`ALL_PROFILES` / `$profileScope`.

**Projects** = discovered git repos + explicit folders + repos inferred from
intentional sessions. Discovery scans the home directory to bounded depth, configurable
per profile:

```yaml
desktop:
  repo_scan_enabled: true
  repo_scan_roots: []
  repo_scan_exclude_paths: []
```

"Set `repo_scan_enabled: false` … Existing disk-discovery cache rows for that profile
are cleared; explicit projects and repositories inferred from intentional Hermes
sessions remain available." Changing any value "invalidates only that profile's
disk-discovery cache." **"Hide from sidebar" is a separate per-item curation action** —
good separation of *discovery* from *curation*. `⌘O` opens a folder as a project (upsert:
enter the owning project, else create). `⌘⇧B` / **New worktree** creates a worktree on a
new branch; worktrees are their own lanes under the project, and removing one offers to
delete the directory (branch stays) or just hide the lane — with a force option when it
has uncommitted changes.

Sidebar selection **follows the focused chat pane**: "Opening or focusing a session tab
clears a page's highlight, including contributed pages such as Kanban, even when the
workspace retains that page's route."

### 3.6 Status bar

`app/shell/hooks/use-statusbar-items.tsx` (780 lines) owns the bar; plugins add via
`statusBar.left` / `statusBar.right`. Core items, in order: `version-client`,
`version-backend`, `command-center`, `gateway-switcher`, `profile-switcher`,
`gateway-health`, `free-tier`, `workspace-cwd` (with `copy-workspace-path`,
`reveal-workspace-finder`, `reveal-workspace-sidebar` submenus), `agents`, `cron`,
`webhooks`, `running-timer`, `context-usage`, `cache-hit-rate`, `tokens-per-second`,
`session-timer`, `terminal`.

Right-click (**Show in status bar**) toggles items and can hide the bar entirely
(`⌘⇧S`). `cache-hit-rate` and `tokens-per-second` ship **off by default** and are
explained in-product: cache hit rate is "the share of this session's prompt tokens served
from the provider's prompt cache … so you can watch a session get cheaper as it warms up";
tokens/second is output throughput averaged over the last 10 model calls. Both update
live during a turn. The **Context Usage** popover opens a token breakdown by category
(system prompt, tool definitions, skills, memory, rules, MCP, subagent definitions,
conversation). Per-session **YOLO toggle** lives here too.

---

## 4. Distinctive surfaces

### 4.1 The plugin system — the thing to copy

Two on-disk doors, one contract, hot reload, no build step:

- `$HERMES_HOME/desktop-plugins/<id>/plugin.js` — standalone, **enabled by default**.
- `$HERMES_HOME/plugins/<id>/desktop/plugin.js` — the desktop half of a unified
  agent-plugin package (alongside its `plugin.yaml` and `dashboard/plugin_api.py`). The
  Electron shell copies it into `desktop-plugins/<id>/` with a `.hermes-package.json`
  marker — "the ONLY root the renderer loads from — so the pane is app-level and does
  not come and go with the selected profile." **Opt-in**: it inventories in
  Settings → Plugins but stays off until toggled.

Only three import specifiers resolve: `@hermes/plugin-sdk`, `react`,
`react/jsx-runtime`. **JSX syntax will not parse** (uncompiled) — use
`jsx('div', {children: …})`. Load within seconds of the file landing; every later save
hot-reloads in place; ⌘K → **Reload desktop plugins** is the fallback. Failures show a
toast naming the error.

**Registry areas** (the complete seam list):

| Area constant | id | What it grants |
|---|---|---|
| `PANES_AREA` | `panes` | A layout pane with `data:{ placement, dock?, width?, height? }` |
| `STATUSBAR_AREAS` | `statusBar.left` / `.right` | Chip or arbitrary node |
| `TITLEBAR_AREAS` | `titleBar.left` / `.center` / `.right` | Permanent titlebar mounts |
| `WORKSPACE_PAGE_HEADER_AREA` | `workspace.pageHeader` | Mount-scoped page header controls |
| `PALETTE_AREA` | `palette` | ⌘K commands |
| `KEYBINDS_AREA` | `keybinds` | Rebindable actions |
| `ROUTES_AREA` | `routes` | A full page at `data.path` |
| `SIDEBAR_NAV_AREA` | `sidebar.nav` | A sidebar row `{ codicon, label, path, tier? }` |
| `SIDEBAR_PROFILE_GROUP_HEADER_AREA` | `sidebar.profileGroup.header` | A render above each gateway/profile group |
| `SIDEBAR_NAV_PREFS_AREA` | `sidebarNav.prefs` | Hide/reorder nav rows |
| `SESSION_ROW_AREAS` | `.leading` / `.trailing` | Decorate sidebar session rows |
| `TRANSCRIPT_DIRECTIVE_AREA` | (transcript directives) | Render your component inline in a chat message |
| `LAYOUTS_AREA` | `layouts` | A layout preset tree |
| `APPEARANCE_AREAS.extra` | `appearance.extra` | Append controls to the Appearance page |
| `THEMES_AREA` | themes | A full `DesktopTheme` |
| mount-scoped `Contribute` | any | Titlebar chrome that leaves when the page unmounts |

**Pane placement** is two-layered:

```js
data: { placement: 'left'|'right'|'bottom'|'main' }   // semantic role — stacks (tabs) with same-role panes
data: { placement: …, dock: { pane: 'workspace', pos: 'bottom' } }  // land on a specific EDGE
```

`pos` ∈ `top|bottom|left|right|center`; `pane` is any pane id (`workspace`, `sessions`,
`terminal`, `files`, `review`, `logs`). The canonical example, with its own sizing note:

> "e.g. 'below the conversation' = `dock: { pane: 'workspace', pos: 'bottom' }` — declare
> a `height` (e.g. `'200px'`) so it doesn't take half the zone."

A **full page** registers `ROUTES_AREA` with `data:{path:'/my-page'}` + a render; it
mounts in the workspace pane like any built-in view. Make it reachable with a
`SIDEBAR_NAV_AREA` row "and/or a `PALETTE_AREA` command calling `host.navigate(path)`."
`'advanced'` on the nav row's `tier` keeps it out of Simple mode.

**Transcript directives** are the cleverest extension. Register
`TRANSCRIPT_DIRECTIVE_AREA` with `data:{ name:'task', render:({attrs, streaming}) => jsx(…) }`
and the assistant renders your component inline by emitting a bare line
`::task{id="BB-12"}`. Attrs are untrusted `key="value"` strings — **validate them**.
Unclaimed/malformed directives fall back to plain text. Core's own `::preview{file="…"}`
is the reference. Crucially: "After registering one, **TELL the model it exists** (a
bundled skill or the user's instructions) — it won't discover the name on its own."
And `ctx.rest`/`ctx.socket` are scoped to `/api/plugins/<id>` by construction with
traversal rejected.

**Curated OS door** — `ctx.os` is attributed to the plugin and throttled per plugin:
`notify({title, body?, silent?, icon?, activate?, onActivate?, actions?})` posts a native
notification and **fires only while the user is away from Hermes** (use `host.notify`
for the in-app toast). Gated by Settings ▸ Notifications ▸ "Plugin notifications."
`openExternal`, `revealPath`, `writeClipboard` "resolve `false` (never throw)."

**Data + i18n**: the app's **one** React Query client is exported (`queryClient`,
`useQuery`, `useMutation`) — "never hand-roll a poll loop." `ctx.i18n.register({en, ja,…})`
ships a plugin's own locale bundle (never edit core `en.ts`); resolution is app locale →
plugin `en` → raw key; `usePluginI18n(id)` re-renders on locale switch.
`ctx.i18n.registerAppLocale('pl', {...})` adds a whole-app **language pack** (keyed off
`locales/_keys.desktop.json`) without changing `display.language`.

**Reactive state for plugins**: `host.state.*` readonly atoms —
`activeSessionId`, `busy`, `awaitingResponse`, `busyBySession`, `cwd`, `gateway`,
`model`, `profile`, `viewport`, plus tile-aware `focusedSessionId` (runtime id),
`focusedStoredSessionId` (durable id), `focusedSessionProfile`, and `focusedUsage`
(live streamed `UsageStats`, no RPC needed). Read `.get()` in handlers,
`useValue(atom)` in components. `busy` = focused chat working after a send;
`awaitingResponse` = until the first assistant payload. The doc is explicit that
`focusedSessionProfile` beats `profile`: "`profile` is the gateway socket's home, which
does not move with tab focus."

Canonical plugin pitfalls, verbatim and worth adopting as a review checklist:

> - "Handlers must read state imperatively (`$atom.get()`), never from render closures — rapid events will otherwise see stale values."
> - "Keep components small; subscribe (`useValue`) only in the leaf that renders the value."
> - "Reference only what you imported — a component you forgot to import (e.g. `StatusDot`) is a ReferenceError at render."
> - "Canvas panes MUST track their container with a `ResizeObserver` and re-size the canvas (`width`/`height` attributes, not just CSS)."
> - "JSX syntax will not parse — the file loads uncompiled."
> - "`ctx.socket` is a **no-op on OAuth remotes**, so always keep a polling fallback."

**Contribution identity is namespaced** as `${pluginId}:${id}`, each contribution gets
**its own error boundary** ("a broken contribution degrades to an inline error instead
of a dead page"), and per-plugin disposers drop registrations on disable/reload.
`defaultEnabled: false` on the default export ships an opt-in plugin. `Contribute`
(render-scoped, mount-scoped) vs `ctx.register` (permanent) is a clean distinction.

### 4.2 Bot Mode

Sourced: `/user-guide/bot-mode` docs + `app/agents/`. **On by default in the desktop
app, no install.** A **Bots** tab next to Sessions in the left sidebar, with a
**Routines** tile docked beside the conversation while the tab is active.

"A Bot **is** a Hermes profile — isolated config, memory, skills, credentials, and chat
history under `~/.hermes/profiles/<name>/`. Bot Mode is a UI over that primitive, so
everything you do in it is visible from the CLI too."

Roster row = avatar + latest-message preview + timestamp. **New Agent** quick path is
three fields (Name, Title, Description) and the Bot introduces itself as the first
message of its own Bot Chat. An **Advanced** disclosure opens Custom SOUL.md,
per-skill/per-toolset/per-MCP enablement, and "Copy API keys from the main profile"
(on by default — but single-use OAuth logins are deliberately *not* copied).
**Edit Profile** via right-click reopens the same surface on the live profile.
**Create on** picks which machine the Bot lives on. Routines are plain cron jobs
namespaced `[bot:<name>] <routine>`, so they appear in `hermes cron list` and the core
Cron page. Plugins watch members work via the durable room log + an
`on_room_member_activity` hook.

The profile-group header contribution area (`SIDEBAR_PROFILE_GROUP_HEADER_AREA`) exists
for exactly this: "First consumer: the Bots plugin's Screen portal."

### 4.3 Bot Screen (remote desktop)

Each profile gets its own Xfce screen on a headless Linux gateway host, streamed live
into Hermes Desktop over RFB — via `display.observe` and a sibling WebSocket to
`/api/display/ws`. "Watch what the bot does, **take over** when it hits a login, 2FA
prompt, CAPTCHA or payment step, then **hand control back**." The screen lives on the
gateway host, so the bot keeps working after you close the app.

```yaml
bot_desktop:
  geometry: "1440x900"        # viewer scales to fit the pane
  auto_start: false           # start on first computer_use call
  min_free_memory_bmb: 1536   # actually min_free_memory_mb
  idle_stop_minutes: 30
  placement: auto             # auto | terminal | gateway
```

Memory: gateway ~300 MB, Xvnc+Xfce ~+220 MB, headed Chromium during takeover +0.5–1 GB
(§ one page ~550 MB). Plan **~1.1–1.5 GB per open screen with a browser.** Sandbox
mode (`terminal.backend: docker|ssh|singularity`) puts the screen *inside* the sandbox
so computer_use never acts outside the boundary.

The threat model is stated honestly and is worth reading before shipping anything
similar: "Screens are work surfaces, not security boundaries." Chromium's DevTools port
on loopback "is reachable by **any local user** on the [host]." The control lease is a
**tool-level** fence on `computer_use` and the browser tools, not an OS one.

### 4.4 Kanban — a desktop plugin, shipped in-repo

`plugins/kanban/dashboard/` — and note what the manifest looks like:

```json
{
  "name": "kanban", "label": "Kanban", "version": "1.0.0", "icon": "Package",
  "tab": { "path": "/kanban", "position": "after:skills" },
  "entry": "dist/index.js", "css": "dist/style.css", "api": "plugin_api.py"
}
```

That is the **web dashboard** plugin manifest shape (path + position). The desktop half
would live beside it at `plugins/kanban/desktop/plugin.js` (per the desktop-plugins doc).
Docs also state Bot Mode/Kanban appear in the desktop, and the repo contains
`apps/desktop/src/contrib/kanban-i18n.test.tsx`, which asserts registrations at
`ROUTES_AREA` (`kanban:page`) and `SIDEBAR_NAV_AREA` (`kanban:nav`) — i.e. **a first-party
kanban surface ships as a contribution, ids namespaced `kanban:*`.** **[INFERRED]** the
full kanban desktop page itself lives in the bundled desktop-plugins folder rather than
`apps/desktop/src/`, since no `kanban/` dir exists under `apps/desktop/src/`; the test
file under `src/contrib/` verifies the registration contract.

Data model: every task is a row in `~/.hermes/kanban.db`; every handoff is a row anyone
can read and write; every worker is a full OS process. Two front doors on the same
`kanban_db` layer: the `kanban_*` toolset (13 tools — `kanban_show/list/complete/block/
heartbeat/comment/attach/attach_url/attachments/create/link/unblock`) for agents, and
`hermes kanban` / `/kanban` / the dashboard for humans and cron. Column vocabulary in the
bundle: `todo`, `review`, `done`, `blocked`.

Its plugin CSS is a **case study in theme-immune plugin styling** — copy this comment
and the approach:

> "Themes (shipped AND user-installable) routinely paint every `<code>` and `<pre>` on
> the page with an opaque accent-color fill … Rather than play whack-a-mole with theme
> rules, reset EVERY `<code>`/`<pre>` inside the plugin container to transparent with
> `!important`, then opt back in ONLY on the class that carries intentional styling."

…and "All colors reference theme CSS vars so the board reskins with the active dashboard
theme. **No hardcoded palette.**"

### 4.5 Cron / scheduled jobs

`app/cron/`: `index.tsx`, `job-state.ts`, `cron-actions.ts`, `cron-job-model.ts`,
`deliver-checkboxes.tsx`, `blueprints.tsx`. A route (`/cron`) that renders as an
**OverlayView** — a full-screen modal card over the shell. Async cron completions arrive
**in the transcript as collapsed timeline disclosures**: "Open the completion label to
read the result body (including job output) as Markdown; long reports scroll within the
disclosure. **Task instructions and delivery envelopes are not shown as report content.**"
A sidebar **Cron jobs section** lists them next to sessions.

### 4.6 Memory — Memory Graph ("Starmap")

`app/starmap/` (15 files): `star-map.tsx`, `simulation.ts` (d3-force), `geometry.ts`,
`render.ts`, `timeline.tsx` + `time-axis.ts` + `playback-hotkey.ts`, `color.ts`,
`types.ts`, `text.ts`, `share-code.ts` + `share-controls.tsx`, `node-context-menu.tsx`.

An interactive, zoomable node graph of skills and memories with a **timeline**,
filterable **All / Used / Learned**. The share control exports the map layout as a
compact pasteable code and imports it back — and the boundary is explicit:

> "exports the map layout as a compact code you can paste to someone else (**layout
> only — none of your memory or skill text is included**)"

Settings section **Memory & Context** (`app/settings/constants.ts`) owns:
`memory.memory_enabled`, `memory.user_profile_enabled`, `memory.memory_char_limit`,
`memory.user_char_limit`, `memory.provider`, `context.engine`, plus the compression family
(`compression.enabled/threshold/target_ratio/protect_last_n/codex_gpt55_autoraise`,
`auxiliary.compression.timeout`). UI: `app/settings/memory/connect.tsx`,
`provider-config-panel.tsx`, `provider-config-modal.tsx`, `field-control.tsx`.

### 4.7 Session search

`session.focusSearch` = `⌘⇧F`. Backed by FTS5 in `~/.hermes/state.db`
(`hermes_state_fts.py`), surfaced in the sidebar as `session-index.ts` +
`virtual-session-list.tsx` + `filter-menu.tsx` + `strip-fts-markers.test.ts`, and in ⌘K
under the `sessions` heading. A session-search *tool* also exists
(`session_search_recall`) and is categorized as `search` in the run summary.

### 4.8 Other distinctive surfaces

- **Quick Entry** — a composer summoned by a global OS hotkey (default `Ctrl/Cmd+Shift+Space`, configurable, "needs at least one modifier"; if another app owns the chord the settings row says so).
- **Artifacts** (`/artifacts`) — a searchable gallery of images/files/links from sessions, each showing its originating session with jump-back, images/files opening in a preview with download / open-in-browser / copy.
- **Command Center** (`⌘.` → `/command-center`) — branches, projects, go-to, commands, settings fields, API keys, MCP servers, archived chats, pinned, skills/plugins.
- **Settings**: 9 curated sections (`model, chat, appearance, workspace, safety, browser, memory, voice, advanced`), each an explicit **allow-list of config keys** — "Curated desktop config surface: only fields a user might tune from the app." Voice alone lists 34 keys. Light/Dark/System mode options.
- **Preview browser comment mode** — click **Annotate**, click any element or drag a box on the live page, type a note. Each saved comment is a numbered pin. "Saving a pin **never sends a turn**" — **Add N comments** attaches a cropped screenshot per pin plus a short prompt, and *you* hit send. Each comment carries its CSS selector, markup, and the computed styles that matter for layout, so the agent can find the element in source. "Password and hidden field values, and any attribute that looks like a key or token, are **redacted on the page before the markup leaves it**." Larger batches are grouped by which part of the page each comment sits in — "so twenty-odd comments become a handful of pieces of work … because the groups are separate DOM subtrees they usually touch separate files, which is what makes handing them to parallel workers safe." Pin numbers hold steady if you delete one; switching chats clears the stack.
- **Toasts/tours**: `tours.ts` + `driver.js` guided tours, `gui_tour` desktop tour, `translucency` / `backdrop` / `particles` cosmetic layers, pet mascots (`petdex`).

---

## 5. Ink TUI (`ui-tui/`)

### 5.1 Architecture

React + Ink, TypeScript owns the screen, Python owns everything else. `src/entry.tsx`
exits early if stdin isn't a TTY, starts `GatewayClient`, renders `App`.

```
python -m tui_gateway.entry          ← spawned by GatewayClient
  stdin/stdout: newline-delimited JSON-RPC
  stderr: captured into an in-memory log ring
```

Malformed stdout → `gateway.protocol_error`; stderr → `gateway.stderr`; "**Neither writes
directly into the terminal.**" The JSON-RPC contracts in `tui_gateway/contracts` are
**code-generated** into `apps/shared/src/gateway-contract.generated.ts` (5,904 lines) by
`scripts/gen_gateway_contracts.py`; `tests/tui_gateway/contracts/test_generated.py` fails
when stale. The comment is worth noting:

> "Any method the desktop may route to a named profile (`requestGatewayForProfile` adds
> `profile`)."

Custom Ink fork at `ui-tui/packages/hermes-ink` (`@hermes/ink`, esbuild bundle,
`sideEffects: true`) with subpaths `@hermes/ink/text-input` and a `bootstrap` dir.

### 5.2 Components (`ui-tui/src/components/`)

`appLayout.tsx` (the composition root), `appChrome.tsx`, `appOverlays.tsx`,
`overlay.tsx` + `overlayPrimitives.tsx` + `overlayScrollbar.tsx` + `overlayControls.tsx`,
`branding.tsx` (Banner / Panel / SessionPanel), `markdown.tsx`, `streamingMarkdown.tsx`,
`messageLine.tsx`, `streamingAssistant.tsx`, `thinking.tsx`, `todoPanel.tsx`,
`agentsOverlay.tsx` + `agentsPanel.tsx`, `activeSessionSwitcher.tsx`, `modelPicker.tsx`,
`queuedMessages.tsx`, `textInput.tsx`, `themed.tsx`, `goalBar.tsx`, `journey.tsx`,
`widgetGrid.tsx`, `petSprite.tsx`/`petPicker.tsx`, `pluginsHub.tsx`, `skillsHub.tsx`,
`prompts.tsx`, `helpHint.tsx`, `loaders.tsx`, `fpsOverlay.tsx`, `maskedPrompt.tsx`,
`billingOverlay.tsx`, `subscriptionOverlay.tsx`, `connectionSetupOverlay.tsx`,
`gridStreamsDemo.tsx`, `accordion.tsx`.

Layout geometry constants live next to the layout (`appLayout.tsx`):

```ts
const PET_BOTTOM = 3; const PET_PAD_LEFT = 2; const PET_RIGHT = 1; const PET_GUTTER_GAP = 1;
const KITTY_PLACEHOLDER = '\u{10eeee}';
const MIN_GUTTER_BODY_COLS = 72;
```

The mascot "reserves no layout rows (the transcript scrolls underneath); instead it
**publishes its footprint** so the transcript can keep its text clear of it — a right
gutter on wide terminals, reserved bottom rows on narrow ones." For kitty graphics the
width counts real placeholder cells because "zero-width diacritics make string length lie."

Streaming blocks flatten into one ordered list so "each block's leading gap can be
derived from the block directly above it … Tracking the predecessor rather than the live
text is what keeps the streaming block from jumping when it flushes into a settled
segment." Block grouping: `domain/blockLayout.ts` →
`type BlockGroup = 'diff'|'event'|'intro'|'model'|'note'|'slash'|'trail'|'user'` +
`hasLeadGap()` + `blockRenders()`.

### 5.3 Design system — seeds → palette

`ui-tui/src/theme.ts` (971 lines). The palette is **built, not enumerated**:

> "skins/base themes supply identity seeds; derivative roles are computed from the
> theme's own base colors and can never be incoherent."

`DARK_SEEDS`: `accent #FFBF00`, `bg #101014`, `surface #1a1a2e`, `text #FFF8DC`,
`primary #FFD700`, `prompt #FFF8DC`, `border #CD7F32`, `error #ef5350`, `ok #4caf50`,
`warn #ffa726`, `activeRow #333355`, `selection #3a3a55`, `shellDollar #4dabf7`,
`statusGood #8FBC8F`, `statusWarn #FFD700`, `statusBad #FF8C00`, `statusCritical #FF6B6B`.

`LIGHT_SEEDS`: `bg #ffffff`, `text #3D2F13`, `prompt #2B2014`, `accent #956E00`,
`primary #867000`, `ok #367E39`, `error #C14240`, `border #A56628`, `shellDollar #377BB3`.

**Three-layer background adaptation** (`adaptColorsToBackground`, `contrastRatio`,
`ensureContrast`, `liftForContrast`):
1. **WCAG contrast floor** for foreground roles (the dark seeds are documented as derived so hosts *without* a contrast pass still render).
2. **Fill polarity** — background-role colors (completion menu, status bar) flip against the base palette.
3. **SEMANTIC 2.2 floors** for alert colors (ok/error/warn/status).

The "beloved classic look is the authored palette rendered essentially RAW" — a
documented decision to let authored skins win.

Default prompt glyph `'❯'`, overridable per skin (`branding.prompt_symbol`), with
`cleanPromptSymbol` and a Termux guard because "Termux fonts/terminal backends can
render decorative prompt glyphs with [wrong widths]". `/skin` live-previews while you
browse.

**TUI theme auto-detection is three-layer** (docs):
1. `HERMES_TUI_THEME` = `light` | `dark` | a raw 6-char background hex.
2. `COLORFGBG` (the xterm-derived hint).
3. **OSC 11 terminal background probe** — works on Ghostty, Warp, iTerm2, WezTerm, Kitty.

### 5.4 Information architecture & affordances

**Startup banner** — four collapsible sections with `▸`/`▾` chevrons, each section's state
local to the banner instance:

| Section | Default |
|---|---|
| Tools | **Open** |
| Skills | Collapsed |
| System Prompt | Collapsed |
| MCP Servers | Collapsed |

"The Tools list opens by default because it's the most-checked section at session start;
Skills, System Prompt, and MCP Servers collapse by default so the banner stays compact
even when you've installed dozens of skills."

**Instant first frame** (banner paints before the app finishes loading) and
**non-blocking input** (type and queue before the session is ready; the first prompt
sends the moment the agent comes online).

**Details mode — the section-visibility system.** This is the TUI's most transferable
idea. Four sections (`thinking`, `tools`, `subagents`, `activity`) × four modes
(`hidden | collapsed | expanded`), with **opinionated per-section defaults**:

```yaml
display:
  skin: default
  personality: helpful
  details_mode: collapsed
  sections: {}          # per-section overrides, any subset
  thinking: expanded    # always open
  tools: expanded       # always open
  activity: collapsed   # opt back IN to the activity panel (hidden by default)
  mouse_tracking: all   # off | wheel | buttons | all
```

Shipped defaults: `thinking` **expanded** (reasoning streams inline as the model emits
it), `tools` **expanded** (calls and results render open), `subagents` follows the global
`details_mode`, `activity` **hidden** (ambient meta is "noise for most day-to-day use";
tool failures still render inline on the failing tool row, ambient errors surface via a
floating-alert backstop when every panel is hidden). Runtime: `/details [hidden|collapsed|
expanded|cycle]` globally, `/details <section> <mode>` per section. The stated goal:
"stream the turn as a live transcript instead of a wall of chevrons." Precedence is
explicit: per-section override > section default > global `details_mode`, and "anything
set explicitly in `display.sections` wins over the defaults, so existing configs keep
working unchanged."

**Mouse tracking presets** are unusually well thought out: `wheel` (1000+1006, scroll +
click, no hover) is "recommended inside tmux to silence the prompt-row 'No image in
clipboard' spam from hover events"; `buttons` adds 1002 for terminal-side drag selection;
`all` adds 1003 for hover (scrollbar paginate-on-hover, link mouseenter). Styles ship
with **matched glyph widths "so the rest of the status bar doesn't jitter on rotation."**
Selection highlight is a uniform background rather than SGR inverse.

**Status line** — real-time agent state:

| Status | Meaning |
|---|---|
| `starting agent…` | id live, tools still coming online; messages queue |
| `ready` | idle, accepting input |
| `thinking…` / `running…` | reasoning or running a tool |
| `interrupted` | turn cancelled; Enter to send again |
| `forging session…` / `resuming…` | connect / `--resume` handshake |

Plus: **working directory with git branch** (`~/projects/hermes-agent (docs/two-week-gap-sweep)`,
mtime-cached so a side-terminal `git checkout` is reflected); **per-prompt elapsed time**
(`⏱ 12s/3m 45s` live → `ⲷ 32s / 3m 45s` frozen, first = since last user message, second
= session total, reset each prompt); **`🗜️ N`** auto-compressions; **`▶ N`** background
tasks; **`⚠ YOLO`** warning — "the same badge also appears in the startup banner so you
cannot launch an auto-approving session without noticing." Once a session is named, its
title appears as an **accent-colored badge at the far-right edge**, taking the workspace
label's place and truncating on narrow terminals.

**Busy indicator is pluggable** (`display.tui_status_indicator: kaomoji|emoji|unicode|ascii`,
`/indicator`): the default rotates the kawaii-face palette every 2.5s.

**Overlays, not inline flows**: `/help` (categorized, arrow-navigable), `/sessions`
(live switcher — `Ctrl+X`; list/switch/close/new; `↑↓` + mouse; `Enter` switch, `Ctrl+D`
close, `Ctrl+N` new, `Ctrl+R` refresh, `Esc` close; click the `N live sessions` count in
the status line; select `+new`, type a prompt, `Enter` to dispatch, `Tab` first for a
model), `/model` (modal, grouped by provider, cost hints), `/skin` (live preview),
`/usage` (token/cost/context panel), `/agents` `/tasks` (observability: live subagent tree
with kill/pause, per-branch cost/token/file rollups, turn-by-turn history), `/details`.

Slash autocompletion opens as a **floating panel with descriptions**, not an inline
dropdown. `Ctrl+G` / `Ctrl+X Ctrl+E` opens the input buffer in `$EDITOR` and sends the
result as the prompt. Markdown pipeline renders LaTeX inline (`$E=mc^2$`, `$$\frac{a}{b}$$`)
as Unicode-formatted math, "unsupported syntax falls back to showing the literal TeX
wrapped in a code span so it remains copyable." **Alternate-screen rendering** for
differential updates — no flicker, no scrollback clutter.

Requirements: Node ≥ 20 (verified by `hermes doctor`), TTY (else falls back to
single-query mode). First launch installs `ui-tui/node_modules`; the bundle rebuilds
when sources are newer than `dist`. `HERMES_TUI_DIR` points at a prebuilt bundle
(containing `dist/entry.js`) for Nix/system packages.

---

## 6. Web Dashboard (`web/` + `hermes_cli/web_routers/`)

### 6.1 How it differs from the desktop app

`hermes dashboard` → local server on **`http://127.0.0.1:9119`** (flags `--port`,
`--host`, `--no-open`, `--isolated`). `--insecure` is **deprecated/no-op**: "a public bind
always requires an auth provider (password or OAuth)." "The dashboard runs entirely on
your machine — no data leaves localhost." Hosted mode uses Nous Portal OAuth.

It is a **machine-level admin surface**: one server manages every profile. A profile
switcher in the sidebar (visible only when >1 profile exists) decides which profile the
management pages read and write, and **Config, API Keys, Skills, MCP, Models, and the
Chat tab all follow it.** Two details worth stealing:

- While a non-owning profile is selected, an **amber banner names the managed profile**
  "so the write target is never ambiguous."
- The selection lives in the URL (`?profile=`), "so deep links like
  `http://127.0.0.1:9119/skills?profile=worker` land with the switcher preselected and
  survive refresh."
- `worker dashboard` from a profile alias **routes to the machine dashboard** rather than
  starting a second server; `--isolated` opts into a dedicated per-profile server.

Routing structure — `web/src/App.tsx` with `react-router`, lazy-loaded route pages
(`RouteFallback` = "Loading…", `UnknownRouteFallback` for unknown paths):

`/chat`, `/sessions`, `/files`, `/analytics`, `/models`, `/logs`, `/cron`, `/skills`,
`/plugins`, `/mcp`, `/channels`, `/webhooks`, `/pairing`, `/profiles`, `/config`,
`/env`, `/system`, `/docs`.

**Plugin tabs are first-class in the dashboard** and more mature than the desktop's
equivalent. `buildRoutes(builtinRoutes, manifests)` classifies manifests by
`tab.override` (replaces a built-in route), `tab.hidden` (registered, not nav-linked),
and plain add-ons; skips `/plugins` (reserved) and any path already built in; nav items
split into `coreItems` vs `pluginItems` with `tab.position` (e.g. `"after:skills"`) for
ordering. The manifest shape (`plugins/kanban/dashboard/manifest.json`):

```json
{ "name","label","description","icon","version",
  "tab": { "path": "/kanban", "position": "after:skills" },
  "entry": "dist/index.js", "css": "dist/style.css", "api": "plugin_api.py" }
```

So a Python `dashboard/plugin_api.py` backend is declared in the manifest and reached
over the same `/api/plugins/<id>` namespace as the desktop's `ctx.rest`.

### 6.2 The Chat tab — it really does embed the TUI

`web/src/pages/ChatPage.tsx`'s header comment is the cleanest architecture diagram in
the repo:

```
<div host> (dashboard chrome)
  └─ <div wrapper> (rounded, dark bg, padded — the "terminal window" look)
      └─ @xterm/xterm Terminal (WebGL renderer, Unicode 11 widths)
           │ onData    keystrokes → WebSocket → PTY master
           │ onResize  terminal resize → `\x1b[RESIZE:cols;rows]`
           │ write(data) PTY output bytes → VT100 parser
           ▼
  WebSocket /api/pty?token=<session>
      ▼
  FastAPI pty_ws  (hermes_cli/web_server.py)
      ▼
  POSIX PTY → `node ui-tui/dist/entry.js` → tui_gateway + AIAgent
```

Add-ons: `FitAddon`, `Unicode11Addon`, `WebLinksAddon`, `WebglAddon`. Imports the
shared `@nous-research/ui` primitives (`Button`, `Typography`) — **the dashboard and
desktop share a UI package**, which is the strongest signal in the repo that these are
one design language with two hosts.

The PTY plumbing is unusually complete: `pty-reconnect.ts` exports
`PTY_CONNECTING_TIMEOUT_MS`, `PTY_KEEPALIVE_INTERVAL_MS`, `PTY_RECONNECT_INPUT_MESSAGE`,
`PTY_RECONNECT_MAX_ATTEMPTS`, `PTY_RESUME_RECONNECT_THROTTLE_MS`,
`PTY_RESUME_SANITIZE_WINDOW_MS`, `PTY_TICKET_TIMEOUT_MS`, plus pure predicates
`ptyReconnectDelayMs`, `shouldBlockPtyInput`, `shouldReconnectPtyOnPageResume`.
`createPtyCompositionForwarder` and `PtyResumeSanitizer` handle the two classic xterm.js
failures:

> "xterm occasionally drops committed dead-key/IME text … [the forwarder] interferes with
> xterm.js's own IME handling on its hidden textarea"

and

> "xterm.js relies on native compositionstart/compositionend on its … [textarea]"

Also: `ChatSidebar`, `ChatSessionList`, `ChatWorkspacePicker`, `latchChatActivation`,
`shouldRestoreTerminalFocus`, `loseWebglContexts`, `buildTerminalTheme(bg, fg)`, and a
`/chat` route that is a **sink** when a persistent `ChatPage` host is mounted outside
`<Routes>` (so the terminal survives chat ⇄ page navigation — `data-chat-active` +
`aria-hidden` toggle rather than an unmount).

**The PTY child is profile-scoped**: "a scoped chat spawns its PTY child with the selected
profile's `HERMES_HOME`, so the conversation runs with that profile's model, skills,
memory, and session history. Switching profiles starts a fresh terminal session."

### 6.3 The `HERMES_TUI_GATEWAY_URL` contract (important integration point)

Docs are unusually precise here, and this is the contract SamAgent must respect:

> "By default the TUI spawns its own in-process gateway, so each TUI instance is
> self-contained — there's nothing to configure.
>
> You may see a `HERMES_TUI_GATEWAY_URL` env var referenced in the codebase or logs.
> This is an **internal wiring detail of the web dashboard**, not a user-facing
> remote-attach knob. When you open the dashboard's 'Chat' tab (`hermes dashboard` →
> `/chat`), the dashboard's web server spawns an embedded TUI child process and injects
> `HERMES_TUI_GATEWAY_URL` so that child attaches to the dashboard's own in-process
> `tui_gateway` over a loopback WebSocket (`/api/ws`). The `/api/ws` endpoint exists
> **only inside the dashboard server** (`hermes_cli/web_server.py`) and is bound to that
> process's lifetime and auth.
>
> There is **no general 'point any TUI at any standalone gateway port' mode.** In
> particular, the OpenAI-compatible API server (`hermes gateway` / the `api_server`
> platform) does **not** serve `/api/ws` — it's the model-backend surface
> (`/v1/chat/completions`, `/v1/models`, …) and deliberately does not expose the TUI's
> JSON-RPC control channel. Setting `HERMES_TUI_GATEWAY_URL` to that port will **404**."

`/api/ws` also serves as the desktop's own chat socket; Bot Screen uses a **sibling**
WebSocket at `/api/display/ws` ("same origin, same auth resolution as chat").

### 6.4 Dashboard design tokens (`web/src/index.css`, 255 lines)

Same `--color-*` shadcn-style alias layer, different generation:

```css
--foreground-base: #ffffff; --midground-base: #ffe6cb; --background-base: #041c1c;
--foreground: color-mix(in srgb, #ffffff 0%, transparent);   /* + -base, -alpha pair per role */
--theme-font-sans/-mono/-display; --theme-base-size: 15px; --theme-line-height: 1.55;
--theme-radius: 0.5rem;  --theme-density: comfortable;
--spacing: calc(0.25rem * var(--theme-spacing-mul, 1));
--radius-sm: calc(var(--theme-radius) - 4px);  --radius-md: - 2px;  --radius-lg: =;  --radius-xl: + 4px;
--series-input-token: #ffe6cb;  --series-output-token: #34d339;
--color-destructive: #fb2c36; --color-success: #4ade80; --color-warning: #ffbd38;
```

Three things the dashboard has that the desktop's CSS layer does not:
**`--theme-density: comfortable`** (an explicit density knob), **`--theme-display-font`**
(three-role typography: sans / mono / display), and **`--series-input-token` /
`--series-output-token`** — reserved chart series colors so analytics charts can't
collide with UI accents. Presets: `default`, `default-large`, `midnight`, `ember`, `mono`,
`cyberpunk`, `rose`, `nous-blue`.

---

## 7. `apps/shared` — the typed client

`apps/shared/src/` is imported by desktop (`@hermes/shared`, `file:../shared`) **and by
the TUI** (`@hermes/shared/format`, `@hermes/shared/skin`). Contents:

`gateway-contract.generated.ts` (5,904 lines, generated from `tui_gateway/contracts`),
`gateway-events.ts`, `json-rpc-channel.ts`, `json-rpc-gateway.ts`, `gatewayClient` reuse,
plus `ansi.ts`, `backend-scope.ts`, `billing-*`, `catalog-*`, `color.ts`, `format.ts`,
`fuzzy.ts`, `i18n.ts`, `model-search-text.ts`, `reasoning-effort.ts`,
`reconnect-backoff.ts`, `skill-scaffold.ts`, `skin.ts`, `slash.ts`, `theme-presets.ts`,
`translucency.ts`, `charge-settlement.ts`, `cron-trigger-controller.ts`.

`json-rpc-gateway.ts` — `ConnectionState = 'idle'|'connecting'|'open'|'closed'|'error'`,
`GatewayClientOptions` (transport injection, timeouts, request-id factory, socket-close
intercept, reconnect `replay`), and three timeouts with **documented reasons**:

```ts
const DEFAULT_REQUEST_TIMEOUT_MS     = 120_000
export const APPROVAL_RESPOND_TIMEOUT_MS = 300_000   // must match tools/approval_context.py approvals.timeout (default 300s)
const REPLAY_REQUEST_TIMEOUT_MS       = 10_000
const DEFAULT_CONNECT_TIMEOUT_MS      = 15_000
```

> "`approval.respond` rides the SAME deadline the backend grants the user to answer: a
> shorter client timeout races that window — the frontend has already rejected its own
> RPC while the backend happily applies the decision — the desktop shows 'request timed
> out' and freezes on a card that is actually resolved (#60654)."

> "A reconnect after sleep/wake must not hang forever in 'connecting' … fail to 'error'
> so callers can retry."

Also `DEFAULT_HEARTBEAT_INTERVAL_MS` / `DEFAULT_HEARTBEAT_DEADLINE_MS`, and a
`'*'` wildcard event subscription. Desktop-side: `apps/desktop/src/hermes.ts` is the
typed facade (`getHermesConfig`, `listAllProfileSessions`, `getLogs`, `hermesApi`,
`Gateway`); `lib/gateway-rpc.ts` has `isMissingRpcMethod` / `isMissingRestEndpoint`
capability probes, `connection-scoped.ts` handles per-connection scoping, and
`lib/backend-scope.ts` / `store/settings-scope.ts` keep gateway/profile/remote scoping
honest. `store/session-request-router.ts` routes a request to the right connection.

**Fuzzy matching lives in shared** (`shared/src/fuzzy.ts`) alongside `model-search-text.ts`
and `reasoning-effort.ts` — model picker search is therefore identical across surfaces.

---

## 8. Strengths

1. **Everything is a contribution, including the core.** `app/index.tsx` is six lines
   pointing at a registry-driven shell. The seams (`panes`, `statusBar.*`, `titleBar.*`,
   `palette`, `keybinds`, `routes`, `sidebar.nav`, `sidebarNav.prefs`, `layouts`,
   `appearance.extra`, session-row slots, transcript directives, themes) are a complete,
   enumerated extension surface. **This is the single most valuable thing in the repo for
   an overlay product** — you can ship a full tab/panel without forking.
2. **Layout is a real spatial model, not hardcoded CSS.** Binary split/group tree +
   persisted tree + FancyZones-accurate editor, with six shipped presets that state their
   geometry in weights. Presets are contributions, so user layouts and plugin layouts are
   the same type as built-ins.
3. **A genuinely principled theme system.** Seeds → `color-mix()` percentages → semantic
   `--ui-*` tokens, with the *same algorithm* in light and dark and only the mix
   percentages flipping. Every comment explains the perceptual reason for its number
   (shadow alpha, input border %, selection tint, comment contrast at 11px). Plugins are
   told to hardcode nothing, and the reference plugin's CSS proves the discipline.
4. **Design-system primitives are importable by plugins.** 100+ `components/ui/*`
   + `Codicon` + the whole `@ui-*` token layer, so third-party UI is native by
   construction rather than by imitation.
5. **The comments are a design rationale, not a changelog.** Nearly every unusual
   constant has a stated reason and often an issue number (`#71627`, `#76185`, `#92569`,
   `#94260`, `#123085`, `#60654`, `#49340`, `#91603`, `#21086`, `#61392`). For a design
   reference this is unusually high value — the *why* is recoverable.
6. **Transparency is solved once, correctly.** The "one painter" glass model with nested
   surfaces is the right answer to layered-alpha compositing, and it's written down.
7. **The tool-call ticker is a better answer than a list.** One line that rolls, with
   `run-summary` prose in a fixed clause order, reads as narrative rather than log.
8. **TUI details-mode with per-section defaults** is the most reusable TUI idea — an
   opinionated streaming transcript instead of a wall of chevrons, with explicit
   precedence and a config shape that stays backward-compatible.
9. **Attention to terminal reality.** OSC 11 background probing, three mouse-tracking
   presets with a stated tmux failure mode, matched glyph widths to stop status-bar
   jitter, width via `event.code` on macOS where `⌥+letter` emits a symbol.
10. **Failure modes are documented and defended.** Connect timeouts per state, approval
    timeouts matched to the backend window, reconnect replay bounded, PTY reconnect with
    exponential backoff and resume sanitization, protocol noise never touching the
    terminal, IME/dead-key forwarders.
11. **Capability probing over capability assumptions.** `isMissingRpcMethod`,
    `isMissingRestEndpoint`, `ctx.socket` no-op on OAuth remotes with a mandated polling
    fallback, `requestTheme()` returning `false` as an availability check rather than
    coercing the user.
12. **The kanban plugin's theme-immune CSS is a template.** Namespace everything, reset
    the theme's global rules with `!important` inside your container, opt back in only on
    your own classes.

---

## 9. Weaknesses — honest

1. **Enormous surface area, shallow default discoverability.** The desktop has ~200
   stores, ~100 UI primitives, 61 keybind actions, 6 layout presets, 9 settings sections,
   14+ routes, 18 nav/tier rows, plus overlays, a HUD, Quick Entry, a Command Center, and a
   ⌘K with ~20 group headings. The docs are the only map, and they're 37k+ words. A new
   user has no route to "the thing I want"; ⌘K is the intended answer, which is itself a
   sign the surface is too large for chrome to express. **For SamAgent, resist this.**
2. **The keybind map has collisions that leak.** ⌘1–⌘9 are *simultaneously* tab slots,
   profile switches, and (via `mod+alt+1…9`) profiles 10–18. `⌘G` is Review normally and
   find-next only while the find bar is open. `⌘L` is focus-composer and
   selection-to-composer in the same gesture with a hand-written priority ladder
   (`composer/focus-chord.ts`). The `passthrough` mechanism handles this gracefully, but
   the fact that a chat app needed positional session slots *and* positional profile
   switches *and* positional tab slots on one chord family is a design smell.
3. **Inconsistent defaults is a deliberate choice that leaks confusion.** "Ships unbound so
   a user who doesn't want a chord claimed" is defensible — and applied to
   `composer.dictate`, `composer.reasoningUp/Down`, `session.archive`, `session.togglePin`,
   `view.cycleSidebarGrouping`, `view.toggleProfileRail`, `view.toggleSimpleMode`,
   `view.showFiles`, `profile.create`, and six `nav.*` rows. But a user must learn that
   half the things they expect are behind a panel, and `view.showFiles` being unbound
   (while the *files pane* is a core default-layout pane) is a genuine inconsistency.
4. **The design system's own comments flag unresolved theme fights.** Light-mode
   comments needed a manual hex remap because `github-light-default` fails at 11px.
   `--dt-input-border` differs between modes purely because a dark card needs a lighter
   resting border. `--ui-widget-surface-background` needs a manual 88%/`#000` step-down
   "so an inline widget doesn't read as the brightest thing in the transcript." These are
   symptoms of an under-specified code surface; third-party themes will keep finding them.
5. **Preset drift is acknowledged in-tree.** `BASIC_TREE`'s comment: "A tree that simply
   omitted them was a lie — applying it adopts every missing pane back in as workspace
   tabs, which is Focus." And `presets.ts` heals user presets that "cloned the live tree
   and so baked in `session-tile:` / `preview-tile:` / `route-tile:` pane ids (#94260)."
   A preset model that can be falsified by a live-tree clone is fragile; the healing
   code is the admission.
6. **Simple vs Advanced mode adds a second axis of confusion.** The resolver
   (`sessionReveal ?? policy[mode] ?? userPreference`) plus per-surface storage scoping
   (`modeLayout`) plus `modeBound` atoms plus `tier` tags plus "a toggle pressed while a
   surface is shadowed lands in the session layer" means **layout state has three
   storage scopes**. A user who arranges a layout in Simple and switches to Advanced
   finds a different arrangement, and the docs do not make that discoverable.
7. **Depth of surface exceeds depth of onboarding.** `driver.js` tours exist, but there is
   no progressive disclosure of the *system* (busy vs awaitingResponse, streaming vs
   settled segments, tool ticker vs tool cards, three session-id flavors —
   `sessionId` / `focusedSessionId` / `focusedStoredSessionId`). The plugin SDK has to
   explain `focusedStoredSessionId` vs `focusedSessionId` because the underlying model
   has two id notions.
8. **The TUI's chat-in-dashboard is the weakest link in the product.** A full agent UI
   delivered as PTY-over-WebSocket-into-xterm.js means: no native selection semantics, IME
   and dead-key workarounds, manual right-click-to-select-word, `Ctrl+C` behaving
   differently with/without a selection, lost scrollback, and browser-only chrome. The
   reuse is enormous (one TUI codebase), but the result cannot be styled to match the
   dashboard, and every browser/xterm divergence becomes a bug in the Chat tab.
9. **Three surfaces, three token systems.** Desktop uses `--ui-*`/`--dt-*` with a seed/mix
   algorithm; the dashboard uses `--color-*` with `--theme-*` scalars plus a
   `--theme-density` knob the desktop lacks; the TUI uses hex `ThemeSeeds`. They share
   `@nous-research/ui` primitives and `apps/shared`, and the theme *presets* overlap
   (`nous`, `midnight`, `ember`, `mono`, `cyberpunk`), but a `--theme-density` change in
   the dashboard has no desktop analogue. Expect drift.
10. **`~/.hermes/state.db` as a single point of failure.** Sessions, FTS search, and the
    gateway share one SQLite file with WAL contention across processes — there is a whole
    doc page (*Session Storage Recovery*) about "another process holds an old copy of the
    session database's write-ahead log."
11. **Model-switch cost is a UX trap the docs admit.** "Switching the model inside a live
    chat means the next message re-reads the whole conversation at full input price…
    on a long chat, a fresh chat on the new model is often cheaper than bouncing back and
    forth." The UI offers the cheap-looking action and cannot prevent the expensive one.
12. **Plugin trust surface is broad.** A plugin gets gateway JSON-RPC (`host.request`),
    reactive state, the React Query client, storage, i18n, native notifications, an OS
    door, and full-page routes. There is no declared per-plugin permission manifest for
    the desktop SDK (the Python half has capability scoping; the JS half's comment says
    notifications are "gated by Settings ▸ Notifications ▸ 'Plugin notifications'" and
    throttled per plugin — that's presentation gating, not capability scoping).
13. **Onboarding cost of the desktop binary.** macOS/Windows/Linux Electron + optional
    `hermes desktop --cwd` + WSL2 GPU hints (`GALLIUM_DRIVER=d3d12`), Wayland compositor
    workarounds, Hyprland IPC, `ozone_platform_hint: x11` for COSMIC. The WSLg,
    Wayland, and Hyprland handling is genuinely excellent engineering and also evidence of
    how much surface there is to support.
14. **Comments-as-documentation cuts both ways.** The code comments are exceptional, but
    some are longer than the code and several read as design-doc fragments pasted into
    source (the glass "ONE PAINTER" essay, the `passthrough` rationale, the WHY behind every
    chord). A new contributor must read a lot of prose to learn a small amount of
    mechanics.

---

## 10. What to lift into SamAgent (concrete)

**Architecture**
- Register against `apps/desktop`'s contribution registry rather than forking; core areas
  are already plugin-facing. `ROUTES_AREA` + `SIDEBAR_NAV_AREA` + `PALETTE_AREA` is the
  complete "add a tab" recipe.
- Model panes as a split/group tree with named zones (`workspace`, `sessions`,
  `terminal`, `files`, `review`) and ship at least two layout presets (a chat-first one
  and a tooling one) rather than one.
- If a plugin ships a Python backend, declare `"api": "plugin_api.py"` and reach it via
  `/api/plugins/<id>` (`ctx.rest` / `ctx.socket`), keeping a polling fallback for OAuth
  remotes.
- Generate typed contracts from `tui_gateway/contracts` rather than hand-writing them,
  and pin client timeouts to backend windows (approval = 300 s) in one shared constant.

**Design**
- Seed + `color-mix()` theme generation with per-surface mix percentages; flip
  percentages, not algorithms, for dark mode.
- Semantic ordinal tokens (`--ui-text-{primary,secondary,tertiary,quaternary}`,
  `--ui-stroke-*`, `--ui-bg-{chrome,sidebar,editor,elevated}`), never component-named.
- One **category color per context-usage bucket** so the meter explains itself.
- Contrast-audit syntax themes at the real code size and remap specific hexes
  (`#6e7781 → #57606a`) rather than switching whole themes.
- Bundled mono font so code/diff matches the embedded terminal.
- One glass painter with nested surfaces, not per-surface alpha.
- Cursor-style diffs: color + a 2px gutter accent, no `@@`/file-header noise.

**Chat**
- The **tool ticker**: one rolling line with prose summaries in a fixed clause order,
  routed by tool name so categories stay honest.
- `ExpandableBlock` with `scrollHeight > 121` → `max-h-[7.5rem]`, expand to `max-h-[40dvh]`,
  a `pointer-events-none` fade and a compact toggle clear of the scrollbar.
- Drop reasoning groups with no text; follow live reasoning until the user scrolls up.
- Async cron/delegation results as collapsed Markdown disclosures that explicitly omit
  the instruction envelope.
- Per-section visibility (`hidden|collapsed|expanded`) with opinionated defaults and a
  documented precedence chain.
- Composer affordances worth copying verbatim: queue-edit-on-stop, ↑/↓ prompt history,
  shrinkable truncating model pill with no arbitrary `max-w`, status-section stack with
  collapsed-state preview.

**Extension**
- Namespace contribution ids `${pluginId}:${id}`; give each its own error boundary and a
  per-plugin disposer so disable/reload is clean.
- Copy the plugin pitfall list into your own authoring guide — every item in it is a real
  bug someone shipped.
- Prohibit hardcoded colors in plugin code at review time; require reading tokens via
  `getComputedStyle` for canvas.
- Copy the kanban plugin's theme-immune CSS comment verbatim as the pattern.
