# Desktop Design System

Conventions for the Electron desktop app (`apps/desktop`). Read this before
adding a component, overlay, or style. The rule of thumb: **one source per
concern, tokens over literals, flat over boxed.** If you reach for a raw color,
a one-off shadow, a bespoke button, or a hardcoded `px-*` on a control — stop,
there's already a primitive for it.

This file owns the visual and interaction contract. Read
[`AGENTS.md`](./AGENTS.md) for architecture, state, resolver, transport, and
testing rules.

This doc contains two kinds of content, maintained differently:

- **Principles** (flatness, intent, feedback, motion, cancellation) are durable.
  They hold as components come and go.
- **Named contracts** (tokens, `Button` variants, primitive names) are the
  design system's current API. They are maintained _with_ the code: if you
  change a primitive, token, or variant, update its entry here **in the same
  change** — a stale name in this file is a bug, exactly like a stale type.

When a rule and the code disagree, fix whichever is wrong rather than forking a
one-off at the call site.

## Principles

1. **Flat, not boxed.** No card-in-card, no divider borders inside a panel.
   Group with whitespace and a single hairline, never nested rounded boxes.
2. **Borderless elevation for floating panels.** Overlays float on
   `shadow-nous` + a `--stroke-nous` hairline, not thick framed boxes. In-panel
   structure may use token hairlines sparingly.
3. **One primitive per concern.** One `Button`, one set of control variants,
   one `SearchField`, one `Loader`, one `ErrorState`. Migrate onto them; don't
   fork.
4. **Tokens, not literals.** Reference CSS vars (`--ui-*`, `--shadow-nous`,
   `--theme-*`), never raw hex / ad-hoc rgba in components.
5. **Style lives in the primitive.** Variants and sizes own padding, radius,
   color, chrome. Call sites pass a `variant`/`size`, not `className` overrides
   that re-specify those.
6. **Intent before automation.** Surface useful actions and previews, but do not
   open panes, move focus, or navigate because a tool happened to produce
   something.
7. **Immediate feedback.** Direct manipulation updates the view first. Network
   or disk persistence reconciles afterward and rolls back visibly on failure.

## Information architecture

- **Chat is the home surface.** The transcript and composer stay primary; tools,
  previews, files, review, and terminal complement the conversation.
- **Pages are durable destinations.** Chat, Skills, Messaging, and Artifacts
  remain in shell chrome. Do not hide a distinct product noun inside an
  unrelated page.
- **Route overlays are short tasks.** Settings, Command Center, Cron, Profiles,
  Agents, and Starmap render as `OverlayView` cards and return to the previous
  route on close. Model/session pickers and dialogs layer above the current
  surface; they are not navigation stacks.
- **Panes are working context.** Preview, files, review, and terminal remain
  attached to the current task. Their state survives temporary hiding and chat
  switches where the underlying tool is meant to persist.
  Activating a tool tab also restores its open state; hiding a stacked terminal
  must not leave a visible tab pointing at an inert shell.
  Review follows the focused chat's worktree and sends agent-ship actions back
  to that same conversation, including when focus moves into tool chrome.
- **One action, one home.** A command may have keyboard, palette, and visible
  affordances, but they invoke the same action and state. Do not fork behavior
  per entry point.
- **Projects own workspace cwd.** Use Sidebar → Projects for local folders and
  worktrees; do not reintroduce a per-session/right-sidebar folder-picker flow.

Profile icons and condensed profile rows offer **Open in new window** and
**Set as default** in their existing context menus. Opening a profile creates a
full peer window without switching the source window. The desktop default
applies at startup and to generic new chats; explicit profile/project actions
and profile-specific windows keep their own destinations. Changing the default
does not move existing sessions or replace the active conversation.
Ordinary **New Window** (`⌘⇧N` / `Ctrl+Shift+N`) inherits its opener's device and
profile only at startup, not as a window-specific default. Later device/profile
selections remain authoritative for new chats unless a desktop default is set.

Navigation must preserve context. A background session finishing, a tool result
arriving, or a project refresh may update badges and cached data; it must not
replace the foreground transcript or steal focus.

The boxed conversation sidebar follows Figma Desktop 10 (`147:7470`, machine
box `143:6270`), not the alternate crowded-agent card mockup. Each expanded
machine has a 6px container, 5% neutral body fill, and a brighter 7% selected
heading; the heading's lower corners join the body and round when collapsed.
Agent rows are transparent, 45px tall, with 10px horizontal padding, 7px between
groups, and 20px colored robot tiles using the exported Figma artwork. Do not
replace them with blue selection cards or large antenna-outline avatars.
Selected/hovered names brighten without painting an agent-row background.
Names, descriptions, colors, and connection status use actual owner metadata;
never invent team names, sharing counts, or connection state.

The agent name/avatar opens that owner's canonical chat without a redundant
child row. Its menu offers **New chat with this agent**, **Open agent**, and
**Hide from chat list**. Collapse remains a separate keyboard/touch control:
reserve a 16px leading gutter so its 24px target stays inside the machine card
without overlapping the avatar. Nested 31px session rows have a 1px, 20% neutral
indentation guide centered beneath the avatar, 36px from the machine's edge.
Titles align with agent names at 54px. Empty and collapsed agents have no
dangling guide. Session hover and selection brighten only the name, like agent
rows; the click target still includes the guide gutter. Reveal actions for
hover, visible keyboard focus, or an open menu, not lingering mouse focus;
touch controls remain visible. Only truncated titles need a tooltip.
Screen is a tool rather than a conversation: use **Open Screen** in the command
palette for the focused owner; classic/Bot Mode entry points remain available.

**Hidden agents** stays outside the agent list's scroll viewport so its count
and attention dot remain reachable in crowded installations. Its expanded rows
have their own bounded scroll area. **Show** restores an agent; a row needing
attention opens the source conversation when its owner-qualified route is known.
Hiding affects presentation only: chats remain searchable, work and notifications
continue, and the current conversation is not closed or replaced. Hidden agents'
pinned rows disappear from the ordinary list but remain available in search.
Visibility and expansion persist by `(connectionId, targetProfile)`; local/null
owners normalize identically. Renaming migrates preferences and deletion clears
them. Scroll position and the Hidden agents disclosure survive navigation/reload.

Attention priority is approval, input, failure, running, then unread. Use existing
owned runtime/session signals and owner-tagged schedule snapshots, not guessed
state or extra polling. Same-name agents and colliding session IDs on different
machines must not borrow each other's signals. Display elapsed time only when
the source supplies a real timestamp. An unobserved remote failure cannot be
inferred from a reachable roster entry.

Finished settings and management frames use bundled, OFL-licensed Oswald for
display headings and Mona Sans for body text. The Figma trial navigation font
is not redistributed. Redesigned geometry is root-scoped because route
overlays render outside the shell: 60px rails, navigation up to 320px, and a
640px settings reading column. Chat uses the wide layout described below. Native titlebar sizing,
user-controlled panes, themes, and the classic layout remain functional.

Appearance starts with working Light/Dark/System previews, text size,
language, and tips controls. Advanced retains the full settings and theme
gallery, including deep links. Agent management uses owner-scoped Overview,
History, and Settings tabs. Settings has its own grouped, agent-only navigation:
184px navigation, 32px gap, and a 584px body at full width. Model selection,
fallbacks, and collapsed Advanced controls share one page; provider setup stays
available inside the model picker. Reuse the existing editors and save paths.
Auxiliary-model deep links reveal enclosing Advanced disclosures before
focusing the target or consuming the link. Activating a settings owner preserves
the requested page rather than redirecting to a new chat.
Navigation and editor bodies scroll independently; changing section/subpage resets
only the editor scroll. Instructions use a 322px card including the Save footer.
Classic dispatch preserves the same semantic destination and owner. Retained
instruction revisions and config draft/baseline/save queues live above the
presentation branch, not in duplicate hidden editors; a pending save cannot erase
a newer draft after switching presentation.
The overview header uses a 40px light Mona Sans title and the outline avatar.
**Open chat** sits beside the action menu: duplicate, set default, hide/show in
the chat list, copy the actual profile path, and a hint directing deletion to
Settings → General. Menu rows have 44px targets. Disconnection warnings reflect
the backend's actual failed/retrying/stopped states for configured, enabled channels.
Overview cards use 436:300 columns with a 24px gap, 16px section-title spacing,
44px rows, 64px activity rows, and 4px corners; narrow layouts collapse to one column.
Global desktop settings remain available outside the agent editor. New agent
opens creation directly on the captured machine rather than another management page.

User-facing identities are **Machine / Machines** and **Agent / Agents**;
scheduled automation is **Schedule / Schedules** in both layouts. Navigation,
page headings, and plugin surfaces reuse `settings.connections.title`,
`profiles.title`, and `cron.title`; their nouns come from
`src/i18n/desktop-terms.ts` for all nine bundled locales. Service operations say
**Restart Hermes**, not “restart the machine,” which would imply rebooting the
computer. Reuse existing catalog actions and descriptions rather than introducing
local synonyms. Team Gateway and Tool Gateway remain product names.
Actual Telegram/Discord bots, browser profiles, and user-authored names retain
their meanings. Protocol fields, `/profiles` and `/cron` routes, CLI commands,
cron expressions, plugin IDs, and the exact stored `Bot Chat` title are technical
contracts, not copy to rename.

Drift coverage exercises navigation into Agent and Schedule editors, cancellation,
and return to the current view with the redesign enabled and disabled. Component
tests use canonical translated controls while checking ownership, routing, and
mutations; avoid source scans or snapshots that freeze incidental sentences.

Owner actions preserve the roster's backend `targetProfile`, including aliases
before Bot Mode has mounted. History uses bounded 25-row requests with offsets,
deduplicates pinned backfills, and keeps earlier pages reachable past 500 sessions.
Source/archive filters are server-side; explicit text search and cleanup walk
remaining bounded pages. Read-only previews expose Open chat, Archive, and Delete.
Cleanup previews its candidates before confirmation and rechecks owner, running
state, and eligibility before each fenced deletion. Protected/running sessions
are skipped; failed writes are never counted as deleted. Selection and confirmation
belong to one gateway/profile and are discarded when that owner changes.
History dialogs use a dimmed, unblurred backdrop, 24px content padding, display
headings, and semantic destructive colors for confirmed deletion.
Bulk pin/archive operations retain failed selections and report partial success.
Search, independent filter chips, and pagination reuse the same owner-scoped queries.
The read-only History preview is a right-edge drawer below the real titlebar,
not a centered dialog. It explicitly resets both shared-dialog translation axes;
at narrow widths it fills the available viewport without hiding its actions.
Session lists capture their serving connection before awaiting the response;
cached rows never inherit a newly selected foreground gateway. Selection follows
the focused conversation's owner. Explicit owners survive tab and window opens.
Primary-local project rows retain their legacy profile-door shape; deletion
normalizes both lookup claims and project-only targets using the tree's captured
source. Never infer that source from the newly foregrounded machine.
Conversation settings links resolve profile-only chats through the primary source,
not the ambient connection; unresolved owners cannot open another agent's settings.
The current per-window runtime caches cannot host identical stored IDs from two
owners simultaneously: conflicting opens fail visibly without reassigning the
existing chat; use separate windows for those conversations.
Tile identity comparisons live in `src/store/session-tile-identity.ts`; strip
ordering and close-to-next selection live in `src/store/session-tile-order.ts`.

Agent-profile settings use `agentProfile` in the URL; the plugin settings
contract keeps `agent` for the plugin key. Settings pin the complete gateway and
profile on entry, including links that omit an explicit connection.
Config, credentials, messaging, memory, and plugin operations retain this owner
through asynchronous work. Same-name agents on another gateway keep their own
configuration, plugin cache, conversations, and open tiles.

Keep section resolution, visibility, and model-default derivation separate from
editor composition. Refactoring these boundaries must preserve owner checks and
must not mount inactive capability editors or pending inventory content.
The agent header uses the Figma robot and display type. The directory count and
default badge come from the owning machine's actual metadata, never sample data
or a same-named agent on another machine. Each settings page supplies its own
localized description. Instructions keeps the SOUL.md heading, proportional
editor text, character count, and Save footer without changing draft ownership.
Voice groups Conversation, Listening, and Speaking; the mode selector exposes
only backend-supported values, while detailed provider/pricing help stays under
Advanced. Schema/config availability still determines which provider fields exist.

Page toolbars occupy the native titlebar band on the window canvas so open
chat/tool tabs cannot bleed through; Back is a 30px chrome button and controls
remain outside the drag region.

The redesigned shell is a window canvas with panels laid on it
(`app/redesign/portal-shell.css`, tokens in `portal-colors.css`). The titlebar
band and the 60px rails show the canvas (the skin's `--portal-skin-window`; Hermes
Portal dark `#0a0a1a`, light `#ededed`). Zones touching the window top become
panels starting below the band; only the row's outer top corners round at 9px,
and neighbours share a 5% hairline. Dark is one navy family (Variation 1 147:7470,
pages 895/856): the chat sidebar, page navs and side tool panels are the canvas at
.90 (translucent, so the wing reads through them at a tenth of its strength), the
stage is the skin chrome at .96 (`#07071a`); light panels lift toward white (30% /
90%). Pages lay their body on the canvas; their nav panel carries its own
viewport-registered wing layer (`--portal-page-panel-fill`). The wing artwork is
painted once, on the canvas, and reads only through translucent panels. Window Glass
(macOS default) never switches the redesign off: its rules are unlayered and beat
the `@layer base` Glass rules, and only the canvas (rails, band) takes the native
material, tinted with the portal canvas at the user's Glass intensity. Titlebar, rail and pane
header buttons are 30px r3 chrome buttons: 2% ink fill, quaternary glyph,
hover 5%, pressed 10%, the open destination or shown pane in brand selection
with a 1px brand ring. The titlebar band is a shell property: 60px in the
redesign (Figma's header; content starts at y 60), 34px classic (44px browser
touch). Electron reads the shell choice from the shared UI preferences before
creating a window and re-seats the macOS traffic lights (centred, in the rail
column) and the Windows/Linux caption overlay when it flips
(`electron/window-chrome.ts`); the popped-out Browser keeps the classic band.
The band carries the leading cluster past the rail, the open session's title
pill with "+" at the stage's leading edge, and the panel toggles at the end.
Workspace pages (Capabilities, Library, …) share the overlay pages' square,
full-bleed frame. Without an explicit skin pick, the redesign paints the
Hermes Portal skin and classic paints Nous (`defaultSkinName`); flipping the
shell never writes a pick.

Redesigned surfaces use neutral foreground-derived fills and strokes, not
accent-tinted greys. Global settings cards use 5% fill and 5% borders with 8px
corners; agent settings cards use 2% fill, 5% borders, and 4px corners.
Settings-navigation selection uses neutral 5% fill and a 0.5px neutral
5% inset stroke with 6px corners; sub-pages reveal under the active row, without
per-row carets. Settings pages compose `SettingsOverview` / `SettingsCard` /
`SettingsDisclosure` (`settings-card.tsx`), which fall back to the classic wrapper. Brand-blue selection remains on the global
rail. Agent-management navigation tiles stay translucent; agent rows and the
overview header use the Figma robot with the real owner's color.
The global rail tracks the owning agent even inside its settings editor.
Shared rail/sidebar geometry lives at the document root while the redesign is
enabled. Sibling route overlays reserve the rail with logical inline offsets
in both LTR and RTL layouts.
Checked switches use the brand-blue track and light thumb across settings
and the capability catalog; unchecked tracks are neutral 14% fills.

Below New session and search, the redesigned sidebar follows Figma 147:7470
("Variation 1: Flattened gateway/host"): sections under 10px uppercase captions
separated by whitespace (Agents carries a brand count badge), pinned rows of 30px
with a 7px leading dot (row ink at rest, brand for the open chat or unread news, the
status tones while a turn runs or waits), machines as 40px rows (name, inline
organisation tag, status dot and label at the end), an expanded machine as one 5%
container under a 7% header bar, agents as a 20px solid tile in the agent's snapped
Figma primary (`lib/portal-profile-color.ts`) over a name and a second line (roster
description, else the chat count), and an agent's chats on a 31px pitch under a 1px
guide. Hover is a 4% fill (2% on agents), the open pinned chat a 2% fill. Row and
agent actions (ellipsis, caret, kebab) are bare glyphs that appear on hover, keyboard
focus or an open menu; touch keeps them visible. Arrow keys, Home and End walk the
visible rows; right-click opens the row menu. Captions and meta keep the AA ladder
where Figma's inks fail 4.5:1. The band's leading cluster starts at
`--portal-band-start` (x 130 past macOS traffic lights, else 80) and the session pill
is anchored to the stage's leading edge with CSS anchor positioning, so it tracks a
sash drag in the same frame.
The global Agents icon uses the Figma robot silhouette tinted by `currentColor`,
sharing its peers' inactive, hover, and selected tones in both light and dark themes;
rail destinations rest at .8 ink and the open one wears the brand selection with a
60% brand ring.
The Hermes Portal theme uses the Figma pages' navy (`#0a0a1a` canvas, `#07071a` stage) and
floating surface (`#0d0d1e`) without generic neutral blending; the shell's panels
derive from those two colours. Foreground-derived
card fills remain subtle; custom theme choices and classic styling stay available.

The **Redesigned interface** preference gates the setup guide, catalog/agent
controls, composer layout, machine menu, and hidden-agent presentation as well as
the shell. One seam owns it (`src/store/shell-variant.ts`): `usePortalShell()` /
`$portalShell` say whether this window renders the redesign; `installShellAttribute()`
(main.tsx, before first render) mirrors it to `<html data-shell="portal">` and follows
every toggle (absent in classic). Every redesign selector is gated
`:root[data-shell='portal'] …`; never gate on a mounted component
(`:has(.portal-sidebar)`) or on a shell ancestor. The chromeless HUD and
popped-out Browser windows stay classic in both modes. `$portalPreferred`, the
window-independent preference, is only for behaviour every window must agree on
(setup flow, metrics, session scope, the plugin SDK). Brand fills
(`--portal-brand-fill`, `--portal-on-brand`) are the same in every theme and mode;
thin rings and blue text use `--portal-brand-ink`, which lightens on dark. Floating
surfaces (`app/redesign/portal-floating.css`) are part of the redesign, not of the
Hermes Portal theme, which carries palette and typography only: solid elevated fill,
hairline stroke, r4 menus / r8 popovers and dialogs, Figma's single drop shadow
(lighter in light mode), no backdrop blur. Shared primitives take the Figma page
control scale (36px, r3) inside pages and dialogs via `portal-controls.css`.
Turning it off restores classic onboarding and all classic agent rows without
migrating conversations or destroying the redesign's saved visibility.
Chat chrome leaves the keyboard tab order when covered by a route overlay, without
remounting the draft-owning host. The session-search shortcut reopens and focuses
the redesigned sidebar. Global settings keep seven primary destinations; extra
configuration remains under Advanced, with active deep links revealed.
An unqualified Settings entry opens Account & Plan (billing) in the redesign rather
than a legacy Model page tucked under Advanced; explicit destinations and the classic
default remain unchanged. The redesigned rail follows Figma 823:15804 ("does this
change anything for another agent?" → global, else Agent settings): Account & Plan,
Appearance, Notifications, Keyboard Shortcuts, Gateways & Machines, Sessions & History,
About & Updates, Advanced (device-level and remaining global pages), and an Agent
settings card. Provider accounts and keys live under Agent › Providers & Keys.

Transcript styling is isolated in `components/assistant-ui/thread/redesign.css`,
following the Figma conversation (`491:40346`): one centred column shared with the
docked composer (640px, `--portal-conversation-*` in `app/chat/stage.css`), prose
20px inside it, 30px turn gaps and 20px output gaps. User bubbles hug their text at
the right edge up to 380px, with 6px corners, 6/10px insets and a 5% (light 2%)
fill and no border; their time and restore actions sit beside the bubble. Base text
is 13px at 1.6 leading before user scaling. A pinned prompt keeps 12px of clearance
below its bubble so clipped reply lines never touch it. Retain the current branded
body font, timestamp preference, and working message actions; enabled timestamps
use caption ink and sit above the bubble. Collapsed tool runs, including
single-tool runs, use the dashed [2,4] summary frame with a status dot. The stage's
hero photo is luminosity-blended at Figma's crop behind the landing and fades to a
trace behind a conversation. The composer has no ring: its static 2px spectrum edge
is full strength while it holds focus and dims when focus leaves; its native base
text is 14px before the
user's scale, retaining the browser minimum. Floating dialogs inherit redesigned padding, display typography, and
brand-blue primary actions and the shell's corner radius through root-scoped
rules; their classic defaults remain unchanged. Route overlays and body-portalled dialogs are siblings of the
chat shell, so their redesign styles must not require the shell as an ancestor.

Redesigned setup is a sessionless preferences/model guide. It resumes from
root `onboarding.run` / `onboarding.step`, retains a working provider, and offers
the free model only when the backend reports eligibility. Finishing settles setup
without replacing the existing draft, attachments, owner, or route; no setup
transcript is created. Explicit sessionless migration retires legacy setup
metadata/cards without deleting conversation history and
broadcasts the end once so connected classic clients leave setup immediately.
Resuming an existing guide is read-only; the first-start latch preserves newer
peer progress and cannot reopen completed setup. A completed/skipped guide with a
persisted step stays settled after a fresh renderer launch; this does not
acknowledge unrelated account-level model notices. Older runtimes show an explicit
upgrade/retry offer rather than silently dropping setup or creating a setup chat.
Local-model configuration releases the modal and leaves a **Resume setup** action.

Global Capabilities is the inventory/catalog entry; agent settings owns
enablement. Agent Tools groups On/Off switches with per-tool configuration
disclosures; the page-level Advanced disclosure retains Keys and Config.
Installations and tool enablement retain their actual agent and machine scopes.
Hosted connector accounts belong to the Nous identity: agents
sharing that identity share those accounts, and removing one affects them all.
Read-only command search must not replace an agent editor's writable inventory.

Usage collection belongs to a consenting connection/profile, not merely the
foreground UI. Changing scope revokes the old metrics binding immediately;
events during rebinding are discarded, never replayed under later consent.

The redesigned composer is 480 × 148 on an empty landing. In a populated native
conversation it keeps the 148px base height, spans the shared 640px column and
docks 30px above the stage floor, with model/effort as quiet right-aligned text above the input and attachment, microphone, and labeled Send
controls below. Browser touch targets keep their larger accessibility sizing.
Gateway menus distinguish connection from inference
readiness. Simple mode keeps reconnect and lazy recent activity; Advanced adds
system/restart/logs, with 28px icon controls and destructive restart styling at rest.

## Surfaces & elevation

Floating panels (base `Dialog`, route overlays, boot/install/update surfaces,
model-picker, onboarding, prompt overlays, notifications) use:

```
shadow-nous           /* downward-weighted, layered contact→ambient falloff */
border-(--stroke-nous) /* currentColor hairline, theme-adaptive */
```

Both are CSS vars in `src/styles.css` — tune in one place, everything inherits.
Don't add per-overlay `shadow-[…]` or `border-(--ui-stroke-secondary)`
one-offs; if elevation needs to change, change the token.

Menus and popovers use their own shared `shadow-md` +
`--ui-stroke-secondary` primitive treatment. Every floating list —
`DropdownMenu`, `Select`, and Popover + cmdk pickers
(`<PopoverContent variant="menu">` + `<Command variant="menu">`) — paints
through `src/components/ui/menu.ts`, so a list reads the same wherever it
opens. Typed fields with suggestions use `ComboboxInput`, never
`<input list>` + `<datalist>` (Chromium paints that as its own OS popup).

Drag affordances may use tokenized
dashed targets and local blur. These are semantic surface classes, not licenses
for call-site shadow or border inventions.

**Queued cards:** `CardStack` (`src/components/ui/card-stack.tsx`) consumes a live,
keyed list, retaining the current item when more arrive. Inline and floating
approvals and both toast placements share its gesture handling and geometry.
The Cursor-reference treatment uses one 96%-scale silhouette 7px above the
front, 220ms promotion, and 180ms upward clearance; no rotation or lateral throw
on button/keyboard decisions. Consumers supply the existing surface tokens and
own the exact-request response. Gestures never grant approval. Departing cards
are immediately inert; toasts can expand to the full live list. One persistent
transcript-level host owns approvals, independent of tool rows and assistant
message boundaries. Prepared approvals can precede tool.start: execution must
not relocate or remount the stack. While approvals remain, a real activity line
above the cards changes from awaiting approval to current-turn command status;
represented execution rows appear only when explicitly expanded. Empty text
continuations must not introduce paragraph gaps. Keep inline approvals beside
the conversation and let genuine content scroll normally; do not inject padding
or write scroll offsets to pin the decision. Preview this order with delayed
start and completion events, not pre-created tool rows. Final approval removal
retires both the painted card and its measured layout footprint; restoring tool
rows must not insert their full height before the outgoing stack can settle.
No completion callback may clear the measurement of a newly arrived card.
Reduced motion settles immediately without retaining empty clearance.

## Window background behavior

Settings → Appearance → Window layout offers **Minimize to tray**, off by default and
local to this desktop installation. When enabled, minimizing ordinary windows
hides them without stopping their work. Close, Alt+F4, and Cmd+Q keep their
normal behavior. The tray's **Show Hermes** restores hidden windows;
**Quit Hermes** keeps the ordinary active-work confirmation and teardown.
On macOS the tray lives in the menu bar; the Dock icon hides only when no normal
window remains visible and returns on restore. If the tray is unavailable,
ordinary minimize/close behavior is retained rather than hiding an unreachable app.

## Window glass

Glass defaults to **29% Tint, Sidebar only** in both light and dark appearances.
Fade defaults to zero so the content column and text stay opaque. Native frost
keeps its platform/appearance defaults. Explicitly saved settings take precedence;
changing defaults must not overwrite a user's existing choices. The shared
`apps/shared/src/translucency.ts` resolver owns these defaults for both the
renderer and Electron's first window paint.

## Stroke & color tokens

| Token                                        | Use                                                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `--ui-stroke-primary…quaternary`             | hairlines, in descending strength                                                           |
| `--ui-stroke-tertiary`                       | the default in-panel divider / list hairline — and every bordered surface in the transcript |
| `--stroke-nous`                              | the overlay hairline (pairs with `shadow-nous`)                                             |
| `--ui-text-primary / -secondary / -tertiary` | text hierarchy                                                                              |
| `--ui-bg-quaternary`                         | soft control fill (secondary button)                                                        |
| `--ui-widget-surface-background`             | fill for inline chat widgets (`WIDGET_SHELL_CLASS`)                                         |
| `--chrome-action-hover`                      | hover fill for quiet controls                                                               |
| `--theme-primary`, `--ui-accent`             | brand/accent                                                                                |

Never hardcode `border-gray-*`, `bg-white`, `text-black`, etc. The two tiles in
`BrandMark` (white in light mode, `#0d1117` in dark) are the sanctioned literals
(the mark needs a fixed backdrop).

## Buttons — one component

`src/components/ui/button.tsx` is the single source. Pick a `variant` + `size`;
do **not** pass `h-*`, `px-*`, `py-*`, or icon-size overrides.

**Variants:** `default` (primary), `destructive`, `secondary` (soft fill —
the default non-primary look), `outline` (transparent + 1px inset ring, no
fill/shadow), `ghost`, `floating` (a control loose from any surface — opaque
popover fill + `shadow-md`, hover lifts the glyph only), `link`, `text`
(boxless quiet inline — "Cancel", "Clear"), `textStrong` (bold underlined
inline affordance — "Change", "Open logs").
`grip` is the quiet, fill-free drawer handle; pair it with size `grip` for a
48×16 hit area around a small horizontal ridge.

**Sizes:** `default`, `xs`, `sm`, `lg`, `inline` (flush, zero box — for buttons
that sit inside a heading/sentence; replaces `h-auto px-0 py-0`), `micro`
(status-stack/table-footers), and the icon family `icon` / `icon-xs` /
`icon-sm` / `icon-lg` / `icon-titlebar`.
`touch` provides a 44px minimum target for compact mobile navigation.

**Tooltips only when hover teaches something new.** `<Tip>` is for discovery,
not a tax on every icon. Ask: does hover reveal something the user cannot
already see or infer? If not, skip the tip; keep an `aria-label` for a11y.

Tip unlabeled chrome when the job (or a keybind / truncated path / host /
other detail) is not already on screen — toolbar / titlebar / statusbar icons,
`TipKeybindLabel` shortcuts, ownership chips, unlabeled icon grids.

Do **not** tip:

- Menu triggers (kebabs / ⋯ / `ActionsMenu` / `DropdownMenuTrigger`) — the
  affordance is "open menu"; verbs live in the menu. Never tip
  `"Actions for ${row title}"` / `"Project actions"` / `"Actions"`.
- Close / dismiss X buttons — the glyph is the label (`aria-label` only).
- Controls whose visible label already says what the tip would ("click to…",
  paraphrases of the same words, timer labels restating "Running").

Never use native HTML `title=` on buttons — unstyled, ~500ms OS delay, clashes
with the themed `Tip`. `src/components/ui/__tests__/no-native-title.test.ts`
fails on any `<button>` / `<Button>` that still carries `title=`.

**Tooltip timing.** A hover is not a click — the cursor crosses triggers on
the way somewhere else. `Tip` waits 200ms before the first open so a sweep
does not flash a trail. After a tip has opened the page is warm: the next
trigger within 300ms opens instantly. The cooldown starts on close, so a
hover a second later waits again. Once triggered, entrance has no animation.
Exit fades over 100ms and moves 0.125rem toward the anchor; reduced motion
disables the exit animation. `OverflowTip` stays on its own longer delay
(list titles must not trail while scanning). Bubbles use a 0.25rem radius.

**Tooltip placement.** Choose intent through `placement`: `control` above (default), `toolbar` below, `row` to the right, and `left-rail` / `right-rail` inward. Explicit `side` and `align` override the preference. Radix flips and shifts for collisions, keeps the arrow attached, and hides detached triggers. Controls and toolbars use their owning pane as a boundary; row descriptions and rails may extend into the window. Use `boundary="viewport"` for an intentional escape. Short labels size to content; descriptions wrap within 24rem and the available space, in one rounded bubble.

**Slash descriptions.** Keep autocomplete rows single-line and ellipsized, but reveal the complete catalog description to the right of the hovered row. Use the shared bounded tooltip, collision padding, and word wrapping; it must not intercept row selection. Catalog and completion producers preserve the full author-supplied description.

**Model search.** Model filters and their highlighted labels treat hyphens, dots, underscores and spaces equivalently. Preserve original label spelling inside marks. The shared highlighter remains literal for other surfaces such as the command palette; model callers explicitly opt in. Model identifier search does not use dictionary spellcheck.

**Keybind hints in tooltips.** On a tipped button bound to a rebindable hotkey,
use `<TipKeybindLabel actionId="..." />` — it reads the i18n label and the
current combo from `$bindings`. Pass `text={...}` only when the label is
context-dependent (e.g. "Show" / "Hide"). Never hardcode combos; always use
`useKeybindHint` or `TipKeybindLabel`.

Notes:

- Text buttons are square (no radius) and sized by padding + line-height (no
  fixed heights). Only icon buttons carry the shared 4px radius.
- SVGs inherit `size-3.5` (`size-3` at `xs`). Don't re-set icon size.
- Polymorph with `asChild` when the button must render as a link/Slot.

## Badges — one component

`src/components/ui/badge.tsx`. Variants: `default` (tinted primary), `muted`,
`warn`, `destructive`, `outline`, `solid` (primary fill — icon-corner counts).
Sizes: `default`, `xs`, `overlay` (titlebar glyph counts).

## Profile rail

The profile rail keeps its create/import actions outside the scrolling squares.
Clipped horizontal edges reuse `edgeMask(edges, 'x')` from `fade-scroll.tsx`;
the default axis remains vertical for `FadeScroll`. Fitting content is unmasked,
and profile drag gestures temporarily remove the mask so the dragged square stays legible.

## Context-sensitive dialogs

Sudo password dialogs keep the backdrop unblurred (`DialogContent`'s
`blurBackdrop={false}`) and show the complete, selectable command before the
password field. Long commands wrap and scroll; missing backend context is
explicit, never inferred from another tool row. Other dialogs retain the shared
blurred backdrop.

## Form controls

- **`controlVariants`** (`src/components/ui/control.ts`) is the shared shape for
  `Input` / `Textarea` / `SelectTrigger`. New text-entry controls compose it.
- **`SearchField`** — borderless, underline-on-focus, auto-width. The only
  search input. Don't build boxed search bars; don't wrap it in a bordered tile.
  Empty lists hide their search field.
- **`SegmentedControl`** — the choice control for small mutually-exclusive sets
  (color mode, tool-call display, usage period). Replaces radio piles and
  pill rows.
- **`Switch`** (`size="xs"`) — bare, with `aria-label`. No bordered text wrapper.
- **`FanMenu`** (`src/components/ui/fan-menu.tsx`) — one hub control that
  fans sibling toggles out on hover: `direction` `vertical` | `horizontal`
  (split around the hub) | `arc`. Discs are `Button` `floating` off /
  `default` on; tips face outside the fan according to its geometry. Use it where a row of rarely
  touched toggles is costing input width (the composer's voice controls).

## Layout

In the browser-hosted Desktop, composer controls retain 44px targets on
coarse-pointer surfaces. Its collapse stages budget the actual control-size
token; below the two-target width budget, the context menu and Send occupy
separate rows rather than clipping or shrinking. Native Desktop keeps the
compact stages sized for 24px controls.
The voice engine split-button uses `--composer-voice-menu-size`: compact on
desktop, matching `--composer-control-size` on touch surfaces.
Narrow browser touch layouts show the current session tab with an open-tab
count. Its menu exposes every tab, while the new-tab action remains beside it.
In the browser-hosted Desktop, outside taps dismiss narrow sidebar overlays;
cancelled or scrolling gestures do not. Native Desktop keeps its overlay until
Escape, the toggle or hover-out. Touch transcript gestures select a pane without
focusing its composer.

- **Gutters:** `PAGE_INSET_X` (`src/app/layout-constants.ts`) for page side
  padding; `PAGE_INSET_NEG_X` to bleed a child to the edge. Don't hardcode
  `px-6`/`px-8` on pages.
- **Master/detail overlays:** `OverlaySplitLayout` + `OverlaySidebar` /
  `OverlayMain`. Cron, profiles, etc. ride this — don't rebuild a titlebar
  shell.
- **Settings subpages:** `OverlayNav` keeps navigation and disclosure separate:
  labels navigate; the shared `DisclosureCaret` button opens a branch without
  changing the page. Active paths reveal automatically, inactive paths stay
  folded unless manually opened. General comes first wherever present; parent
  labels and parent URLs open the first ordered subpage, never an overview or
  the last visited child. Explicit child links retain their destination;
  every parent and child uses the same Settings breadcrumb, without a duplicate
  icon-and-title heading. Page-level `SectionHeading page` retains actions and
  counts under breadcrumb-owned chrome; embedded callers keep their headings.
  Narrow windows keep every destination available in the shared navigation dropdown.
  Search and saved field links resolve to the owning child before highlighting.
- **Rows:** `ListRow` (settings `primitives.tsx`) for label/description/action
  rows. Flat, flush-left; no per-row indentation that fights flush headers.
- **No dividers between rows** unless the list genuinely needs them; prefer
  spacing. When you do need one, it's a single `--ui-stroke-tertiary` hairline.

## Panel titlebars

Top-edge panels extend into the native titlebar band. Their tab strips remain
inside their own zones so tab drops, focus, and split boundaries use the same
geometry. Panels without room beside the measured window controls place their
tabs on a full-width row below the controls. Minimized row groups use vertical
restore rails, including groups with multiple tabs. Sidebar buttons and shortcuts
restore minimized or fully hidden side groups without changing the selected tab.
Lower panels keep local headers. Empty header space moves the window;
tabs and actions remain no-drag, with native-control space reserved from the
existing traffic-light and Window Controls Overlay measurements.

On macOS, native traffic lights use a 12-point left inset in every titled window.
Keep the renderer's startup fallback insets aligned with that position so the
adjacent toolbar and panel reservations do not jump when native state arrives.
Windows/Linux retain their previous toolbar startup fallback until native state arrives.

The left cluster shows sidebar, settings, layout editor, and HUD controls. Flip
and the right-sidebar toggle sit on the right; haptics remain in settings.
In Simple interface mode only sidebar, settings and the layout editor render,
and the reserved cluster width shrinks with them (`TITLEBAR_FIXED_TOOLS` is the
one table both the buttons and the width reservation read).
Holding Cmd (Ctrl off macOS) reveals small slot numbers over the target strip's
status dots after 400ms, without changing tab widths. Hints follow the same
binding and hovered/focused-zone resolver as the number shortcuts.

Tab close buttons fade the label with a content mask, not a painted gradient.
The tab reads its surface token directly so glass tint is painted only once.
Close glyphs keep a 24px hit area, including session tabs wrapped in context menus.
`PaneTab` owns its `data-slot` marker after forwarded props so composition cannot
disable that width or the label mask.

Sticky user messages clip covered scrolling content, including the gap above
them. Their wrappers stay unpainted; only the rounded user bubble owns a fill.
Clipping follows the pinned prompt and its live height without changing layout,
so glass and message-bubble transparency do not reveal scrolling text.

## Feedback & empty/error/loading states

- **Loading:** `Loader` (`src/components/ui/loader.tsx`) — animated math/ascii
  curves (`lemniscate-bloom` for long ops). Never ship the literal text
  "Loading…".
- **Errors:** `ErrorState` + the canonical `ErrorIcon` (no bg chip). One look
  for the React boundary, in-dialog errors, and the boot-failure banner. Pass
  nodes for title/description so Radix `DialogTitle`/`Description` can flow
  through for a11y.
- **Logs:** `LogView` — no bg, hairline border, tight padding, small mono.
  Every place we surface raw logs uses it.
- **Empty:** `EmptyState` for plain page bodies; `PanelEmpty` for overlay
  master/detail empties with an icon and action. Don't hand-roll a third
  centered empty.
- **Confirmation:** `ConfirmDialog` is the only way we ask "are you sure". It
  opens focused on Confirm, so `Enter` confirms and `Esc` cancels, and it owns
  the pending → done → close beat and the inline error — a call site passes an
  async `onConfirm` and nothing else. A third way out (e.g. "Remove from
  sidebar" beside "Delete worktree") goes in the one `secondaryAction` slot.
  Never `window.confirm`: it's an unstyled blocking Chromium modal. A handler
  that wants the answer inline instead of a mounted dialog calls `confirm()`
  from `src/store/confirm.ts`, which renders this same primitive through the
  single `ConfirmHost` at the shell — the way `notify()` backs notifications.

## Chat typography

Appearance → Typography keeps **UI Scale** as whole-window zoom (100% by
default). **Chat Text Size** is a separate desktop-local multiplier (110% by
default) on conversation text and the composer editor, including floating and
inline-edit composers.
It does not resize the sidebar, settings, toolbars, media, or pane geometry.
Conversation size and line-height tokens are derived inside the transcript/editor
from their root base tokens; do not multiply the global tokens or nest CSS zoom.

## Chat, tools & boot surfaces

- The transcript and composer are built on `@assistant-ui/react`. Extend the
  existing components under `src/components/assistant-ui` and
  `src/app/chat/composer`; do not fork a second markdown, message, tool-call, or
  approval renderer for one feature.
- **Inline widgets** — a tool result that renders as a panel the user reads or
  acts on (clarify, artifact card) wears `WIDGET_SHELL_CLASS`
  (`src/components/chat/widget-shell.ts`): shared radius, the
  `--ui-widget-surface-background` fill, no border. Its actions sit _outside_
  the panel, below it. Don't give one widget its own radius or fill.
- Bordered surfaces in the transcript (tables, fences, callouts, attachments)
  use `--ui-stroke-tertiary`. Not `border-border` — that's the app-wide
  default and reads too hot against the thread.
- Interactive directive chips in the composer expose their action on hover.
  The action stays visible for a 500ms grace period while the pointer crosses
  from the chip to the floating pill; leaving both dismisses it.
- A tool result may expose an inline action that opens a preview. It must not
  open the rail automatically.
- Tool rows reserve destructive red for explicit failures. Missing read paths and
  ambiguous exit-1 results use neutral notices, with details still available.
  Errors described inside returned data are not tool failures. Expanded failures
  show the actual explanation; supporting output keeps its normal text color.
- Nested transcript scrollers keep their height caps and hand vertical scrolling
  back to the thread at either edge (`overscroll-behavior-y: auto`), even when
  their content fits. Only the outer thread contains vertical overscroll;
  horizontal code/output boundaries may remain contained. Thinking previews
  follow new tokens only while near the bottom, preserving the user's reading
  position until they scroll back down.
- Composer status groups start collapsed except todos. Progress updates and queue
  pause/resume preserve the user's disclosure choice. Error banners meet the
  stack's top edge without a blank padding strip. File and preview links remain
  visible at the bottom of the stack, below the queue and all status groups.
  A centered ridge on the composer's top edge hides/reveals the entire stack,
  including the git row, with a short downward/upward drawer slide. Its choice
  persists per conversation and owner, not globally. Hidden sections stay
  mounted but inert so their disclosure choices survive; reduced motion is instant.
- Popping out a composer makes it the window's only visible composer. It keeps
  its viewport placement while hover or keyboard focus selects a chat pane;
  moving back into the editor retains that recipient. Drafts, attachments and
  queues stay session-owned. Docking restores the individual pane composers.
  In either placement, moving into a chat pane gives its editor typing focus
  immediately and preserves its caret. Layout-only hover events and delayed
  focus callbacks cannot replace that choice, and a live transcript selection
  is never cleared by focus-follow. Movement within the same pane
  must not flush React; deliberate Tab navigation and clicked controls still work.
  An inline message edit is a typing target of its own: opening one keeps focus
  in the edit editor, and neither its mount-time focus nor mouse movement while
  it is open hands the caret back to the pane composer.
  Active dictation or voice conversation pins the recipient until capture ends,
  keeping the microphone's stop controls and shortcut attached to its owner.
- Status-stack rows use `StatusRow` with a leading `dismiss` action, a state
  icon and optional trailing actions. `StatusDismissButton` owns the Codicon
  close button for previews, background tasks and queued prompts; do not swap
  it for a trash icon or a CSS glyph. Icons and controls align to the first text
  line, including messages with attachment metadata.
- `status-stack.css` owns the shared columns and `0.25rem` nesting step. Rows
  own their padding and full-width hover fill. `StatusControlRow` uses the same
  columns for goal/loop/heartbeat details; `StatusPendingIcon` supplies the
  dashed marker for tasks and criteria. The first row keeps its normal padding;
  the stack adds no extra top inset.
- Keep the rounded status card stationary, with the bounded scroll viewport
  inside it. The outer scroll boundary uses `overscroll-behavior-y: contain`;
  nested rosters and transcripts use `auto` so wheel input can hand off at an
  edge without trapping it or scrolling the chat behind the stack.
- Install, onboarding, connecting, boot failure, and reauthentication are
  distinct states with shared visual primitives. Preserve their recovery
  semantics when unifying appearance.
- Guide startup has a dedicated landing screen with the shared long-operation
  Loader and a localized startup label. It remains active until the guide's
  saved conversation is loaded. No greeting, composer, Skip setup, or statusbar
  appears early. Reduced motion uses a static BrandMark and the same label.
  On readiness, reveal the real conversation and fade controls in over 100ms.
  Never fabricate progress, delay readiness for motion, or replace a resumed
  conversation with a new greeting.
- Respect `AppShell` overlay ownership. Persistent terminal/content layers,
  route overlays, dialogs, and boot surfaces must not compete through ad-hoc
  z-index literals. Pick a rung of the ladder in `styles.css` instead —
  `--z-modal-backdrop` / `--z-modal` / `--z-modal-popover`, `--z-over-modal`
  (toasts, tooltips, command surfaces) and `--z-over-modal-content`,
  `--z-switcher-backdrop` / `--z-switcher`, then the boot chain
  `--z-connecting` → `--z-onboarding` → `--z-setup` → `--z-crash`. Plain
  `z-10`/`z-20` are still right for stacking _within_ one component.

## Iconography & brand

- **Tabler** is the default component/chrome set. Import its curated aliases and
  `iconSize` scale from `src/lib/icons.ts`; do not import icon packages directly
  in feature code.
- **`Codicon`** is the compact editor/tool/status vocabulary. Use
  `src/components/ui/codicon.tsx`, including `codiconIcon()` where a
  Tabler-shaped component is required.
- Pick the vocabulary by semantic context and reuse the existing icon for an
  action. Do not introduce a third icon set or mix styles within one control
  group.
- **`BrandMark`** (`src/components/brand-mark.tsx`) is the brand glyph — the
  `nous-girl` mark on a fixed tile, theme-aware: black girl on white in light
  mode, white girl on the `#0d1117` dark tile in dark mode (keyed off
  `renderedMode` so bright surfaces in dark skins stay on the light mark).
  The marks are generated by `scripts/generate_icons.py` from the girl SVGs
  - squircle backgrounds (`assets/nous-girl-*.svg` × `assets/backgrounds/`).
    It replaced scattered Sparkles glyphs in updates / onboarding / about. Use it
    for hero/brand moments; don't reintroduce decorative star/sparkle icons.

## Motion

- Visible windows keep animating when another app takes focus. Hidden/minimized
  windows and inactive panes may pause; background polling stays focus-gated.
- Animated integer counts reuse `AnimatedInt` in `src/components/ui/diff-count.tsx`.
  Its spring updates the DOM directly without per-frame React renders.
- Quick, functional transitions (~100ms on controls). Respect
  `prefers-reduced-motion` for anything beyond a fade.
- Hover paints instantly; only the return eases. Put the transition on the
  resting state and drop it while hovered (`transition-colors duration-100
hover:transition-none`, or the CSS equivalent on `:hover`). A group that
  dims siblings goes instant too. Exceptions are named, e.g. the catalog
  card's arc border fades in and out.
- Choreographed exits (e.g. onboarding's "matrix" fade-down) stagger per-element
  then settle the surface — the outer container's fade is _delayed_ so it
  doesn't swallow the inner animation. Don't let a global fade race the detail.
- Motion follows state; it never delays state. Selection, drag targets, cancel,
  and pressed feedback paint in the current frame.
- Do not animate layout geometry with `transition-all` on a hot interaction.
  Name the properties, avoid backdrop-filter repaints during movement, and
  remove animation before masking a performance problem.

## Direct manipulation & performance

The app should feel instant under real load — long transcripts, several panes,
live streams. Design toward that:

- Direct manipulation paints first; persistence reconciles after and rolls back
  visibly on failure.
- Keep interaction feedback cheap: hot-path state stays local or narrowly
  derived, not wired into heavy trees; pointer work coalesces per frame.
- One drop region has one visual owner, and drop targets speak one affordance
  language across files, sessions, tabs, and panes. Overlapping targets resolve
  to the active one instead of stacking overlays.
- Forgiving geometry beats pixel-perfect triggers; edge actions live near their
  edge, not clustered in the center.
- Expensive stateful surfaces stay mounted when hidden. Visibility is not
  lifecycle.

Prove speed with realistic content. A fast empty-state demo says nothing about a
long transcript or a busy terminal.

## Keyboard & cancellation

- Keyboard ownership follows focus. The focused surface wins its keys; shell
  shortcuts must not steal a terminal's or editor's bindings.
- Zone tabs have one keyboard entry point at the selected tab. Arrows follow
  the strip's orientation; Home/End move focus, Enter/Space activate, and Tab
  leaves the tabs. Delete invokes an available close action and retains focus
  on a surviving sibling. Restoring a vertical rail retains its tab focus.
  The collapsed rail keeps the zone context menu on its outer surface; its
  inner tab list owns vertical keyboard navigation.
- Tab-header focus belongs to navigation: session activation and queued composer
  focus retries must not redirect it. Explicit composer focus still works.
  Entering a split's tab strip selects its visible chat as the typing recipient
  without moving focus into the editor; hidden tabs must not receive that input.
- Sidebar keyboard stops are real controls: session title, reorder handle and
  actions, not timestamps. Focus uses existing theme tokens, stays distinct from
  selection, and reveals hover-only controls. Focused/open actions replace their
  metadata tail; real metadata actions remain independently reachable.
- Focusing the Sessions sidebar preserves the last active chat's visual emphasis.
  Dimming still distinguishes session panes; sidebar navigation must not desaturate
  the chat or transfer its active highlight to a hidden primary tab.
- Focused and hovered chat panes both retain full color and opacity. Only panes
  that are neither focused nor hovered recede, with 20% desaturation.
- Register global shortcuts through the shared layer, not ad-hoc listeners.
- One cancel gesture does one thing: cancel the active interaction, or close the
  topmost dismissable surface — never both, never the control underneath.
- Cancellation is synchronous in the UI even if cleanup is async: overlays,
  cursors, and pending gesture state clear at once.
- Flows that deliberately cannot be dismissed (install/onboarding, destructive
  confirmation) must make that explicit.

## i18n

- Every user-facing string goes through `useI18n()` (`src/i18n/context.tsx`).
  No literals in JSX.
- **Update all locales together** — every catalog registered in
  `src/i18n/catalog.ts`. A string change in `en.ts` that skips the others is a
  regression (drifted punctuation, stale labels). Keep trailing-punctuation and
  tone consistent across all of them. `fr`, `de`, and `es` are complete
  `Translations` objects, so a key missing there fails the type check; the
  `defineLocale()` overlays fall back to English instead.
- **Sparse locales** (`ar`, `ru`) override the English base through
  `defineLocale()`. Large catalogs are split by topic: the Arabic source lives
  in `src/i18n/ar_<topic>.ts`, recomposed by `src/i18n/ar.ts`.

## State (TypeScript)

The detailed state contract lives in the scoped
[`AGENTS.md`](./AGENTS.md). Visual code follows these essentials:

- Shared/cross-component state → small **nanostores**, not prop-drilling.
  Each feature owns its atoms; shared atoms live in `src/store`.
- Rendering components subscribe with `useStore`; non-render actions read with
  `$atom.get()`.
- Subscribe to derived coarse facts instead of high-frequency source atoms when
  the component does not render the full value.
- Colocated action modules over god hooks. A hook owns one narrow job.
- Keep persistence beside the atom that owns it. Route roots stay thin.
- Prefer `interface` for public props; extend React primitives
  (`React.ComponentProps<'button'>`, `Omit<…>`).

## Affordances

- `cursor-pointer` at the primitive level (Button, dropdown/select) — don't
  hardcode it per call site.
- Keep keyboard focus visible through existing control primitives and theme
  tokens; never globally suppress outlines or focus rings. Preserve native
  focus indicators on controls without a themed replacement. Titlebar actions
  have no active-background state.
- Each control has one focus treatment: its existing ring, outline, or field
  border. Suppress the native outline only when that replacement is visible.
  In forced-colors mode, use one system-color outline on the focused control
  (or its field wrapper); don't outline unfocused menu rows or containers.
  Keep command-picker input and selected-row fallback outlines inset so scroll
  and group clipping cannot cut off their edges.
  Inline message editors delegate focus chrome to their card border, just like
  the main composer; their inner contenteditable must not add a second outline.
- Keyboard users can activate image viewers and return to their available
  opener when the viewer closes. Composer focus-follow must preserve focus
  returning from a floating layer while updating the owning chat surface.
- `Esc` closes every dismissable overlay/dialog (install/onboarding excluded);
  close is an x-icon, not the word "Close".

## Before you add something — checklist

- [ ] Reuse a primitive (`Button`, `SearchField`, `SegmentedControl`,
      `ListRow`, `Loader`, `ErrorState`, `LogView`, `ConfirmDialog`) instead of
      forking one?
- [ ] Tokens (`--ui-*`, `shadow-nous`, `--stroke-nous`) — zero raw colors /
      one-off shadows?
- [ ] No `className` overriding a primitive's padding / size / radius / chrome?
- [ ] Tips only where hover teaches something new (no kebab / menu-trigger
      tips; unlabeled chrome that needs discovery gets `<Tip>` + `aria-label`)?
- [ ] No native `title=` on buttons?
- [ ] Keybind hints on tipped buttons use `useKeybindHint` / `TipKeybindLabel`?
- [ ] Overlay uses `shadow-nous` + `border-(--stroke-nous)`, no hard border?
- [ ] Flat — no card-in-card, no gratuitous row dividers?
- [ ] No automatic navigation, focus steal, or pane opening from background
      events?
- [ ] Direct manipulation paints immediately and rolls back cleanly on failure?
- [ ] Hot interactions avoid broad subscriptions, layout thrash, and
      `transition-all`?
- [ ] Keyboard ownership and single-action `Esc` behavior are correct?
- [ ] All registered locales updated for any new/changed string?
- [ ] `cursor-pointer`, focus ring, and `Esc`-to-close behave?
- [ ] Touched a primitive, token, or variant? Its named-contract entry in this
      file is updated in the same change.
