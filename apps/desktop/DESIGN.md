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
overlays render outside the shell: rails, the panel width and the page columns are the layout
tokens in `tokens/space.css`. Chat uses the wide layout described below. Native titlebar sizing,
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
(`app/redesign/portal-shell.css`; every colour, radius and size comes from the tokens in
"Portal design language" below). The titlebar band and the 60px rails show the canvas; zones
touching the window top become panels starting below the band, and only the row's outer top
corners round. The wing artwork is painted once, on the canvas, and reads only through
translucent panels; page nav panels carry their own viewport-registered wing layer
(`--portal-page-panel-fill`). Window Glass (macOS default) never switches the redesign off: its
rules are unlayered and beat the `@layer base` Glass rules, and only the canvas (rails, band) takes
the native material, tinted with the portal canvas at the user's Glass intensity. Titlebar, rail and
pane header buttons are chrome buttons (`--portal-surface-chrome*`); the open destination or shown
pane wears the brand selection with a brand ring. The titlebar band is a shell property: 60px in the
redesign (Figma's header; content starts at y 60), 34px classic (44px browser
touch). Electron reads the shell choice from the shared UI preferences before
creating a window and re-seats the macOS traffic lights (centred, in the rail
column) and the Windows/Linux caption overlay when it flips
(`electron/window-chrome.ts`); the popped-out Browser keeps the classic band.
The band carries the leading cluster past the rail (Back and Forward through the window's own router
history on every screen, `app/shell/history-position.ts`; a page with nothing behind it closes to the
chat), the open session's title pill with "+" at the stage's leading edge, and the panel toggles at
the end. Pages cover the chat, they do not replace it (`contrib/route-cover.tsx`): a route overlay or
workspace page paints over a chat that stays mounted, skipped by style, layout and paint
(`content-visibility: hidden`), inert and aria-hidden, so returning to Chats is a flip with its
scroll, draft and focus intact. Toasts of every kind stack at the stage's end corner, resting on a
docked composer, clear of page headers. Page content columns start at Figma's left insets
(`--portal-frame-inset` past a sidebar panel, `--portal-stage-inset` without one), and Schedules lists its
schedules in the shared page sidebar panel like Agents.
Workspace pages (Capabilities, Library, …) share the overlay pages' square,
full-bleed frame. Without an explicit skin pick, the redesign paints the
Hermes Portal skin and classic paints Nous (`defaultSkinName`); flipping the
shell never writes a pick.

Redesigned surfaces use neutral foreground-derived fills and strokes, not
accent-tinted greys (the colour roles below). Global settings cards and agent settings cards still
differ (see "To unify in Phase B", U1). Settings-navigation selection is the neutral selection role;
sub-pages reveal under the active row, without per-row carets. Settings pages compose
`SettingsOverview` / `SettingsCard` / `SettingsDisclosure` (`settings-card.tsx`), which fall back to
the classic wrapper. Brand-blue selection remains on the global rail. Agent-management navigation
tiles stay translucent; agent rows and the overview header use the Figma robot with the real
owner's color.
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
guide. An agent's card lists its three most recent chats (plus the open one and any running or unread)
over a "Show all (N)" / "Show less" row remembered per agent; search lists every match. The open
machine's header bar sticks to the top of the list while its card scrolls under it. Hover is a 4%
fill (2% on agents), the open pinned chat a clearly stronger 7% pill. Row and
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
The Hermes Portal skin supplies only the seeds the roles read (`--ui-base`, `--ui-bg-chrome`, the
floating `--ui-bg-elevated`, the canvas `--portal-skin-window`: navy `#0a0a1a` / `#07071a` /
`#0d0d1e` on dark); the shell's panels derive from them. Custom theme choices and classic styling
stay available.
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
(setup flow, metrics, session scope, the plugin SDK). Brand marks, floating
surfaces (`app/redesign/portal-floating.css`) and the control scale (`portal-controls.css`, 36px
inside pages and dialogs) belong to the redesign, not to the Hermes Portal theme, which carries
palette and typography only.
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
Every routed page keeps one page frame (`tokens/space.css`): beside a sidebar
panel the content column starts `--portal-frame-inset` (20px, the freshest frames 1036:7189 /
895:*) after the panel's divider, on Agents, Settings and Schedules alike; without a panel it
starts `--portal-stage-inset` (Figma 919: x 150 at 1280) into the stage. Columns are
left-anchored at every window width, never centred, with one max width per page kind:
`--portal-column-form` (840) for forms and details, `--portal-column-catalog` (980) for
catalogs, lists and tables. Every H1 (or agent hero) tops out at the same y; header actions end
at the column's right edge.
An agent's page has one vertical menu (Figma 1036:7189). On Overview and History the
page sidebar panel lists the agents; on the Settings tab the same panel swaps to
"← Back to agents" (a history step back to the Overview), the agent's own row, and the
settings sections (10px group labels, 32px rows, counts at the row end), while the stage
holds the agent header, the tabs and the selected page at full width. When the panel is
folded away (⌘B) or the window is too narrow for panels, the sections menu leads the
page in the stage instead, so every section stays reachable.

Transcript styling is isolated in `components/assistant-ui/thread/redesign.css`,
following the Figma conversation (`491:40346`): one centred column shared with the
docked composer (`--portal-column-conversation`, read as `--portal-conversation-*` in `app/chat/stage.css`), prose
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

## Portal design language (the redesign)

The redesign (`<html data-shell="portal">`) is one small design language. Its
source is the Figma file `EuTlgG8nMQo02z5RlImIu9`, newest frame wins: 1036:7189
(Settings · General) and 1038:7284 (History preview), then agents 895/877,
settings 856, capabilities 919, the chat sidebar's Variation 1 (147:7470) and
the session shell 906:19396; Brand rev 491 supplies conversation content and
Light. Every value lives in one folder, `src/app/redesign/tokens/`, imported once
by `main.tsx` right after `styles.css` (the order matters: the root-gated tokens
tie `styles.css`'s own `:root.dark` rules on specificity and must come later).

1. **Navy, flat, layered.** One navy family: canvas → translucent panels → stage
   → cards. Depth comes from 2–5% ink layers and hairlines, not shadows; only
   floating things (menus, popovers, dialogs, toasts, the composer) cast one.
2. **One brand gesture.** `#0000F2` fills primary actions, toggles, checked
   states and the active tab underline; everything else is neutral ink. Thin
   blue (rings, links, counts) uses the lighter brand ink on dark.
3. **Display for names, sans for everything else.** Oswald Light only for H1,
   panel/dialog titles and the wordmark; Mona Sans for UI; mono for paths, ids
   and code (OFL stand-ins for Figma's Rules Compressed / Aeonik Fono Pro).
4. **A small, fixed type scale and an AA ink ladder.** Nine sizes, four text inks
   that all pass 4.5:1 on every surface.
5. **Radius by role.** 2 chip, 3 control, 4 card, 6 chat surface, 8 floating,
   9 panel corner.
6. **Rows, not boxes.** Grouped rows in one card with hairline dividers; no card
   in a card; sections separated by space, not rules.
7. **Instant navigation, quiet motion.** Navigation and selection never animate;
   hover paints at once and eases back (120ms); disclosures and popovers 180ms;
   drawers 240ms; nothing moves under reduced motion.
8. **States are derived, not drawn.** Figma draws rest and selected. Hover and
   pressed are the next ink steps, focus is one 2px brand-ink ring, disabled is
   40% opacity — the same rule for every component.

### Responsive (compact, medium, wide)

One renderer serves the desktop window and phones (the browser host), so the language has three
width classes and two breakpoints, defined once in `src/app/redesign/breakpoints.ts` (`$widthClass`,
`widthClassFor`) and documented as `--portal-breakpoint-medium` / `--portal-breakpoint-wide` in
`tokens/space.css`. A custom property cannot feed a media query, so every redesign query repeats the
literal (`max-width: 639.98px` compact, `max-width: 1023.98px` below wide) with a comment naming the
class. Classic's sidebar dock point is the medium breakpoint; classic is otherwise untouched.

- **Wide (≥ 1024)**: the desktop layout above, pixel for pixel.
- **Medium (640–1023)**: the rail stays; the chat sidebar and page panels leave the layout and open
  as **sheets** from the band's sidebar button over a full-width stage; page columns start at the
  panel inset.
- **Compact (< 640)**: no rail column (`--portal-rail-reserve: 0`). The band is the menu, the title
  and "+" (no Back/Forward: the browser has them), 48px in the browser (44px on touch; native windows
  keep 60, where Electron seats its controls). The menu opens the desktop's left side as one sheet:
  the rail beside the chat sidebar or the page's panel, or the rail alone on a page without one.
  Page insets collapse to 16px; the developer rail has no compact form.
- **Sheets** (`app/shell/shell-sheet.ts`): full height past the rail, `--portal-sheet-width`
  (`min(20rem, 100vw − 3.5rem)`), opaque, over `--portal-surface-scrim`. A tap on the scrim, Esc, or
  any pick (navigation) closes one; focus moves into it and returns to the menu. Sheets sit under
  menus and popovers (`--portal-layer-sheet*`), so a row's menu opens over its sheet.
- **Short (max-height 30rem, `SHORT_QUERY`)**: a landscape phone or a raised keyboard puts the
  composer on one row (attach, editor, send), hides the model row until there is room, drops the
  hero art, and takes the browser's compact band (48, touch 44).
- **Touch** (`(hover: none), (pointer: coarse)` on the browser host, any width): chrome buttons,
  rows, menu items, tabs and page buttons take `--portal-touch-target` (44px); switches and
  checkboxes keep their visual and extend their hit area; text fields never go under
  `--portal-type-field-touch-min` (16px, iOS zoom); hover-only affordances are shown.
- **Floating surfaces** stay inside the window (Radix's available size less a 16px gutter); on
  compact a dialog is the whole window with its footer on the floor, and toasts span the window above
  the composer.
- **Browser chrome** (`app/redesign/browser-chrome.ts`): `theme-color` follows the canvas;
  `viewport-fit=cover` lays the page under the notch, and on a device that reports a safe area the
  app root is inset by it as a whole (body-portalled dialogs read `--portal-safe-*`).

### Tokens

`raw.css` holds the only colour literals (primitives: brand, status, danger, the
ink ladder of `--ui-base` alphas). The other files hold roles — what a value is
*for* — and read the primitives. Base blocks are dark; `:not(.dark)` blocks hold
light; Glass changes only the canvas (one rule in `color.css`). `bridge.css`
maps the roles onto the names shared components already read (`--ui-text-*`,
`--ui-stroke-*`, `--ui-bg-card`, …), so classic primitives rendered inside the
redesign follow the language without forking. Names that predate the role names
remain as aliases (`--portal-brand-fill` → `--portal-surface-brand`, …) until
their readers move. Component CSS may keep local custom properties only as
aliases of tokens (`--portal-sidebar-hover: var(--portal-raw-ink-4)`).

The tables below are generated from the CSS; do not edit them by hand.

<!-- design-tokens:begin (generated by scripts/check_desktop_design_tokens.py --write-docs) -->

#### Primitives (`raw.css`): the only colour literals

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-raw-blue` | `#0000f2` | ″ | Figma primary/blue: fills of primary actions, toggles, checked states, the active tab underline. |
| `--portal-raw-blue-ink-dark` | `#8585ff` | ″ | The blue lightened to AA for thin marks (rings, links, counts) on navy. |
| `--portal-raw-white` | `#f2f2f2` | ″ | Figma primary/white: ink on the brand fill. |
| `--portal-raw-pure-white` | `#fff` | ″ |  |
| `--portal-raw-black` | `#000` | ″ |  |
| `--portal-raw-green` | `#00f279` | ″ | Figma secondary/green (live dots and gradients). |
| `--portal-raw-mint` | `#6df2a8` | ″ | Connected / online ink on dark (877:4308); its AA-darkened twin for light. |
| `--portal-raw-mint-deep` | `#1f8a65` | ″ |  |
| `--portal-raw-orange` | `#f27900` | ″ | Figma secondary/orange: pending and warning. |
| `--portal-raw-danger` | `#a22a20` | ″ | Danger fill and stroke (877:4212, 878:7066, 1038:7836) and its soft status text (877:4312). |
| `--portal-raw-danger-soft` | `#cc8a84` | ″ |  |
| `--portal-raw-spectrum-red` | `#aa1515` | ″ | The composer's spectrum edge (906:19872). |
| `--portal-raw-spectrum-yellow` | `#ddd61c` | ″ |  |
| `--portal-raw-shadow-4` | `rgb(0 0 0 / 0.04)` | ″ | Shadow inks. |
| `--portal-raw-shadow-8` | `rgb(0 0 0 / 0.08)` | ″ |  |
| `--portal-raw-shadow-20` | `rgb(0 0 0 / 0.2)` | ″ |  |
| `--portal-raw-shadow-25` | `rgb(0 0 0 / 0.25)` | ″ |  |
| `--portal-raw-shadow-50` | `rgb(0 0 0 / 0.5)` | ″ |  |
| `--portal-raw-ink-2` | `color-mix(in srgb, var(--ui-base) 2%, transparent)` | ″ | The ink ladder (LANGUAGE §1.1): Figma's light/2, 5, 10, 40 plus the page steps 3, 4, 8, 14, 22, 28. No other neutral alpha belongs to the language; the off-ladder steps still in use are listed in color.css and the Phase B table. |
| `--portal-raw-ink-3` | `color-mix(in srgb, var(--ui-base) 3%, transparent)` | ″ |  |
| `--portal-raw-ink-4` | `color-mix(in srgb, var(--ui-base) 4%, transparent)` | ″ |  |
| `--portal-raw-ink-5` | `color-mix(in srgb, var(--ui-base) 5%, transparent)` | ″ |  |
| `--portal-raw-ink-8` | `color-mix(in srgb, var(--ui-base) 8%, transparent)` | ″ |  |
| `--portal-raw-ink-10` | `color-mix(in srgb, var(--ui-base) 10%, transparent)` | ″ |  |
| `--portal-raw-ink-14` | `color-mix(in srgb, var(--ui-base) 14%, transparent)` | ″ |  |
| `--portal-raw-ink-22` | `color-mix(in srgb, var(--ui-base) 22%, transparent)` | ″ |  |
| `--portal-raw-ink-28` | `color-mix(in srgb, var(--ui-base) 28%, transparent)` | ″ |  |
| `--portal-raw-ink-40` | `color-mix(in srgb, var(--ui-base) 40%, transparent)` | ″ |  |
| `--portal-raw-brand-8` | `color-mix(in srgb, var(--portal-raw-blue) 8%, transparent)` | ″ | Brand tints: row selection 8, info banner 10, light brand selection 12, dark brand selection 25, info banner stroke 60. |
| `--portal-raw-brand-10` | `color-mix(in srgb, var(--portal-raw-blue) 10%, transparent)` | ″ |  |
| `--portal-raw-brand-12` | `color-mix(in srgb, var(--portal-raw-blue) 12%, transparent)` | ″ |  |
| `--portal-raw-brand-25` | `color-mix(in srgb, var(--portal-raw-blue) 25%, transparent)` | ″ |  |
| `--portal-raw-brand-60` | `color-mix(in srgb, var(--portal-raw-blue) 60%, transparent)` | ″ |  |

#### Colour roles (`color.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-surface-brand` | `var(--portal-raw-blue)` | ″ | Brand. The fill is the same in every theme and mode; thin rings, underlines and small blue text use the brand ink, which lightens on dark canvases for contrast. |
| `--portal-text-on-brand` | `var(--portal-raw-white)` | ″ |  |
| `--portal-text-brand` | `var(--portal-raw-blue-ink-dark)` | `var(--portal-raw-blue)` |  |
| `--portal-surface-brand-selected` | `var(--portal-raw-brand-25)` | `var(--portal-raw-brand-12)` | Brand selection: the selected segment, scope chip and cleanup choice (856:3444). |
| `--portal-spectrum-red` | `var(--portal-raw-spectrum-red)` | ″ |  |
| `--portal-spectrum-yellow` | `var(--portal-raw-spectrum-yellow)` | ″ |  |
| `--portal-text-primary` | `var(--ui-base)` | ″ | Text ladder. Figma steps ink 1 / .8 / .6 / .5 / .4 / .3, but .4 and .3 fail AA for the 12-13px meta and labels that use them, so the ladder's floor is the least ink that keeps 4.5:1 on every real surface (sidebar, stage, page canvas, cards, popovers, selected rows), and the rungs above are spaced by contrast, not alpha: light ≈ 16 / 8.9 / 6.7 / 5.1:1, dark ≈ 17 / 8.7 / 6.6 / 4.9:1. Dark ink on light canvases loses contrast faster, so light needs more alpha per rung. |
| `--portal-text-secondary` | `color-mix(in srgb, var(--ui-base) 70%, transparent)` | `color-mix(in srgb, var(--ui-base) 80%, transparent)` |  |
| `--portal-text-tertiary` | `color-mix(in srgb, var(--ui-base) 60%, transparent)` | `color-mix(in srgb, var(--ui-base) 72%, transparent)` |  |
| `--portal-text-quaternary` | `color-mix(in srgb, var(--ui-base) 50%, transparent)` | `color-mix(in srgb, var(--ui-base) 64%, transparent)` |  |
| `--portal-surface-canvas` | `var(--portal-skin-window, var(--ui-bg-chrome))` | `var(--portal-skin-window, color-mix(in srgb, var(--ui-base) 5%, var(--ui-bg-chrome)))` | Window canvas (rails, titlebar band): the skin's own canvas, else its chrome. |
| `--portal-surface-panel` | `color-mix(in srgb, var(--portal-surface-canvas) 90%, transparent)` | `color-mix(in srgb, var(--portal-raw-pure-white) 30%, transparent)` | Panels laid on the canvas (chat sidebar, page panels, side tool panels): #0A0A1A@.90 over the canvas (856:2740), so the wing reads through at a tenth of its strength. |
| `--portal-surface-stage` | `color-mix(in srgb, var(--ui-bg-chrome) 96%, var(--portal-surface-canvas))` | `color-mix(in srgb, var(--portal-raw-pure-white) 90%, var(--portal-surface-canvas))` | The stage: #07071A@.96 over the canvas (856:2809, Var1 147:7470), the skin's chrome. |
| `--portal-sidebar-background` | `var(--portal-surface-canvas)` | `color-mix(in srgb, var(--portal-raw-pure-white) 30%, var(--portal-surface-canvas))` | The chat sidebar's opaque fill (the shared sidebar surface). |
| `--portal-page-canvas` | `var(--ui-bg-chrome)` | `var(--portal-surface-canvas)` | Routed pages lay their body on the canvas (Overview 856:2809) and their nav on a panel. |
| `--portal-page-background` | `var(--portal-page-canvas)` | `var(--portal-surface-stage)` |  |
| `--portal-page-sidebar-background` | `var(--portal-surface-panel)` | `var(--portal-sidebar-background)` |  |
| `--portal-surface-elevated` | `var(--ui-bg-elevated)` | ″ | Menus, popovers, dialogs, drawers, toasts: the solid floating fill (878:6044 #0D0D1E on dark). |
| `--portal-surface-card` | `var(--portal-raw-ink-2)` | ″ | Cards of grouped rows (1036:7576, 1038:7803). |
| `--portal-surface-selected` | `var(--portal-raw-ink-5)` | ″ | Neutral selection: list rows, nav rows (1036:7409), with --portal-stroke-selected. |
| `--portal-surface-row-hover` | `var(--portal-raw-ink-3)` | ″ | A row under the pointer on a panel, the stage or a card: one ink step over its rest. |
| `--portal-surface-chrome` | `var(--portal-raw-ink-2)` | ″ | 30px chrome buttons and the session pill (906:19469): rest, hover, pressed. |
| `--portal-surface-chrome-hover` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-surface-chrome-pressed` | `var(--portal-raw-ink-10)` | ″ |  |
| `--portal-surface-control` | `var(--portal-raw-ink-5)` | `var(--portal-raw-ink-3)` | Secondary and icon buttons (877:4189) at rest, hover and pressed. |
| `--portal-surface-control-hover` | `var(--portal-raw-ink-8)` | `color-mix(in srgb, var(--ui-base) 6%, transparent)` |  |
| `--portal-surface-control-pressed` | `color-mix(in srgb, var(--ui-base) 11%, transparent)` | `color-mix(in srgb, var(--ui-base) 9%, transparent)` |  |
| `--portal-surface-field` | `var(--portal-raw-ink-4)` | `var(--portal-raw-ink-2)` | Inputs, search, segmented tracks (895:9670). |
| `--portal-surface-select` | `var(--portal-raw-ink-8)` | `var(--portal-raw-ink-4)` | Select triggers (856:3456). |
| `--portal-surface-floating-hover` | `color-mix(in srgb, var(--ui-base) 6%, transparent)` | `var(--portal-raw-ink-5)` | A menu item under the pointer, on the elevated surface. |
| `--portal-surface-scrim` | `var(--portal-raw-shadow-50)` | `var(--portal-raw-shadow-20)` | The modal scrim: a plain dim, never a blur. |
| `--portal-surface-danger-solid` | `var(--portal-raw-danger)` | ″ | The confirm of a destructive dialog (878:7066): a solid danger slab with white ink (7.3:1). |
| `--portal-text-on-danger` | `var(--portal-raw-pure-white)` | ″ |  |
| `--portal-stroke-hairline` | `var(--portal-raw-ink-5)` | ″ | Strokes. Card edges, the panel-to-stage edge, page footer rules (1038:7803). |
| `--portal-stroke-divider` | `var(--portal-raw-ink-3)` | ″ | Between rows inside a card (877:4229, 1038:7499). |
| `--portal-stroke-edge` | `var(--portal-raw-ink-10)` | ″ | The chat sidebar's edge and header rule (Var1 147:7470: light/10). |
| `--portal-stroke-control` | `var(--portal-raw-ink-10)` | `var(--portal-raw-ink-14)` |  |
| `--portal-stroke-field` | `var(--portal-raw-ink-14)` | `color-mix(in srgb, var(--ui-base) 18%, transparent)` |  |
| `--portal-stroke-strong` | `var(--portal-raw-ink-28)` | ″ | Selects, checkbox and radio at rest. |
| `--portal-stroke-selected` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-stroke-floating` | `var(--portal-raw-ink-10)` | `color-mix(in srgb, var(--ui-base) 12%, transparent)` |  |
| `--portal-floating-separator` | `color-mix(in srgb, var(--ui-base) 6%, transparent)` | ″ | Menu separators and the floating menu's item ink (Figma .9). |
| `--portal-floating-item-ink` | `color-mix(in srgb, var(--ui-base) 90%, transparent)` | ″ |  |
| `--portal-status-online` | `var(--portal-raw-mint)` | `var(--portal-raw-mint-deep)` | Status inks, one set for the whole redesign (sidebar machines, dots, chips, page status text): online is the pages' spec green (877:4308), pending Figma's orange, failed its soft red; each passes 4.5:1 on its mode's panels. |
| `--portal-status-pending` | `var(--portal-raw-orange)` | `color-mix(in srgb, var(--portal-raw-orange) 70%, var(--portal-raw-black))` |  |
| `--portal-status-error-ink` | `var(--portal-raw-danger-soft)` | `var(--portal-raw-danger)` |  |
| `--portal-focus-ring-color` | `var(--portal-text-brand)` | ″ | The one focus ring's colour (geometry in shape.css). |
| `--portal-brand-fill` | `var(--portal-surface-brand)` | ″ | Names in use before the role names; Phase B moves their readers to the role and deletes them. |
| `--portal-on-brand` | `var(--portal-text-on-brand)` | ″ |  |
| `--portal-brand-ink` | `var(--portal-text-brand)` | ″ |  |
| `--portal-selection-background` | `var(--portal-legacy-brand-selected)` | ″ |  |
| `--portal-neutral-selection-background` | `var(--portal-surface-selected)` | ″ |  |
| `--portal-neutral-selection-stroke` | `var(--portal-stroke-selected)` | ″ |  |
| `--portal-window-background` | `var(--portal-surface-canvas)` | ″ |  |
| `--portal-sidebar-panel-background` | `var(--portal-surface-panel)` | ″ |  |
| `--portal-panel-divider` | `var(--portal-stroke-hairline)` | ″ |  |
| `--portal-control-fill` | `var(--portal-surface-control)` | ″ |  |
| `--portal-control-fill-hover` | `var(--portal-surface-control-hover)` | ″ |  |
| `--portal-control-fill-pressed` | `var(--portal-surface-control-pressed)` | ″ |  |
| `--portal-control-stroke` | `var(--portal-stroke-control)` | ″ |  |
| `--portal-field-fill` | `var(--portal-surface-field)` | ″ |  |
| `--portal-field-stroke` | `var(--portal-stroke-field)` | ″ |  |
| `--portal-select-fill` | `var(--portal-surface-select)` | ″ |  |
| `--portal-select-stroke` | `var(--portal-stroke-strong)` | ″ |  |
| `--portal-control-danger-fill` | `var(--portal-surface-danger-solid)` | ″ |  |
| `--portal-control-on-danger` | `var(--portal-text-on-danger)` | ″ |  |
| `--portal-floating-surface` | `var(--portal-surface-elevated)` | ″ |  |
| `--portal-floating-stroke` | `var(--portal-stroke-floating)` | ″ |  |
| `--portal-floating-item-hover` | `var(--portal-surface-floating-hover)` | ″ |  |
| `--portal-legacy-card-fill-settings` | `var(--portal-raw-ink-5)` | ″ | Global settings cards (--ui-bg-card at the root): 5%; agent pages already read the card role. |
| `--portal-legacy-disclosure-fill` | `var(--portal-raw-ink-3)` | ″ | The global settings Advanced disclosure: its own fill and a control-strength stroke. |
| `--portal-legacy-disclosure-stroke` | `var(--portal-raw-ink-10)` | ″ |  |
| `--portal-legacy-tab-rail` | `var(--portal-raw-ink-5)` | ″ | The rail under the agent page tabs: a hairline where the divider role is lighter. |
| `--portal-legacy-row-hover-quiet` | `var(--portal-raw-ink-2)` | ″ | Hover on the disclosure summary and on pane tab-strip tabs: a step quieter than row hover. |
| `--portal-legacy-brand-selected` | `var(--portal-raw-brand-25)` | `var(--portal-raw-brand-10)` | The shell's brand selection (selected destinations), RUNNING and DEFAULT chips: light at 10% where the role takes 12%. |

#### Type (`type.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-font-display` | `'Oswald', var(--dt-font-sans)` | ″ | H1, panel, dialog and drawer titles, the wordmark. |
| `--portal-font-body` | `'Mona Sans', var(--dt-font-sans)` | ″ | Every other line of UI. |
| `--portal-type-display-xl-size` | `2.5rem` | ″ | Page H1, agent name (1036:7378). |
| `--portal-type-display-lg-size` | `1.875rem` | ″ | Panel title, sub-page H2, dialog and drawer title (1036:7476, 1038:7797). |
| `--portal-type-section-size` | `0.9375rem` | ″ | Section heading, stat value (1036:7481). |
| `--portal-type-body-size` | `0.875rem` | ″ | Row label, button, tab, input value (label 500, body 400). |
| `--portal-type-ui-size` | `0.8125rem` | ″ | Nav row, sidebar row, menu item, description (1036:7410). |
| `--portal-type-meta-size` | `0.75rem` | ″ | Timestamps, row meta, the composer's model row (906:19738). |
| `--portal-type-micro-size` | `0.6875rem` | ″ | Nav counts, tags, keycaps, the "You" label (1036:7434). |
| `--portal-type-caption-size` | `0.625rem` | ″ | UPPERCASE group labels, table headers, stat labels, menu labels (1036:7408). |
| `--portal-type-chip-size` | `0.5625rem` | ″ | UPPERCASE DEFAULT / RUNNING / EMPTY chips (1036:7318). |
| `--portal-type-field-touch-min` | `16px` | ″ | The smallest text a field may use on a coarse pointer in the browser: iOS Safari zooms into any focused field under 16px (px, not rem: the zoom rule reads CSS pixels). |
| `--portal-tracking-display` | `-0.01em` | ″ | Tracking per role: display titles tighten, small UI opens up, captions and chips open most. |
| `--portal-tracking-ui` | `0.01em` | ″ |  |
| `--portal-tracking-chip` | `0.04em` | ″ |  |
| `--portal-tracking-caption` | `0.06em` | ″ |  |
| `--portal-legacy-caption-tracking-sidebar` | `0.04em` | ″ | The chat sidebar's section captions (Pinned / Agents / Machines) track at the chip's 0.04em. |

#### Space and layout (`space.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-rail-width` | `3.75rem` | ″ | The global rail and the developer rail (1036:7202). |
| `--portal-rail-reserve` | `0px` | ″ | What the rail takes from the layout: its width, none on compact, where it rides the sheet. |
| `--portal-breakpoint-medium` | `640px` | ″ | The responsive model (app/redesign/breakpoints.ts, DESIGN.md › Responsive): compact below the medium breakpoint, medium below the wide one, wide from there. A custom property cannot feed a media query, so every redesign query repeats the literal (max-width: 639.98px / 1023.98px) with a comment naming the class; these two document it. |
| `--portal-breakpoint-wide` | `1024px` | ″ |  |
| `--portal-sheet-width` | `min(20rem, 100vw - 3.5rem)` | ″ | Below wide the chat sidebar and page panels are sheets over the stage: the panel's width, leaving a 56px strip of scrim to tap. |
| `--portal-sheet-panel-width` | `calc(var(--portal-sheet-width) - var(--portal-rail-width) + var(--portal-rail-reserve))` | ″ | The sheet's panel past the rail: the whole sheet on medium (the rail stays in the layout), the sheet less the rail on compact (the rail rides the sheet). |
| `--portal-touch-target` | `2.75rem` | ″ | The smallest hit area on a coarse pointer (WCAG 2.5.5); a control's visual may stay smaller. |
| `--portal-space-compact` | `0.75rem` | ″ | A phone's tighter rhythm (compact and short windows): the composer's padding and gaps, its dock's floor, and on touch its control rows, whose 44px targets overlap the padding. |
| `--portal-space-tight` | `0.5rem` | ″ |  |
| `--portal-composer-row-touch` | `1.75rem` | ″ |  |
| `--portal-safe-top` | `env(safe-area-inset-top, 0px)` | ″ | The browser's safe area (notch, home indicator) once the page is laid out under it (viewport-fit=cover, app/redesign/browser-chrome.ts); zero on every other host. The app root is inset by it as a whole (portal-shell.css), so only body-portalled surfaces (dialogs) read these. |
| `--portal-safe-bottom` | `env(safe-area-inset-bottom, 0px)` | ″ |  |
| `--portal-safe-start` | `env(safe-area-inset-left, 0px)` | ″ |  |
| `--portal-safe-end` | `env(safe-area-inset-right, 0px)` | ″ |  |
| `--portal-panel-width` | `var(--portal-sidebar-width, var(--sessions-pane-width))` | ″ | ONE sidebar column for every rail destination: page panels (Agents, Schedules, Settings) take the chat sidebar's width — the user's sash width once they size it (published by the pane tree as --portal-sidebar-width), else its default — so switching destinations never moves the panel edge, its divider or the stage's content edge. |
| `--portal-frame-inset` | `1rem` | ″ | The page-frame contract: where a page's content column starts, from the stage edge it shares with the page's sidebar panel (the freshest frames, 1036:7189 / 895:*: 20px after the divider, one value for Agents, Settings and Schedules alike), and from the stage's own edge on a page with no panel (919:9281: x 90 in the stage). Left insets, never centring, at every window width, so switching pages never moves the column sideways. |
| `--portal-stage-inset` | `1rem` | ″ |  |
| `--portal-column-form` | `52.5rem` | ″ | The column's one max width per page kind: forms and details (1036:7189 General: 840), catalogs and tables (919:9281: 980), and the conversation shared by the transcript and the docked composer. |
| `--portal-column-catalog` | `61.25rem` | ″ |  |
| `--portal-column-conversation` | `40rem` | ″ |  |
| `--portal-column-landing` | `30rem` | ″ | The landing's composer before the first message (906). |
| `--page-content-inset` | `1rem` | ″ | Full-width pages without a panel pad their catalog column by this much on each side. |
| `--page-content-max-width` | `calc(var(--portal-column-catalog) + 2 * var(--page-content-inset))` | ″ |  |
| `--portal-page-end-space` | `2rem` | ″ | The space a page scroller keeps below its last row, under its bottom fade. |
| `--portal-row-chrome` | `1.875rem` | ″ | Row heights: 30 chrome buttons (906:19469), 40 panel list rows (1038:7408). |
| `--portal-row-panel` | `2.5rem` | ″ |  |
| `--page-nav-width` | `var(--portal-panel-width)` | ″ | Names in use before the role names; Phase B moves their readers to the role and deletes them. |
| `--page-panel-inset` | `var(--portal-frame-inset)` | ″ |  |
| `--page-stage-inset` | `var(--portal-stage-inset)` | ″ |  |
| `--page-form-max-width` | `var(--portal-column-form)` | ″ |  |
| `--page-catalog-max-width` | `var(--portal-column-catalog)` | ″ |  |
| `--portal-chrome-button-size` | `var(--portal-row-chrome)` | ″ |  |

#### Shape: radii, strokes, shadows (`shape.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-radius-chip` | `2px` | ″ | Chips, counts, segments, tags (877:4186). |
| `--portal-radius-control` | `3px` | ″ | Buttons, inputs, selects, checkboxes, keycaps, chrome buttons, page-panel rows (1036:7409). |
| `--portal-radius-card` | `4px` | ″ | Cards, tables, banners, menus (1036:7576, 878:6044). |
| `--portal-radius-surface` | `6px` | ″ | Chat content (bubble, tool row, code card) and chat-sidebar rows (491:40655, 147:7470). |
| `--portal-radius-floating` | `8px` | ″ | Popovers, dialogs, choice tiles (616:1685). |
| `--portal-radius-panel` | `9px` | ″ | The outer top corner of panels, the stage and the composer (906:19804). |
| `--portal-radius-round` | `999px` | ″ | Switch tracks, swatches, dots. |
| `--portal-focus-ring-width` | `2px` | ″ | The one focus ring: 2px of brand ink as an outline, so it never replaces a control's own stroke shadow. Outset by default; a context whose scroller or overflow would clip an outset ring (sidebar, page nav) sets --portal-focus-ring-offset to the inset value on its container and every control inside follows. |
| `--portal-focus-ring-outset` | `1px` | ″ |  |
| `--portal-focus-ring-inset` | `calc(-1 * var(--portal-focus-ring-width))` | ″ |  |
| `--portal-focus-ring-offset` | `var(--portal-focus-ring-outset)` | ″ |  |
| `--portal-shadow-floating` | `0 8px 24px var(--portal-raw-shadow-50)` | `0 1px 2px var(--portal-raw-shadow-4), 0 8px 24px var(--portal-raw-shadow-8)` | Menus, popovers, dialogs (878:6044): one soft drop shadow. |
| `--portal-shadow-composer` | `0 4px 4px var(--portal-raw-shadow-25)` | ″ | The composer card on dark (906:19844); light draws none. |
| `--portal-legacy-card-radius-settings` | `8px` | ″ | Global settings cards and the agent-settings link card: the floating radius where cards take 4. |
| `--portal-control-radius` | `var(--portal-radius-control)` | ″ | Names in use before the role names; Phase B moves their readers to the role and deletes them. |
| `--portal-menu-radius` | `var(--portal-radius-card)` | ″ |  |
| `--portal-panel-radius` | `var(--portal-radius-panel)` | ″ |  |
| `--portal-floating-shadow` | `var(--portal-shadow-floating)` | ″ |  |

#### Motion (`motion.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-motion-fast` | `120ms` | ″ |  |
| `--portal-motion-base` | `180ms` | ″ |  |
| `--portal-motion-slow` | `240ms` | ″ |  |
| `--portal-ease` | `cubic-bezier(0.2, 0.8, 0.2, 1)` | ″ |  |

#### Layers (`layers.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-layer-dock` | `5` | ″ | The docked composer's own top layer (its spectrum edge over the card). |
| `--portal-layer-stage-overlay` | `20` | ″ | What floats over the transcript inside the stage (the conversation header row). |
| `--portal-layer-sheet-scrim` | `44` | ″ | Sheets below wide (the chat sidebar, a page panel) and their scrim: over the stage and the band, under menus and popovers (z-50), so a row's menu opens over its sheet. The rail joins a compact sheet from outside the page layer (route pages are z-50), so it alone sits just over it. |
| `--portal-layer-sheet` | `45` | ″ |  |
| `--portal-layer-sheet-rail` | `51` | ″ |  |

#### Bridge (`bridge.css`): portal roles under the names shared components read

| Token | Dark | Light | Note |
|---|---|---|---|
| `--theme-fill-primary-accent-mix` | `0%` | ″ | Brand marks belong to the shell, not the skin: neutral fills and rows take no accent tint. |
| `--theme-fill-secondary-accent-mix` | `0%` | ″ |  |
| `--theme-fill-tertiary-accent-mix` | `0%` | ″ |  |
| `--theme-fill-quaternary-accent-mix` | `0%` | ″ |  |
| `--theme-fill-quinary-accent-mix` | `0%` | ″ |  |
| `--theme-row-hover-accent-mix` | `0%` | ″ |  |
| `--theme-row-active-accent-mix` | `0%` | ″ |  |
| `--theme-control-hover-accent-mix` | `0%` | ″ |  |
| `--theme-control-active-accent-mix` | `0%` | ″ |  |
| `--ui-text-primary` | `var(--portal-text-primary)` | ″ |  |
| `--ui-text-secondary` | `var(--portal-text-secondary)` | ″ |  |
| `--ui-text-tertiary` | `var(--portal-text-tertiary)` | ″ |  |
| `--ui-text-quaternary` | `var(--portal-text-quaternary)` | ″ |  |
| `--ui-bg-card` | `var(--portal-legacy-card-fill-settings)` | ″ |  |
| `--ui-stroke-primary` | `var(--portal-stroke-strong)` | ″ |  |
| `--ui-stroke-secondary` | `var(--portal-raw-ink-14)` | ″ | Ink 14 in both modes; the field role is 14 / 18 (Phase B). |
| `--ui-stroke-tertiary` | `var(--portal-stroke-hairline)` | ″ |  |
| `--ui-stroke-quaternary` | `var(--portal-stroke-hairline)` | ″ |  |
| `--sidebar-edge-border` | `var(--portal-stroke-edge)` | ″ |  |
| `--ui-sidebar-surface-background` | `var(--portal-sidebar-background)` | ″ |  |
| `--ui-chat-surface-background` | `var(--portal-surface-stage)` | ″ |  |
| `--ui-editor-surface-background` | `var(--portal-surface-stage)` | ″ |  |
| `--radius-scalar` | `0.6` | ″ | Tailwind rounded-* in shared components: the redesign's squarer scale. |
| `--terminal-panel-min-width` | `min(22rem, 40vw)` | ″ |  |
| `--global-nav-width` | `var(--portal-rail-reserve)` | ″ |  |
| `--page-nav-row-height` | `var(--portal-row-panel)` | ″ |  |
| `--composer-width` | `var(--portal-column-conversation)` | ″ |  |
| `--dialog-body-padding` | `1.5rem` | ″ |  |
| `--dialog-body-gap` | `1.25rem` | ″ |  |
| `--titlebar-control-size` | `var(--portal-row-chrome)` | ″ | Figma's 30×30 chrome buttons (r3, 16px glyphs): titlebar, rails, pane headers; breadcrumb spacing 906:19467. |
| `--titlebar-control-height` | `var(--portal-row-chrome)` | ″ |  |
| `--titlebar-icon-size` | `1rem` | ″ |  |
| `--titlebar-control-gap` | `0.625rem` | ″ |  |
| `--ui-green` | `var(--portal-status-online)` | ″ |  |

<!-- design-tokens:end -->

### To unify in Phase B

Phase A made the system real without moving a pixel, so roles that still have
two values in the product keep both, each under its own token or literal. Each
row is one reviewable change: point the legacy reader at the role (or edit the
role), delete the loser, capture before/after in dark and light. Ordered by
visual impact.

| # | Role | Today: value · where | Winner (Figma) |
|---|---|---|---|
| U1 | Card (grouped rows) | `--portal-legacy-card-fill-settings` ink 5 (global settings cards via `--ui-bg-card`); `--portal-legacy-card-radius-settings` 8px (`.global-settings-card`, `.settings-agent-link`); `--portal-legacy-disclosure-fill` ink 3 + `--portal-legacy-disclosure-stroke` ink 10 (`.global-settings-disclosure`); capability rows stroke ink 10 (`capabilities/ui/page.css`, shared with classic) | `--portal-surface-card` ink 2, `--portal-stroke-hairline` ink 5, `--portal-radius-card` 4px (1036:7576) |
| U2 | Page rhythm | global settings `--settings-section-gap` 36px and page-heading gap 36px (`global-overviews.css`, `appearance-overview.css`, `portal-pages.css`) vs 32px on agent pages (`settings-editor.css`) | 32 between sections and after the heading (1036:7405) |
| U3 | Page tabs | two implementations (`.agent-management-tabs`, `.page-header-tabs`); rail `--portal-legacy-tab-rail` ink 5 and an ink 5 literal in `page-search-shell.css`; Capabilities tabs keep a 0.5rem bottom pad | one `PageTabs`, rail `--portal-stroke-divider` ink 3 (1038:7499) |
| U4 | Divider inside a card | `--ui-stroke-tertiary` ink 5 (global settings rows, appearance groups, capability rows) vs `--portal-stroke-divider` ink 3 (agent, history, config cards) | `--portal-stroke-divider` ink 3 (light ink 5) |
| U5 | Row hover | `--portal-surface-row-hover` ink 3 (pages); `--portal-sidebar-hover` ink 4 (chat sidebar); `--portal-legacy-row-hover-quiet` ink 2 (disclosure summary, pane tab-strip tabs) and `--portal-sidebar-agent-hover` ink 2; theme-mode tile ink 7 and gateway tile ink 8 literals | `--portal-surface-row-hover` (+1 step over rest) |
| U6 | Neutral selection | `--portal-surface-selected` ink 5 + `--portal-stroke-selected` (page nav) vs `--portal-sidebar-selected` ink 7, no stroke (open pinned chat) and the pane tab-strip's active tab (ink 5, no stroke) | ink 5 + 0.5px ink 5 stroke (1036:7409) |
| U7 | Caption | `--portal-legacy-caption-tracking-sidebar` 0.04em with weight 500 and line height 1.2 (`.portal-section-label`); new-session picker label 0.08em/500/brand ink; gateway menu `tracking-[0.08em]` | 10/1.4, `--portal-tracking-caption` 0.06em, 400, tertiary (1036:7408) |
| U8 | Chip and Count | sidebar Agents count r10 pill at 700 (`.portal-section-count`); new-session tag 0.06em/500/r3; RUNNING/EMPTY chip line boxes | Chip h14 9/0.04em r2, Count h14 10/500 r2 (1036:7318, 856:3474) |
| U9 | Keycap | `--portal-sidebar-action-key` (dark 20%) with a 0.5px ink 5 rim (sidebar, New agent row) vs `[data-slot=kbd]` field ink 4 + 1px control stroke | ink 4 fill, 1px ink 14, r3, 11px (1038:7401) |
| U10 | Danger | `destructiveOutline` on the theme's `--color-destructive` at 10/18/24/45%; destructive menu item at 12%; thread error card `--err-alert`; `--portal-surface-danger-solid` `#a22a20` | one family: `#a22a20` fill 8% / stroke 50%, ink `#ff7373` dark / `#a22a20` light (1038:7836) |
| U11 | Display-lg | pane title 2rem (`portal-shell.css`); dialog title without tracking (`portal-typography.css`) | `--portal-type-display-lg-size` 30, `--portal-tracking-display` (1038:7797) |
| U12 | Text ink ladder | `--portal-floating-item-ink` ink 90; history heading/group ink 90 and settings-editor ink 98 literals; sidebar machine ink 80, head meta 70/55, dark caption 65 | the four `--portal-text-*` steps |
| U13 | Dialog and drawer | `--dialog-body-gap` 20 (root) vs 16 (`.agent-history-dialog-body`, px) ; drawer footer buttons 36 | padding 24, block gap 24, footer buttons 44 (1038:7792) |
| U14 | Small text in mixed TSX | 14 sizes between 0.6 and 0.82rem and 0.07–0.2em tracking in the classic/portal branches | caption / micro / meta / ui sizes |
| U15 | Radius by role | off-scale `10px` (sidebar count), `0.5rem` / `0.375rem` (agent tile and avatar), Tailwind `rounded-*` in redesign TSX (G2 warnings) | the seven `--portal-radius-*` roles |
| U16 | Focus ring | chat sidebar inset box-shadow; tabs and history rows reshape the ring's radius | one 2px outline, `--portal-focus-ring-*` |
| U17 | Brand selection (light) | `--portal-legacy-brand-selected` 25% / light 10% (`--portal-selection-background`: rail destinations; RUNNING and DEFAULT chips) vs `--portal-surface-brand-selected` 25% / light 12% (segmented, scope chips) | 25% / light 12% (1036:7318) |
| U18 | Control and floating states | `--portal-surface-control-hover` light ink 6, `--portal-surface-control-pressed` ink 11 / light 9; `--portal-stroke-floating` light ink 12; `--portal-floating-separator` ink 6 | control hover 8/5, pressed 10/8; floating stroke ink 10; separator `--portal-stroke-hairline` |
| U19 | Units | px font sizes, line heights and radii in `agent-settings.css`, `history.css`, capabilities `page.css` (the guard's baseline lists them) | rem through the type tokens |
| U20 | Spacing drift | 3/5/7/9/11/13/15/17/22/23px insets (sidebar, catalog, history, settings editor) | the 4px scale; Figma optical offsets keep `/* token-exempt */` |

### Rules of thumb

- A new colour must be a role in `color.css`. A new alpha is not allowed: pick
  the nearest ink step.
- A new size must be a type role. If none fits, the design or the language is
  wrong: change the language (here and in `tokens/`) first.
- Cards never nest. A card inside a disclosure body loses its fill and stroke.
- Blue means action or selection, never decoration.
- Status colour lives in dots, chips and one status label — never in row fills
  (except the chat sidebar's attention rows).
- A Figma one-off (a 23px optical offset) is encoded in the component with a
  `/* token-exempt: why */` comment, not as a token.
- Redesign-only TSX uses token-backed classes (the `ds-` utilities: `text-ds-caption`,
  `rounded-ds-card`, `bg-ds-card`, `text-ds-text-3`, from `tokens/tailwind.css`), not arbitrary
  visual values such as a literal rem size, a px radius or an inline colour mix: the
  `hermes-desktop/no-portal-arbitrary` lint warns on them.

### How to change a token

1. Edit the one line in `src/app/redesign/tokens/<group>.css` (dark in the base
   block, light in the `:not(.dark)` block).
2. Regenerate the tables: `python scripts/check_desktop_design_tokens.py --write-docs`
   (`scripts/check` fails while they are stale).
3. Look at it in dark, light and Glass, on the states the role appears in.
4. If the change unifies a Phase B row, delete the legacy token and its row, and
   lower the guard's baseline (`--write-baseline`) in the same change.

`scripts/check` runs `check_desktop_design_tokens.py`: outside `tokens/`, redesign
CSS that restates a colour, an ink mix, a radius, a font size, tracking, a px
line height, a duration or a z-index fails once a file's count rises above its
ratchet (`scripts/desktop_design_tokens_baseline.json`, which may only go down),
and a `var(--portal-…)` naming an undefined token always fails.

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
- **The redesign (portal shell) draws every one of those glyphs from
  Pixelarticons**, the brand set the Figma frames use. The swap is central:
  `src/lib/pixel-icons.ts` maps each Tabler alias and codicon name to its pixel
  twin (the Figma frame's icon where one draws the control), `src/lib/icons.ts`
  aliases and `Codicon` pick the set from the shell, and classic is untouched.
  A new alias or codicon needs a mapping entry (`KEEP_CLASSIC` for a brand or
  language mark); the type and `pixel-icons.test.tsx` fail without one. A role
  that the redesign draws differently from a shared glyph gets its own alias
  (`Gateway`: the classic globe, the redesign's `server`). Directional glyphs
  mirror in RTL from `RTL_MIRRORED`, not per component.
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
