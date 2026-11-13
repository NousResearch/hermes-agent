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

### Terminology: classic vs redesign

The redesign renames five product nouns; classic keeps `main`'s wording verbatim, so turning
the **Redesigned interface** preference off restores exactly what `main` ships.

| Classic (base catalog, `main`'s wording) | Redesign |
|---|---|
| Sessions | Chats |
| Artifacts | Library |
| Profiles / bots (Hermes agents) | Agents |
| Cron / cron jobs / scheduled jobs | Routines |
| Gateways (and "host" meaning a Hermes machine) | Machines |

The swap is a wording layer, not a fork: `resolveTranslations(locale)`
(`src/i18n/registry.ts`) reads the shell from the presentation leaf
(`src/store/shell-presentation.ts`, i.e. `$portalShell`) and, in the redesigned shell,
merges `src/i18n/redesign/<locale>.ts` over the base catalog — before backend/plugin
language packs, which still win. Each overlay holds ONLY the keys whose wording differs,
typed against `Translations`; a locale inherits the English overlay where its base still
shows English. Call sites keep reading `t.x.y` / `translateNow(...)`; never branch a string on
the shell. The HUD and popped-out Browser always render classic, so they read classic
wording. **New shell wording goes in the overlay, never in the base catalog.** Redesign-only
surfaces with their own copy modules (`src/i18n/simplified-agents.ts`, `src/app/**/*-copy.ts`)
use the redesign terms directly. `src/i18n/redesign/terminology-guard.test.ts` fails on a
classic noun left in the redesign and on a redesign-only word reaching classic.

Rulings: a routine *has* a schedule ("schedule" stays the word for its timing; "cron
expression" stays the input syntax's name). Service operations say **Restart Hermes**, not
“restart the machine,” which would imply rebooting the computer. "host" stays only for a
literal hostname or SSH host; skill/plugin "library" is "catalog", so Library means only the
artifacts destination. Team Gateway and Tool Gateway remain product names; Group chats and
Channels keep their names. Actual Telegram/Discord bots, subagents, browser profiles, the
memory "user profile", sign-in sessions and session tokens, and user-authored names retain
their meanings. Protocol fields, `/profiles` and `/cron` routes, CLI commands, cron
expressions, plugin IDs, and the exact stored `Bot Chat` title are technical contracts, not
copy to rename.

Drift coverage exercises navigation into Agent and Routine editors, cancellation,
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
touching the window top become panels starting below the band, together one workspace frame:
a card on the canvas between the rails (`--portal-workspace-inset-start` past the global rail and
before the developer rail, `-end` and `-bottom` to the window edge: x 65–1265, y 60–865 of Figma's
1280×880) with 15px outer corners (`--portal-radius-frame`), a soft shadow
(`--portal-shadow-workspace`) and a top highlight (`--portal-shadow-workspace-highlight`). Its
panels, the developer tool panes among them, meet square inside it with no hairlines (Content
`16292:14463`; a phone keeps the full window). Behind it all lies the launch's artwork (see "Artwork" below), which the canvas
shows at full strength and every panel veils. Window Glass (macOS default) never switches the redesign off: its
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
(`--portal-frame-inset` past a sidebar panel, `--portal-stage-inset` without one), and Routines lists every
agent's routines in the shared page sidebar panel like Agents (see Routines below).
Workspace pages (Capabilities, Library, …) and the overlay pages lie in the same
workspace frame as the chat. Without an explicit skin pick, the redesign paints the
Hermes Portal skin and classic paints Nous (`defaultSkinName`); flipping the
shell never writes a pick.

Redesigned surfaces use neutral foreground-derived fills and strokes, not
accent-tinted greys (the colour roles below). Global settings cards and agent settings cards wear
the one card role. Settings-navigation selection is the neutral selection role;
sub-pages reveal under the active row, without per-row carets. Settings pages compose
`SettingsOverview` / `SettingsCard` / `SettingsDisclosure` (`settings-card.tsx`), which fall back to
the classic wrapper. The global rail marks its open destination with a brighter navy tile. Agent-management navigation
tiles stay translucent; agent rows and the overview header use the Figma robot with the real
owner's color.
The global rail tracks the owning agent even inside its settings editor.
Shared rail/sidebar geometry lives at the document root while the redesign is
enabled. Sibling route overlays reserve the rail with logical inline offsets
in both LTR and RTL layouts.
Checked switches use the brand-blue track and light thumb across settings
and the capability catalog; unchecked tracks are neutral 14% fills.

The global rail (rail 16292:15233) carries no brand badge: Figma hides that layer and draws the
window's traffic lights at the top of the column, so the destinations start 90px down on every host
(clear of macOS lights), 30px navy tiles with 16px glyphs at .4 ink on a 50px pitch, the open one a
brighter tile with its glyph in full ink; the foot (usage, Settings) carries 20px glyphs 40px off the
floor, and Routines draws a clock. The rail never moves between states and has no edge of its own. The redesigned chat sidebar
(340px, "Prod - Desktop" Panel 16292:15497) opens with its header (Top 16292:15498): the "Hermes"
wordmark in Oswald Light 30 over a mono "// NOUS RESEARCH" caption (brand marks, never translated),
and at its end a 30px search button. Search lives behind that button: it brings out a rimmed field
under New chat with the session-search shortcut's keycaps (hidden on touch), and the shortcut or the
command palette bring it out too; Escape on an empty field or the button again puts it away and ends
the search. New chat (16292:15511) is a 40px radius-3 slab with rimless mono keycaps and keeps its
agent picker at the end. The open destination's rail button folds its panel and brings it back: Chats folds the chat sidebar
("Hide chats" / "Show chats", `aria-expanded`), Agents, Schedules and Settings their page panel
("Hide panel" / "Show panel"); it stays the open destination either way, and ⌘B does the same. Picking
a destination with a panel from elsewhere shows it whole; a destination without one leaves a folded
panel alone. Page panels carry no fold button of their own. While a panel is folded on a wide window,
Simple mode brings it back from one floating 44×40 button at the stage's top-leading corner, under the
band (1061:7816, `app/shell/panel-reveal.tsx`), and Advanced from the band's sidebar button; below wide
the band's button is the sheet's menu in both. Dragging the sidebar's sash 72px under its 280px minimum
(`--sessions-pane-collapse-width`) folds it live as the pointer crosses and on release, writing no
width, so it returns at the width it had; a drag that stops short rests at the minimum. Classic defines
no fold width, so its sash never folds.

Below New chat the sidebar follows the same panel (Centre 16292:15509): sections under sentence-case
mono captions with the disclosure chevron right after the label (Agents keeps its brand count badge at
the end), separated by whitespace only, on a 32px text inset with row fills bleeding 12px past it.
Pinned and group-chat rows are 20px of SemiBold title on a 36px pitch after a 5px leading dot (row ink
at rest, full ink for the open chat, brand for unread news, the status tones while a turn runs or
waits), their tags, ages and statuses in the mono font, 10px uppercase. Machines are rows like them,
their dot in the machine's status tone and the status label at the end; an open machine is one radius-12
card at 2% ink (set 16292:15581) whose header sticks to the top of the list while its agents scroll
under it. Agents are cards inside it (16292:15592): radius 9, the agent's 32px colour tile (in its
snapped Figma primary, `lib/portal-profile-color.ts`; Figma's character avatars are not adopted) over
the name and a mono second line (roster description, else the chat count), the caret at the end and the
CRT texture dodged into the fill. The open agent is the brand slab (white .10 / black .40 / blue), the
others a navy card; light draws both as a pale brand card under a 1px brand rule, solid for the open
one. An agent's chats are 40px rows across the machine card with the dot 30px in; no fill marks the
open chat, its title and dot take full ink while the agent's other chats sit at the ladder's floor.
An agent's card lists its three most recent chats (plus the open one and any running or unread)
over a "Show all (N)" / "Show less" row remembered per agent; search lists every match. A row under
the pointer or keyboard focus takes a 4% fill. Figma's "Recent" list is not built: it would repeat the
chats the agent cards already list. A chat row resolves the click and its title is the one focusable button, as in classic, so
plugin row decorations (`sessionRow.leading` before the title, `.trailing` before the age) and an
agent card's `sidebar.profileGroup.header` contributions sit beside that button, never in it. Row and
agent actions (ellipsis, caret, kebab) are bare glyphs that appear on hover, keyboard
focus or an open menu; touch keeps them visible. Arrow keys, Home and End walk the
visible rows; right-click opens the row menu. Captions and meta keep the AA ladder
where Figma's inks fail 4.5:1. The band's leading cluster (Back/Forward on the native app) starts at
`--portal-band-start` (x 150 past macOS traffic lights, navigation 16292:15284; else 80) and the
session pill is anchored 25px past the stage's leading edge (tab 16292:15297) with CSS anchor
positioning, so it tracks a sash drag in the same frame. Band buttons and the pill are navy tiles
(white .05 / black .60 / blue .20) with glyphs at .4. The main chat's Agent settings is the band's
trailing sliders button (settings-2 16292:15304), present wherever a conversation holds the stage;
an extra tile keeps its own in its header. The stage keeps its "machine // agent" line at its top end
in the mono caption style, though Figma draws none.
The global Agents icon uses the Figma robot silhouette tinted by `currentColor`,
sharing its peers' inactive, hover, and selected tones in both light and dark themes;
rail destinations rest at .4 ink on navy tiles and the open one takes the brighter tile with its glyph in
full ink.
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
A page's list panel (Agents, Routines; `PageSidebarPanel`) wears the chat sidebar's and the
Settings column's header: its title 30px in and 30px down on a 36px line, its start row (New agent,
New routine) 122px under the panel's top where New chat starts, as New chat's borderless light/5 slab
with rimless mono keycaps, and its open row a light/5 fill with no stroke.
Every routed page keeps one page frame (`tokens/space.css`): beside a sidebar
panel the content column starts `--portal-frame-inset` (20px, the freshest frames 1036:7189 /
895:*) after the panel's divider, on Agents, Settings and Routines alike; without a panel it
starts `--portal-stage-inset` (Figma 919: x 150 at 1280) into the stage. Columns are
left-anchored at every window width, never centred, with one max width per page kind:
`--portal-column-form` (840) for forms and details, `--portal-column-catalog` (980) for
catalogs, lists and tables. Every H1 (or agent hero) tops out at the same y; header actions end
at the column's right edge. Global Settings is the exception ("Prod - Desktop" 06 · Settings/Account,
16292:16447): a 640px column centred in the stage, its H1 (Oswald Light 40/48, -1px) 104px under the
stage's top with a mono description; the column's content box is its text column, 32px inside
the 640 (the header and the scroller pad to it), so the header, section heads (14 SemiBold at .8),
forms and every page's own content start at one x, and only the card primitives (SettingsCard,
SettingsDisclosure, the empty block) bleed out to the column's edges with their rows padded back in; cards are radius 15, borderless, white .02 / black .50 / blue .10, 16px over
and under their rows, rows 12px with 16px Regular labels and mono descriptions. Its panel (16292:16359)
titles "Settings" on a 30px inset with the settings search at its end (the band's pill is gone; typing
and ⌘K still open the palette), 40px rows 4px apart in SemiBold with no stroke on the selected one, and
the Agent settings card 20px under them: radius 9, borderless, the lattice art masked in from its
leading edge. Figma's "// META INFO" caption (placeholder copy) and its Help & Feedback entry (no
destination yet) are not built; Advanced and the Documentation link stay, though Figma omits them.
An agent's page has one vertical menu (Figma 1036:7189). On Overview and History the
page sidebar panel lists the agents; on the Settings tab the same panel swaps to
"← Back to agents" (a history step back to the Overview), the agent's own row, and the
settings sections (10px group labels, 32px rows, counts at the row end), while the stage
holds the agent header, the tabs and the selected page at full width. When the panel is
folded away (the Agents rail button, ⌘B) or the window is too narrow for panels, the sections menu leads the
page in the stage instead, so every section stays reachable.

Transcript styling is isolated in `components/assistant-ui/thread/redesign.css`,
following the Figma conversation (`491:40346`; width from "Prod - Desktop" chat 16292:16175): one
centred 440px column shared with the docked composer (`--portal-column-conversation`, read as
`--portal-conversation-*` in `app/chat/stage.css`), prose flush with the composer's edges
(`--portal-column-conversation-inset`), 30px turn gaps and 20px output gaps. User bubbles hug their text at
the right edge up to 380px, with 6px corners, 6/10px insets and a 5% (light 2%)
fill and no border; their time and restore actions sit beside the bubble. Base text
is 13px at 1.6 leading before user scaling. A pinned prompt keeps 12px of clearance
below its bubble so clipped reply lines never touch it. Retain the current branded
body font, timestamp preference, and working message actions; enabled timestamps
use caption ink and sit above the bubble. Collapsed tool runs, including
single-tool runs, use the dashed [2,4] summary frame with a status dot. The stage is a
veil over the launch's artwork, lighter behind a new chat than behind a conversation. The composer has no ring: its static 2px spectrum edge
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

The redesigned composer is 440 × 148 on an empty landing (msg 16292:15670). In a populated native
conversation it keeps the 148px base height, spans the shared 440px column and
docks 30px above the stage floor. Its card is navy, white .05 / blue .10 / black .80 flattened onto
the stage (white on light), with no shadow, and its resting prompt is the card's heading line in
full ink (16292:15681); Figma's dim second line ("Prompt suggestions or processing...") is template
copy and is not built, with model/effort as quiet right-aligned text above the input and attachment, microphone, and labeled Send
controls below. Browser touch targets keep their larger accessibility sizing.
Gateway menus distinguish connection from inference
readiness. Simple mode keeps reconnect and lazy recent activity; Advanced adds
system/restart/logs, with 28px icon controls and destructive restart styling at rest.
A routed page waits for its code inside its own frame (`overlays/page-frame-fallback.tsx`: the
page's panel and a header at the PageHeader geometry), never an empty window; rail buttons warm their
page on hover and focus. A redesign confirm has one anatomy: no corner X, a bordered Cancel beside the
solid action, and a destructive confirm opens on Cancel. The developer tool panel (Files, Terminal,
Review, Browser) has one user-resizable width shared by every tool, never under a tool's own minimum.
The rail's Settings menu lists the Settings page's destinations, in its order and under its names
(`settings/redesign-navigation.ts`), with Interface mode and Documentation below a separator.

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
  composer on one row (attach, editor, send), hides the model row until there is room, and takes the browser's compact band (48, touch 44).
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

### Artwork

The redesign paints one artwork per app launch behind the whole window (Figma "Prod - Desktop" ›
Brand, frame `16292:10764`: one row per artwork). Dark rotates Vanilla (the engraved wing as a
trace), Wing (the RGB wing), Angel, Ascend and Karborn; light rotates its own Vanilla and Ascend.
`app/redesign/artwork.ts` picks the next one in that order on every launch (per device, in
localStorage), so two launches in a row never repeat; every window of one launch shares the pick
through the main process's launch id (`hermesDesktop.launchId`), and in a browser each page load
is a launch. A theme switch keeps the same artwork when the other theme draws it, else the next
one it does. The pick is mirrored onto `<html data-artwork-dark/-light>` before the first paint;
classic, the HUD and the popped-out Browser get none.

`app/redesign/portal-artwork.css` holds the art: Figma's geometry in frame pixels (the 1270×870
Background), scaled to cover the window and centred, plus its ground (`--portal-artwork-ground-*`,
which becomes `--portal-surface-canvas`). Panels are veils over it, Figma's own fills: the stage of
a new chat and a page body `--portal-surface-stage-veil`, a conversation the stronger
`--portal-surface-stage-veil-conversation`, the sidebar kind `--portal-sidebar-panel-background`.
Every painter repeats the art on its own box (`--artwork-paint`, `background-attachment: fixed`)
so nested surfaces never compound a veil; covers that cannot (`--portal-surface-stage`,
`--portal-page-background`) take the veil flattened onto the ground. With a sidebar or page nav
docked (wide), the figure artworks move right by Figma's own amount to stay on the stage; every
image carries bleed past every placement (Ascend repeats across), so no window size ever reaches an
image edge. The panels carry Figma's NOISE effects as tiny tiling `feTurbulence` SVGs
(`assets/portal-noise-*.svg`, `--artwork-panel-noise`/`--artwork-stage-noise`): the sidebar kind in
both themes, the stage in light. Figma's
"01 · Artwork" screens are the canvas layer on its own, not an app state; it shows no setting to
turn the art off or choose one.

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
| `--portal-raw-green` | `#00f279` | ″ | Figma secondary/green (live dots and gradients); at .6 it is the connected / online ink on dark (1089:8305). |
| `--portal-raw-green-60` | `color-mix(in srgb, var(--portal-raw-green) 60%, transparent)` | ″ |  |
| `--portal-raw-mint-deep` | `#1f8a65` | ″ | Connected / online ink on light: the green darkened to AA. |
| `--portal-raw-orange` | `#f27900` | ″ | Figma secondary/orange: pending and warning. |
| `--portal-raw-danger` | `#a22a20` | ″ | Danger fill and stroke (877:4212, 878:7066, 1038:7836) and its soft status text (877:4312). |
| `--portal-raw-danger-soft` | `#cc8a84` | ″ |  |
| `--portal-raw-danger-bright` | `#ff7373` | ″ | The danger label on dark (1038:7836): bright enough for 4.5:1 on the danger tint. |
| `--portal-raw-spectrum-red` | `#aa1515` | ″ | The composer's spectrum edge (906:19872). |
| `--portal-raw-spectrum-yellow` | `#ddd61c` | ″ |  |
| `--portal-raw-gray` | `#797979` | ″ | The neutral of a hovered sidebar row's gradient (1089:8351): the selected row's brand ramp in grayscale. |
| `--portal-raw-shadow-4` | `rgb(0 0 0 / 0.04)` | ″ | Shadow inks. |
| `--portal-raw-shadow-8` | `rgb(0 0 0 / 0.08)` | ″ |  |
| `--portal-raw-shadow-20` | `rgb(0 0 0 / 0.2)` | ″ |  |
| `--portal-raw-shadow-50` | `rgb(0 0 0 / 0.5)` | ″ |  |
| `--portal-raw-highlight-5` | `rgb(255 255 255 / 0.05)` | ″ | The workspace frame's top highlight (16292:14463): white at .05. |
| `--portal-raw-shadow-navy-10` | `rgb(18 20 38 / 0.1)` | ″ | Light's workspace frame shadow ink (16292:23801): a navy at .1. |
| `--portal-raw-ink-2` | `color-mix(in srgb, var(--ui-base) 2%, transparent)` | ″ | The ink ladder (LANGUAGE §1.1): Figma's light/2, 5, 10, 40 plus the page steps 3, 4, 8, 14, 22, 28. No other neutral alpha belongs to the language; the off-ladder steps still in use are listed in color.css and the "Still to unify" table. |
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
| `--portal-raw-brand-30` | `color-mix(in srgb, var(--portal-raw-blue) 30%, transparent)` | ″ | The blue under a sidebar agent card (16292:15630). |
| `--portal-raw-brand-60` | `color-mix(in srgb, var(--portal-raw-blue) 60%, transparent)` | ″ |  |
| `--portal-raw-danger-8` | `color-mix(in srgb, var(--portal-raw-danger) 8%, transparent)` | ″ | Danger tints (1038:7836): the tint 8, its hover and pressed steps 14 and 20, the stroke 50. |
| `--portal-raw-danger-14` | `color-mix(in srgb, var(--portal-raw-danger) 14%, transparent)` | ″ |  |
| `--portal-raw-danger-20` | `color-mix(in srgb, var(--portal-raw-danger) 20%, transparent)` | ″ |  |
| `--portal-raw-danger-50` | `color-mix(in srgb, var(--portal-raw-danger) 50%, transparent)` | ″ |  |

#### Colour roles (`color.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-surface-brand` | `var(--portal-raw-blue)` | ″ | Brand. The fill is the same in every theme and mode; thin rings, underlines and small blue text use the brand ink, which lightens on dark canvases for contrast. |
| `--portal-text-on-brand` | `var(--portal-raw-white)` | ″ |  |
| `--portal-text-brand` | `var(--portal-raw-blue-ink-dark)` | `var(--portal-raw-blue)` |  |
| `--portal-composer-fill` | `color-mix( in srgb, var(--portal-raw-pure-white) 8%, color-mix( in srgb, var(--portal-raw-blue) 10%, color-mix(in srgb, var(--portal-raw-black) 80%, var(--ui-chat-surface-background)) ) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 98%, var(--ui-chat-surface-background))` | The composer card (msg 16292:15670): white .05 / blue .10 / black .80 at .94, flattened onto the stage so the transcript never shows through. Figma's card lets 6% of the art through, which lifts it to #171635 on Wing; the opaque card takes white .08 to land there. Light draws it white (16292:27185). Its resting prompt is the card's heading line at full ink (16292:15681). |
| `--portal-composer-prompt-ink` | `var(--portal-text-primary)` | ″ |  |
| `--portal-surface-brand-selected` | `var(--portal-raw-brand-25)` | `var(--portal-raw-brand-12)` | Brand selection: selected destinations, segments, scope chips, choice tiles and the DEFAULT / RUNNING chips (856:3444, 1036:7318). |
| `--portal-spectrum-red` | `var(--portal-raw-spectrum-red)` | ″ |  |
| `--portal-spectrum-yellow` | `var(--portal-raw-spectrum-yellow)` | ″ |  |
| `--portal-text-primary` | `var(--ui-base)` | ″ | Text ladder. Figma steps ink 1 / .8 / .6 / .5 / .4 / .3, but .4 and .3 fail AA for the 12-13px meta and labels that use them, so the ladder's floor is the least ink that keeps 4.5:1 on every real surface (sidebar, stage, page canvas, cards, popovers, selected rows), and the rungs above are spaced by contrast, not alpha: light ≈ 16 / 8.9 / 6.7 / 5.1:1, dark ≈ 17 / 8.7 / 6.6 / 4.9:1. Dark ink on light canvases loses contrast faster, so light needs more alpha per rung. |
| `--portal-text-secondary` | `color-mix(in srgb, var(--ui-base) 70%, transparent)` | `color-mix(in srgb, var(--ui-base) 80%, transparent)` |  |
| `--portal-text-tertiary` | `color-mix(in srgb, var(--ui-base) 60%, transparent)` | `color-mix(in srgb, var(--ui-base) 72%, transparent)` |  |
| `--portal-text-quaternary` | `color-mix(in srgb, var(--ui-base) 50%, transparent)` | `color-mix(in srgb, var(--ui-base) 64%, transparent)` |  |
| `--portal-surface-canvas` | `var(--portal-skin-window, var(--ui-bg-chrome))` | `var(--portal-skin-window, color-mix(in srgb, var(--ui-base) 5%, var(--ui-bg-chrome)))` | Window canvas (rails, titlebar band): the skin's own canvas, else its chrome. Under the launch's artwork (portal-artwork.css) it is that artwork's ground instead. |
| `--portal-artwork-ground-vanilla` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 94%, var(--portal-raw-blue)) )` | ″ | The artworks' grounds (Prod - Desktop › Brand 16291:10506, each row's Background fill): brand blue under black .94 and white .05 (Vanilla 16292:10768), black .90 (Wing 16292:13812), black .96 (Angel 16292:17116, Ascend 16292:20301); Karborn lays black .98 on the blue and lightens it with a blue .10 filter (16292:35757). Light paints #f2f2f2 under both of its artworks (16292:26683). |
| `--portal-artwork-ground-wing` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 90%, var(--portal-raw-blue)) )` | ″ |  |
| `--portal-artwork-ground-photo` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 96%, var(--portal-raw-blue)) )` | ″ |  |
| `--portal-artwork-ground-karborn` | `color-mix( in srgb, var(--portal-raw-blue) 10%, color-mix(in srgb, var(--portal-raw-black) 98%, var(--portal-raw-blue)) )` | ″ |  |
| `--portal-artwork-ground-light` | `var(--portal-raw-white)` | ″ |  |
| `--portal-surface-stage-veil` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 80%, color-mix(in srgb, var(--portal-raw-blue) 40%, transparent)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 80%, transparent)` | Panels over the artwork are veils, Figma's three-layer fills (top layer first), so the art reads through each at its own strength: navy, the brand blue under black, where the previous design put chat on a neutral (white 5% on black). The stage of a new chat and a page body (Panel/flexible 16292:18060, 16292:19906): white .05 / black .80 / blue .40, about 89% opaque, #0E0E23 where it is solid. |
| `--portal-surface-stage-veil-conversation` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 85%, color-mix(in srgb, var(--portal-raw-blue) 60%, transparent)) )` | `color-mix( in srgb, var(--portal-raw-black) 2%, color-mix(in srgb, var(--portal-raw-pure-white) 96%, transparent) )` | A conversation strengthens it to focus the transcript (05 · Workspace/Chat 16292:18949): white .05 / black .85 / blue .60, about 94% opaque. |
| `--portal-surface-stage` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix( in srgb, var(--portal-raw-black) 85%, color-mix(in srgb, var(--portal-raw-blue) 60%, var(--portal-surface-canvas)) ) )` | `color-mix( in srgb, var(--portal-raw-black) 2%, color-mix(in srgb, var(--portal-raw-pure-white) 96%, var(--portal-surface-canvas)) )` | The stage flattened onto the canvas: what covers inside a conversation paint (the docked composer's floor, overlays on the transcript) where the art itself is not repainted. |
| `--portal-sidebar-background` | `color-mix( in srgb, var(--portal-raw-pure-white) 2%, color-mix( in srgb, var(--portal-raw-black) 85%, color-mix(in srgb, var(--portal-raw-blue) 70%, var(--portal-surface-canvas)) ) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 98%, var(--portal-surface-canvas))` | The chat sidebar's opaque fill (the shared sidebar surface, sheets): its veil flattened onto the canvas. |
| `--portal-page-background` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix( in srgb, var(--portal-raw-black) 80%, color-mix(in srgb, var(--portal-raw-blue) 40%, var(--portal-surface-canvas)) ) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 80%, var(--portal-surface-canvas))` | Routed pages lay their nav and body on the window canvas like chat (06 · Settings/Account 16292:19723): the band shows the art, the body is the new chat's stage, flattened here for what covers it. |
| `--portal-page-sidebar-background` | `var(--portal-sidebar-background)` | ″ |  |
| `--portal-surface-collection` | `color-mix( in srgb, var(--portal-raw-pure-white) 2%, color-mix(in srgb, var(--portal-raw-black) 50%, color-mix(in srgb, var(--portal-raw-blue) 10%, transparent)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 70%, transparent)` | A collection's cards and table on a page body (Library 1163:18113, 1163:18155): white .02 / black .50 / blue .10 over the stage. |
| `--portal-surface-elevated` | `var(--ui-bg-elevated)` | ″ | Menus, popovers, dialogs, drawers, toasts: the solid floating fill (878:6044 #0D0D1E on dark). |
| `--portal-surface-card` | `var(--portal-raw-ink-2)` | ″ | Cards of grouped rows (1036:7576, 1038:7803). |
| `--portal-settings-card` | `color-mix( in srgb, var(--portal-raw-pure-white) 2%, color-mix(in srgb, var(--portal-raw-black) 50%, var(--portal-raw-brand-10)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 70%, transparent)` | Global Settings' cards lift off the page with no stroke (Card 16292:16456): white .02 / black .50 / blue .10 (#1A1A30 where solid); their dividers light/5. Section heads at .8 ink. |
| `--portal-settings-divider` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-settings-section-ink` | `color-mix(in srgb, var(--ui-base) 80%, transparent)` | ″ |  |
| `--portal-settings-link` | `var(--portal-raw-ink-5)` | ″ | The column's Agent settings card (16292:16428): light/5. |
| `--portal-surface-selected` | `var(--portal-raw-ink-5)` | ″ | Neutral selection: list rows, nav rows (1036:7409), with --portal-stroke-selected. |
| `--portal-surface-row-hover` | `var(--portal-raw-ink-3)` | ″ | A row under the pointer on a panel, the stage or a card: one ink step over its rest. |
| `--portal-surface-skeleton` | `var(--portal-raw-ink-8)` | ″ | Content still loading (skeleton bars, tiles and rows): one quiet ink block on any surface. |
| `--portal-surface-chrome` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 20%, transparent)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 20%, transparent)` | 30px chrome buttons and the session pill: the band's navy tile, white .05 / black .60 / blue .20 (16292:15285, 16292:15297), the glyph at .4; hover and pressed brighten a step each. |
| `--portal-surface-chrome-hover` | `color-mix( in srgb, var(--portal-raw-pure-white) 8%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 30%, transparent)) )` | `var(--portal-raw-ink-5)` |  |
| `--portal-surface-chrome-pressed` | `color-mix( in srgb, var(--portal-raw-pure-white) 10%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 40%, transparent)) )` | `var(--portal-raw-ink-10)` |  |
| `--portal-text-chrome` | `var(--portal-raw-ink-40)` | `color-mix(in srgb, var(--ui-base) 48%, transparent)` |  |
| `--portal-rail-fill` | `color-mix( in srgb, var(--portal-raw-pure-white) 5%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 20%, transparent)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 20%, transparent)` | The global rail's buttons (ico 16292:15253): a navy tile, white .05 / black .60 / blue .20, the glyph at .4; the open destination white .10 / black .60 / blue .40 (16292:15245) with its glyph in full ink. Hover takes the step between them. |
| `--portal-rail-fill-hover` | `color-mix( in srgb, var(--portal-raw-pure-white) 8%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 30%, transparent)) )` | `var(--portal-raw-ink-5)` |  |
| `--portal-rail-fill-selected` | `color-mix( in srgb, var(--portal-raw-pure-white) 10%, color-mix(in srgb, var(--portal-raw-black) 60%, color-mix(in srgb, var(--portal-raw-blue) 40%, transparent)) )` | `var(--portal-raw-ink-10)` |  |
| `--portal-rail-ink` | `var(--portal-raw-ink-40)` | `color-mix(in srgb, var(--ui-base) 48%, transparent)` |  |
| `--portal-rail-ink-selected` | `var(--portal-text-primary)` | ″ |  |
| `--portal-panel-action` | `var(--portal-raw-ink-5)` | `var(--portal-raw-ink-2)` | A list panel's start row (New agent, New routine) wears New chat's slab (16292:15511): light/5, no rim; dark/2 on light. |
| `--portal-panel-action-hover` | `var(--portal-raw-ink-8)` | `var(--portal-raw-ink-5)` |  |
| `--portal-surface-control` | `var(--portal-raw-ink-5)` | `var(--portal-raw-ink-3)` | Secondary and icon buttons (877:4189) at rest, hover and pressed: one ink step each. |
| `--portal-surface-control-hover` | `var(--portal-raw-ink-8)` | `var(--portal-raw-ink-5)` |  |
| `--portal-surface-control-pressed` | `var(--portal-raw-ink-10)` | `var(--portal-raw-ink-8)` |  |
| `--portal-surface-field` | `var(--portal-raw-ink-4)` | `var(--portal-raw-ink-2)` | Inputs, search, segmented tracks (895:9670). |
| `--portal-surface-select` | `var(--portal-raw-ink-8)` | `var(--portal-raw-ink-4)` | Select triggers (856:3456). |
| `--portal-surface-floating-hover` | `color-mix(in srgb, var(--ui-base) 6%, transparent)` | `var(--portal-raw-ink-5)` | A menu item under the pointer, on the elevated surface. |
| `--portal-surface-scrim` | `var(--portal-raw-shadow-50)` | `var(--portal-raw-shadow-20)` | The modal scrim: a plain dim, never a blur. |
| `--portal-surface-danger-solid` | `var(--portal-raw-danger)` | ″ | Danger, one family (1038:7836). The confirm of a destructive dialog (878:7066) is the solid slab with white ink (7.3:1); every other destructive mark (outline buttons, menu items, banners) is the 8% tint with a 50% stroke and the danger ink, which keeps 4.5:1 on that tint: #FF7373 on dark (≈7:1), the slab's own red on light (≈6:1). |
| `--portal-text-on-danger` | `var(--portal-raw-pure-white)` | ″ |  |
| `--portal-surface-danger` | `var(--portal-raw-danger-8)` | ″ |  |
| `--portal-surface-danger-hover` | `var(--portal-raw-danger-14)` | ″ |  |
| `--portal-surface-danger-pressed` | `var(--portal-raw-danger-20)` | ″ |  |
| `--portal-stroke-danger` | `var(--portal-raw-danger-50)` | ″ |  |
| `--portal-text-danger` | `var(--portal-raw-danger-bright)` | `var(--portal-raw-danger)` |  |
| `--portal-stroke-hairline` | `var(--portal-raw-ink-5)` | ″ | Strokes. Card edges, the panel-to-stage edge, page footer rules (1038:7803). |
| `--portal-stroke-divider` | `var(--portal-raw-ink-3)` | `var(--portal-raw-ink-5)` | Between rows inside a card and under the page tabs (877:4229, 1038:7499); light takes one step more to stay visible on #f2f2f2. |
| `--portal-stroke-edge` | `var(--portal-raw-ink-10)` | ″ | The chat sidebar's edge and header rule (Var1 147:7470: light/10). |
| `--portal-stroke-control` | `var(--portal-raw-ink-10)` | `var(--portal-raw-ink-14)` |  |
| `--portal-stroke-field` | `var(--portal-raw-ink-14)` | `color-mix(in srgb, var(--ui-base) 18%, transparent)` |  |
| `--portal-stroke-strong` | `var(--portal-raw-ink-28)` | ″ | Selects, checkbox and radio at rest. |
| `--portal-stroke-selected` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-stroke-floating` | `var(--portal-raw-ink-10)` | ″ |  |
| `--portal-status-online` | `var(--portal-raw-green-60)` | `var(--portal-raw-mint-deep)` | Status inks, one set for the whole redesign (sidebar machines, dots, chips, page status text): online is secondary/green at .6 (1089:8305, the newest sidebar), pending Figma's orange, failed its soft red; each passes 4.5:1 on its mode's panels. |
| `--portal-status-pending` | `var(--portal-raw-orange)` | `color-mix(in srgb, var(--portal-raw-orange) 70%, var(--portal-raw-black))` |  |
| `--portal-status-error-ink` | `var(--portal-raw-danger-soft)` | `var(--portal-raw-danger)` |  |
| `--portal-focus-ring-color` | `var(--portal-text-brand)` | ″ | The one focus ring's colour (geometry in shape.css). |
| `--portal-brand-fill` | `var(--portal-surface-brand)` | ″ | Names in use before the role names; Phase B moves their readers to the role and deletes them. |
| `--portal-on-brand` | `var(--portal-text-on-brand)` | ″ |  |
| `--portal-brand-ink` | `var(--portal-text-brand)` | ″ |  |
| `--portal-selection-background` | `var(--portal-surface-brand-selected)` | ″ |  |
| `--portal-neutral-selection-background` | `var(--portal-surface-selected)` | ″ |  |
| `--portal-neutral-selection-stroke` | `var(--portal-stroke-selected)` | ″ |  |
| `--portal-window-background` | `var(--portal-surface-canvas)` | ″ |  |
| `--portal-sidebar-panel-background` | `color-mix( in srgb, var(--portal-raw-pure-white) 2%, color-mix(in srgb, var(--portal-raw-black) 85%, color-mix(in srgb, var(--portal-raw-blue) 70%, transparent)) )` | `color-mix(in srgb, var(--portal-raw-pure-white) 98%, transparent)` | The chat sidebar, side tool panels and page navs (Panel 16292:18203): white .02 / black .85 / blue .70, about 96% opaque, so the art still reads through. |
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

#### Type (`type.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-font-display` | `'Oswald', var(--dt-font-sans)` | ″ | H1, panel, dialog and drawer titles, the wordmark. |
| `--portal-font-body` | `'Mona Sans', var(--dt-font-sans)` | ″ | Every other line of UI. |
| `--portal-font-mono` | `var(--dt-font-mono)` | ″ | Paths, ids, code and an agent's own words (the SOUL preview, 877:4245): the theme's mono, the bundled JetBrains Mono standing in for Aeonik Fono. |
| `--portal-type-display-xl-size` | `2.5rem` | ″ | Page H1, agent name (1036:7378). |
| `--portal-type-display-lg-size` | `1.875rem` | ″ | Panel title, sub-page H2, dialog and drawer title (1036:7476, 1038:7797). |
| `--portal-type-section-size` | `0.9375rem` | ″ | Section heading, stat value (1036:7481). |
| `--portal-type-body-size` | `0.875rem` | ″ | Row label, button, tab, input value (label 500, body 400). |
| `--portal-type-ui-size` | `0.875rem` | ″ | Nav row, sidebar row, menu item, description. Figma draws 13px (1036:7410); the designer asked for 14 after seeing it in the app (2026-11-13). |
| `--portal-type-meta-size` | `0.75rem` | ″ | Timestamps, row meta, the composer's model row (906:19738). |
| `--portal-type-micro-size` | `0.6875rem` | ″ | Nav counts, tags, keycaps, the "You" label (1036:7434). |
| `--portal-type-caption-size` | `0.625rem` | ″ | UPPERCASE group labels, table headers, stat labels, menu labels (1036:7408). |
| `--portal-type-chip-size` | `0.5625rem` | ″ | UPPERCASE DEFAULT / RUNNING / EMPTY chips (1036:7318). |
| `--portal-type-field-touch-min` | `16px` | ″ | The smallest text a field may use on a coarse pointer in the browser: iOS Safari zooms into any focused field under 16px (px, not rem: the zoom rule reads CSS pixels). |
| `--portal-settings-h1-line` | `3rem` | ″ | Global Settings (16292:16447): the H1 at Figma's 48px line and -1px tracking; row labels in the 16px body (Nous/Body 22.4 line); descriptions in the mono signal font. |
| `--portal-settings-h1-tracking` | `-1px` | ″ |  |
| `--portal-settings-row-label-size` | `1rem` | ″ |  |
| `--portal-settings-row-label-line` | `1.4` | ″ |  |
| `--portal-settings-section-line` | `1.25rem` | ″ | Section heads: Nous/Compact SemiBold on a 20px line, at the UI step (Figma's 13 drawn at 14). |
| `--portal-settings-meta-font` | `var(--portal-font-mono)` | ″ |  |
| `--portal-tracking-display` | `-0.01em` | ″ | Tracking per role: display titles tighten, small UI opens up, captions and chips open most. |
| `--portal-tracking-ui` | `0.01em` | ″ |  |
| `--portal-tracking-chip` | `0.04em` | ″ |  |
| `--portal-tracking-caption` | `0.06em` | ″ |  |

#### Space and layout (`space.css`)

| Token | Dark | Light | Note |
|---|---|---|---|
| `--portal-rail-width` | `3.75rem` | ″ | The global rail and the developer rail (1036:7202). |
| `--portal-rail-reserve` | `0px` | ″ | What the rail takes from the layout: its width, none on compact, where it rides the sheet. |
| `--portal-rail-top` | `5.625rem` | ″ | The global rail ("Prod - Desktop" rail 16292:15233): no brand badge (its layer is hidden in every row), the destinations from 90px down on every host, clear of macOS traffic lights in the column; 30px buttons (the chrome row, the touch target on a coarse pointer) with 16px glyphs, 20px apart; the foot's buttons carry 20px glyphs, 40px off the window floor. |
| `--portal-rail-button` | `var(--portal-row-chrome)` | ″ |  |
| `--portal-rail-glyph` | `1rem` | ″ |  |
| `--portal-rail-glyph-foot` | `1.25rem` | ″ |  |
| `--portal-rail-gap` | `1.25rem` | ″ |  |
| `--portal-rail-bottom` | `2.5rem` | ″ |  |
| `--portal-rail-popover-offset` | `calc( (var(--portal-rail-width) - var(--portal-rail-button)) / 2 + var(--portal-space-tight) )` | ″ | A rail menu or popover opens one tight gap past the rail's edge, whatever the rail button's size (30px, 44px on touch): the distance from the button's end to the rail's, plus that gap. |
| `--portal-panel-reveal-width` | `2.75rem` | ″ | The way back to a folded sidebar or page panel in Simple mode (1061:7816): a 44×40 floating button at the stage's top-leading corner, clear of the band. Advanced keeps it in the band. |
| `--portal-panel-reveal-height` | `2.5rem` | ″ |  |
| `--portal-panel-reveal-inset` | `1rem` | ″ |  |
| `--portal-panel-reveal-glyph` | `1.125rem` | ″ |  |
| `--portal-breakpoint-medium` | `640px` | ″ | The responsive model (app/redesign/breakpoints.ts, DESIGN.md › Responsive): compact below the medium breakpoint, medium below the wide one, wide from there. A custom property cannot feed a media query, so every redesign query repeats the literal (max-width: 639.98px / 1023.98px) with a comment naming the class; these two document it. |
| `--portal-breakpoint-wide` | `1024px` | ″ |  |
| `--portal-sheet-width` | `min(var(--sessions-pane-width), 100vw - 3.5rem)` | ″ | Below wide the chat sidebar and page panels are sheets over the stage: the sidebar's width, leaving a 56px strip of scrim to tap. |
| `--portal-sheet-panel-width` | `calc(var(--portal-sheet-width) - var(--portal-rail-width) + var(--portal-rail-reserve))` | ″ | The sheet's panel past the rail: the whole sheet on medium (the rail stays in the layout), the sheet less the rail on compact (the rail rides the sheet). |
| `--portal-sheet-panel-max-width` | `calc(100vw - 3.5rem - var(--portal-rail-width))` | ″ | The widest a sheet's panel may be: from the rail's edge to the scrim strip. A plugin pane's sheet keeps the width the pane declares, up to this. |
| `--portal-touch-target` | `2.75rem` | ″ | The smallest hit area on a coarse pointer (WCAG 2.5.5); a control's visual may stay smaller. |
| `--portal-space-compact` | `0.75rem` | ″ | A phone's tighter rhythm (compact and short windows): the composer's padding and gaps, its dock's floor, and on touch its control rows, whose 44px targets overlap the padding. |
| `--portal-space-tight` | `0.5rem` | ″ |  |
| `--portal-composer-row-touch` | `1.75rem` | ″ |  |
| `--portal-safe-top` | `env(safe-area-inset-top, 0px)` | ″ | The browser's safe area (notch, home indicator) once the page is laid out under it (viewport-fit=cover, app/redesign/browser-chrome.ts); zero on every other host. The app root is inset by it as a whole (portal-shell.css), so only body-portalled surfaces (dialogs) read these. |
| `--portal-safe-bottom` | `env(safe-area-inset-bottom, 0px)` | ″ |  |
| `--portal-safe-start` | `env(safe-area-inset-left, 0px)` | ″ |  |
| `--portal-safe-end` | `env(safe-area-inset-right, 0px)` | ″ |  |
| `--portal-panel-width` | `var(--portal-sidebar-width, var(--sessions-pane-width))` | ″ | ONE sidebar column for every rail destination: page panels (Agents, Routines, Settings) take the chat sidebar's width — the user's sash width once they size it (published by the pane tree as --portal-sidebar-width), else its default — so switching destinations never moves the panel edge, its divider or the stage's content edge. |
| `--portal-frame-inset` | `1rem` | ″ | The page-frame contract: where a page's content column starts, from the stage edge it shares with the page's sidebar panel (the freshest frames, 1036:7189 / 895:*: 20px after the divider, one value for Agents, Settings and Routines alike), and from the stage's own edge on a page with no panel (919:9281: x 90 in the stage). Left insets, never centring, at every window width, so switching pages never moves the column sideways. |
| `--portal-stage-inset` | `1rem` | ″ |  |
| `--portal-workspace-inset-start` | `0.3125rem` | ″ | The workspace frame's gaps, where the artwork shows, measured in Figma's 1280×880 window (Content 16292:14463 at x 65–1265, y 60–865): 5px past a 60px rail (and before the developer rail), 15px to the window's end (the "inset" column 18521:22266 and the frame's 5px edge) and to its floor (the "Spacer" 18521:17513). 0 for all three puts the panels back on the edges. |
| `--portal-workspace-inset-end` | `0.9375rem` | ″ |  |
| `--portal-workspace-inset-bottom` | `0.9375rem` | ″ |  |
| `--portal-column-form` | `52.5rem` | ″ | The column's one max width per page kind: forms and details (1036:7189 General: 840), catalogs and tables (919:9281: 980), and the conversation shared by the transcript and the docked composer: both 440 wide, the transcript flush with the composer's edges (chat 16292:16175). |
| `--portal-column-catalog` | `61.25rem` | ″ |  |
| `--portal-column-conversation` | `27.5rem` | ″ |  |
| `--portal-column-conversation-inset` | `0px` | ″ |  |
| `--portal-column-landing` | `27.5rem` | ″ | The landing's composer before the first message, as wide as a conversation's (16292:14822). |
| `--portal-tool-panel-width` | `26.25rem` | ″ | The developer tool panel's default width (Files, Terminal, Review and Browser share one width, redesign/developer-tool-panel.ts): room for an 80-column terminal beside the conversation. |
| `--page-content-inset` | `1rem` | ″ | Full-width pages without a panel pad their catalog column by this much on each side. |
| `--portal-scroll-edge-fade` | `1.5rem` | ″ | How far a sideways scroller's trailing edge fades while more follows (portal-pages.css). |
| `--page-content-max-width` | `calc(var(--portal-column-catalog) + 2 * var(--page-content-inset))` | ″ |  |
| `--portal-page-end-space` | `2rem` | ″ | The space a page scroller keeps below its last row, under its bottom fade. |
| `--portal-page-top` | `1rem` | ″ | The page header (PageHeader, app/redesign/page-header.tsx), one geometry for every routed page so its H1, description and tab bar sit at the same y on all of them: the column's top (856:2827 y 18), a title block that always reserves the agent header's 68 (877:4166: 56 tile, or H1 44 + 6 + one 13px line), 24 to the tabs and from the tabs to the content (877:4166 gap 24), 32h tabs (877:4200) and the page rhythm of 32 between sections and after a heading with no tabs (1036:7405). |
| `--portal-page-title-block` | `4.25rem` | ″ |  |
| `--portal-page-header-gap` | `1.5rem` | ″ |  |
| `--portal-page-rhythm` | `2rem` | ″ |  |
| `--portal-page-tab-height` | `2rem` | ″ |  |
| `--portal-page-tab-inset` | `0.75rem` | ″ |  |
| `--portal-panel-title-top` | `1.875rem` | ″ | A page's list panel (Agents, Routines) shares the chat sidebar's and the Settings column's header (Top 16292:15498, 16292:16360): its title 30px in and 30px down on a 36px line, its first control 122px under the panel's top, where New chat and the Settings rows start. |
| `--portal-panel-title-inset` | `1.875rem` | ″ |  |
| `--portal-panel-title-line` | `2.25rem` | ″ |  |
| `--portal-panel-first-row` | `7.625rem` | ″ |  |
| `--portal-settings-column` | `40rem` | ″ | Global Settings ("Prod - Desktop" 06 · Settings/Account, Panel/flexible 16292:16447): a 640px column centred in the stage, its header 104px under the stage's top; the header, section heads and card rows share a 32px text inset; cards pad 16px over and under their rows, rows 12px. |
| `--portal-settings-top` | `var(--portal-page-top)` | ″ |  |
| `--portal-settings-inset` | `1rem` | ″ |  |
| `--portal-settings-card-pad` | `1rem` | ″ |  |
| `--portal-settings-row-pad` | `0.75rem` | ″ |  |
| `--portal-settings-nav-inset` | `1.875rem` | ″ | The column's panel (Panel 16292:16359): the title on a 30px inset, 30px down, 20px over 40px rows 4px apart; the Agent settings card 20px under them (16292:16428). |
| `--portal-settings-nav-gap` | `0.25rem` | ″ |  |
| `--portal-settings-related-gap` | `1.25rem` | ″ |  |
| `--portal-settings-link-width` | `17.125rem` | ″ |  |
| `--portal-settings-nav-title-line` | `2.25rem` | ″ | Its title is Nous/Identity: 30px on a 36px line (16292:16363). The rows start 122px down: Figma's "// META INFO" caption under the title is placeholder copy and not built, its room is kept. |
| `--portal-settings-nav-head-gap` | `3.5rem` | ″ |  |
| `--portal-dialog-action-height` | `2.75rem` | ″ | The actions of a dialog or drawer footer (1038:7792): taller than the page's 36px buttons. |
| `--portal-session-tab-inset` | `1.5625rem` | ″ | The band's title pill starts 25px past the stage's leading edge (tab 16292:15297 at x 425, the stage at 400), after the Back/Forward group. |
| `--portal-band-trail-inset` | `0.8125rem` | ″ | The band's trailing Agent settings ends 15px inside the workspace frame (settings-2 16292:15303): the trailing cluster stands 12px off the window, the frame 10px, so the button steps 13px in. |
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
| `--portal-radius-tile` | `12px` | ″ | Media cards and the collections they sit beside (Library 1163:18113: 12px; its table draws 15, held to the card's step). |
| `--portal-radius-surface` | `6px` | ″ | Chat content (bubble, tool row, code card) and chat-sidebar rows (491:40655, 147:7470). |
| `--portal-radius-floating` | `8px` | ″ | Popovers, dialogs, choice tiles (616:1685). |
| `--portal-radius-panel` | `9px` | ″ | The outer top corner of panels, the stage and the composer (906:19804). |
| `--portal-radius-frame` | `15px` | ″ | The workspace frame: the one card the chat sidebar and the stage (or a page's nav and body) sit in, inset from the window (Content 16292:14463). Global Settings' cards round the same (Card 16292:16456). |
| `--portal-radius-settings-card` | `var(--portal-radius-frame)` | ″ |  |
| `--portal-radius-settings-link` | `9px` | ″ | The Settings column's Agent settings card (16292:16428). |
| `--portal-shadow-settings-link` | `0 1px 4px var(--portal-raw-shadow-20)` | ″ |  |
| `--portal-radius-round` | `999px` | ″ | Switch tracks, swatches, dots. |
| `--portal-focus-ring-width` | `2px` | ″ | The one focus ring: 2px of brand ink as an outline, so it never replaces a control's own stroke shadow. Outset by default; a context whose scroller or overflow would clip an outset ring (sidebar, page nav) sets --portal-focus-ring-offset to the inset value on its container and every control inside follows. |
| `--portal-focus-ring-outset` | `1px` | ″ |  |
| `--portal-focus-ring-inset` | `calc(-1 * var(--portal-focus-ring-width))` | ″ |  |
| `--portal-focus-ring-offset` | `var(--portal-focus-ring-outset)` | ″ |  |
| `--portal-shadow-floating` | `0 8px 24px var(--portal-raw-shadow-50)` | `0 1px 2px var(--portal-raw-shadow-4), 0 8px 24px var(--portal-raw-shadow-8)` | Menus, popovers, dialogs (878:6044): one soft drop shadow. |
| `--portal-shadow-composer` | `none` | ″ | The composer card casts none (msg 16292:15670). |
| `--portal-shadow-workspace` | `0 -4px 30px var(--portal-raw-shadow-20)` | `0 8px 12px var(--portal-raw-shadow-navy-10)` | The workspace frame lifted off the artwork (16292:14463): a soft shadow up and around it. |
| `--portal-shadow-workspace-highlight` | `inset 0 1px 2px var(--portal-raw-highlight-5)` | `none` | ...and its inner highlight along the top (inner shadow, 16292:14463): white .05, 1px down, 2px soft. |
| `--portal-control-radius` | `var(--portal-radius-control)` | ″ | Names in use before the role names; Phase B moves their readers to the role and deletes them. |
| `--portal-menu-radius` | `var(--portal-radius-card)` | ″ |  |
| `--portal-panel-radius` | `var(--portal-radius-frame)` | ″ | The panel row's outer corners are the workspace frame's. |
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
| `--ui-bg-card` | `var(--portal-surface-card)` | ″ |  |
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
| `--dialog-body-gap` | `1.5rem` | ″ |  |
| `--titlebar-control-size` | `var(--portal-row-chrome)` | ″ | Figma's 30×30 chrome buttons (r3, 16px glyphs): titlebar, rails, pane headers; breadcrumb spacing 906:19467. |
| `--titlebar-control-height` | `var(--portal-row-chrome)` | ″ |  |
| `--titlebar-icon-size` | `1rem` | ″ |  |
| `--titlebar-control-gap` | `0.625rem` | ″ |  |
| `--ui-green` | `var(--portal-status-online)` | ″ | One status family: shared components' success inks read the redesign's online green. |

#### Chat sidebar (`sidebar.css`): the Variation 1 variant's own roles

| Token | Dark | Light | Note |
|---|---|---|---|
| `--sessions-pane-width` | `21.25rem` | ″ |  |
| `--sessions-pane-min-width` | `17.5rem` | ″ | Below this the New-chat split and the agent cards start to truncate (sash drags); the sash reaches 40px past the default. |
| `--sessions-pane-max-width` | `23.75rem` | ″ |  |
| `--sessions-pane-collapse-width` | `calc(var(--sessions-pane-min-width) - 4.5rem)` | ″ | Dragging the sash this far under the minimum folds the sidebar away (1054:22350), as ⌘B does; a drag that stops short of it rests at the minimum. Classic has no such width, so it never folds. |
| `--portal-sidebar-head-top` | `1.875rem` | ″ | The header (Top 16292:15498): the wordmark and its caption on a 30px inset, 30px under the panel's top edge, the search button 28px from its end edge; 30px over New chat (its own 20 plus the list's 10px top pad). |
| `--portal-sidebar-head-bottom` | `1.875rem` | ″ |  |
| `--portal-sidebar-head-start` | `1.875rem` | ″ |  |
| `--portal-sidebar-head-end` | `1.75rem` | ″ |  |
| `--portal-sidebar-wordmark-size` | `1.875rem` | ″ |  |
| `--portal-sidebar-wordmark-line` | `2.25rem` | ″ |  |
| `--portal-sidebar-wordmark-tracking` | `-1px` | ″ |  |
| `--portal-sidebar-wordmark-gap` | `0.625rem` | ″ |  |
| `--portal-sidebar-inset` | `2rem` | ″ | The list (Centre 16292:15509): section captions and row text on a 32px inset; New chat and row fills bleed 12px past it (the 300px New chat in a 340px panel); 16px between the actions and the first section. |
| `--portal-sidebar-bleed` | `0.75rem` | ″ |  |
| `--portal-sidebar-gutter` | `calc(var(--portal-sidebar-inset) - var(--portal-sidebar-bleed))` | ″ |  |
| `--portal-sidebar-actions-gap` | `1rem` | ″ |  |
| `--portal-sidebar-row` | `2rem` | ″ | A caption sits 10px under what precedes it and 16px over its first row's text; rows are 20px of text 16px apart (36 pitch, 04 · Workspace/Agent); an agent's chats are 40px (16292:15602). |
| `--portal-sidebar-pin-row` | `2.25rem` | ″ |  |
| `--portal-sidebar-machine-row` | `2.25rem` | ″ |  |
| `--portal-sidebar-chat-row` | `2.5rem` | ″ |  |
| `--portal-sidebar-row-gap` | `0px` | ″ |  |
| `--portal-sidebar-section-gap` | `0.5rem` | ″ |  |
| `--portal-sidebar-label-top` | `0.625rem` | ″ |  |
| `--portal-sidebar-label-bottom` | `0.5rem` | ″ |  |
| `--portal-sidebar-dot` | `0.3125rem` | ″ | The 5px leading dot, 12px before the title (16px past its box in Figma's item). |
| `--portal-sidebar-dot-gap` | `0.75rem` | ″ |  |
| `--portal-sidebar-caption` | `0.625rem` | ″ | Tags, ages and statuses: Figma's Nous/Uppercase, the mono signal font at 10px, 0.6px tracking. |
| `--portal-sidebar-caption-tracking` | `0.06em` | ″ |  |
| `--portal-sidebar-meta-font` | `var(--portal-font-mono)` | ″ |  |
| `--portal-sidebar-label-font` | `var(--portal-font-mono)` | ″ | Section captions: Nous/Metadata, the mono signal font in sentence case (16292:15526); 14px where Figma draws 13, as every 13px UI step (type.css › --portal-type-ui-size). |
| `--portal-sidebar-label-size` | `var(--portal-type-ui-size)` | ″ |  |
| `--portal-sidebar-label-line` | `1rem` | ″ |  |
| `--portal-sidebar-label-glyph` | `1rem` | ″ |  |
| `--portal-sidebar-row-weight` | `600` | ″ | Row titles: Nous/Compact SemiBold at 20px line (16292:15538). |
| `--portal-sidebar-row-line` | `1.25rem` | ″ |  |
| `--portal-sidebar-hover` | `var(--portal-raw-ink-4)` | ″ |  |
| `--portal-sidebar-press` | `color-mix(in srgb, var(--ui-base) 7%, transparent)` | ″ |  |
| `--portal-sidebar-row-radius` | `var(--portal-radius-control)` | ″ | Rows carry no fill for the open chat (16292:15602): ink tells it, full for the open chat and the flat rows, the floor for an agent's other chats (Figma's .4 fails AA). A row under the pointer takes the hover fill. |
| `--portal-sidebar-row-ink` | `var(--ui-text-primary)` | ″ |  |
| `--portal-sidebar-chat-ink` | `var(--ui-text-quaternary)` | ″ |  |
| `--portal-sidebar-row-selected-ink` | `var(--ui-text-primary)` | ″ |  |
| `--portal-sidebar-card` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-sidebar-action` | `var(--portal-raw-ink-5)` | ″ | New chat (16292:15511): 40px, radius 3, light/5 (dark/2 on light), 12px sides, 10px gap, a 16px glyph; rimless light/5 keycaps (16292:15520) in the mono font at 11px, 60% ink. The search button (16292:15507) is a 30px square on the same fill. |
| `--portal-sidebar-action-key` | `var(--portal-raw-ink-5)` | ″ |  |
| `--portal-sidebar-action-rim` | `var(--portal-raw-ink-10)` | ″ |  |
| `--portal-sidebar-action-pad` | `0.75rem` | ″ |  |
| `--portal-sidebar-action-gap` | `0.625rem` | ″ |  |
| `--portal-sidebar-action-weight` | `500` | ″ |  |
| `--portal-sidebar-key-font` | `var(--portal-font-mono)` | ″ |  |
| `--portal-sidebar-key-ink` | `var(--ui-text-tertiary)` | ″ |  |
| `--portal-sidebar-search-button` | `var(--portal-row-chrome)` | ″ |  |
| `--portal-sidebar-machine-fill` | `var(--portal-raw-ink-2)` | ″ | An open machine is one card (set 16292:15581): radius 12, light/2, 5px in from the rows on each side, its header 16px over and under its name; its agents 6px inside it. |
| `--portal-sidebar-machine-radius` | `12px` | ″ |  |
| `--portal-sidebar-machine-pad` | `0.3125rem` | ″ |  |
| `--portal-sidebar-machine-head` | `3.25rem` | ″ |  |
| `--portal-sidebar-machine-ink` | `color-mix(in srgb, var(--ui-base) 80%, transparent)` | ″ |  |
| `--portal-sidebar-agent-inset` | `0.375rem` | ″ | Agent cards (16292:15592): radius 9, 6px leading pad (12 to the name past the 32px tile), 16px trailing, 12px over and under, a soft drop shadow. The open agent is the brand slab, white .10 / black .40 / blue (#2E2EA8); the others white .10 / black .60 / blue .30 (16292:15630). Both carry the CRT texture at 5% colour-dodge. Light draws both as blue .10 on white .80 under a 1px brand rule, the open one solid brand, the others brand .10 (16292:25188, 16292:25226), the texture at 90%. |
| `--portal-sidebar-agent-radius` | `9px` | ″ |  |
| `--portal-sidebar-agent-pad-block` | `0.75rem` | ″ |  |
| `--portal-sidebar-agent-pad-start` | `0.375rem` | ″ |  |
| `--portal-sidebar-agent-pad-end` | `1rem` | ″ |  |
| `--portal-sidebar-agent-gap` | `0.75rem` | ″ |  |
| `--portal-sidebar-agent-avatar` | `2rem` | ″ |  |
| `--portal-sidebar-agent-avatar-radius` | `var(--portal-radius-surface)` | ″ |  |
| `--portal-sidebar-agent-shadow` | `0 1px 4px var(--portal-raw-shadow-20)` | `0 0 transparent` |  |
| `--portal-sidebar-agent-fill` | `color-mix( in srgb, var(--portal-raw-pure-white) 10%, color-mix(in srgb, var(--portal-raw-black) 60%, var(--portal-raw-brand-30)) )` | `color-mix( in srgb, var(--portal-raw-blue) 10%, color-mix(in srgb, var(--portal-raw-pure-white) 80%, transparent) )` |  |
| `--portal-sidebar-agent-fill-open` | `color-mix( in srgb, var(--portal-raw-pure-white) 10%, color-mix(in srgb, var(--portal-raw-black) 40%, var(--portal-raw-blue)) )` | `var(--portal-sidebar-agent-fill)` |  |
| `--portal-sidebar-agent-rule` | `transparent` | `var(--portal-raw-brand-10)` |  |
| `--portal-sidebar-agent-rule-open` | `transparent` | `var(--portal-raw-blue)` |  |
| `--portal-sidebar-agent-hover` | `var(--portal-raw-ink-4)` | ″ |  |
| `--portal-sidebar-agent-texture-opacity` | `0.05` | `0.9` |  |
| `--portal-sidebar-agent-subtitle` | `var(--ui-text-quaternary)` | ″ |  |
| `--portal-sidebar-agent-subtitle-open` | `var(--ui-text-secondary)` | `var(--ui-text-quaternary)` | On the open agent's brand slab the floor needs .7 to hold 4.5:1. |
| `--portal-sidebar-chats-pad` | `0.375rem` | ″ | An agent's chats (list 16292:15601): 6px over and under, the dot 30px into the card, the title 22px after it, the age 20px from the card's end. |
| `--portal-sidebar-chat-start` | `1.875rem` | ″ |  |
| `--portal-sidebar-chat-end` | `1.25rem` | ″ |  |
| `--portal-sidebar-chat-dot-gap` | `1.375rem` | ″ |  |
| `--portal-sidebar-chat-title` | `calc( var(--portal-sidebar-chat-start) + var(--portal-sidebar-dot) + var(--portal-sidebar-chat-dot-gap) )` | ″ |  |
| `--portal-sidebar-dot-ink` | `var(--ui-text-quaternary)` | ″ | A settled row's dot: the row's ink, quieter than its title; the open chat's in full ink. |
| `--portal-sidebar-label` | `var(--ui-text-quaternary)` | ″ | Captions, tags, subtitles and ages take the ladder's floor where Figma's .3-.4 fail AA. |
| `--portal-sidebar-meta` | `var(--ui-text-quaternary)` | ″ |  |
| `--portal-sidebar-green` | `var(--portal-status-online)` | ″ |  |
| `--portal-sidebar-orange` | `var(--portal-status-pending)` | ″ |  |
| `--portal-sidebar-count-fill` | `var(--portal-raw-brand-25)` | `var(--portal-raw-brand-10)` | The Agents count: a brand-tinted pill (147:6260), lighter on light. |
| `--portal-sidebar-count-radius` | `10px` | ″ |  |

<!-- design-tokens:end -->

### Page header (`app/redesign/page-header.tsx`)

Every routed page with a title or tabs renders `PageHeader` (Agents and an agent's Overview /
History / Settings, Capabilities, Library, Messaging, Routines, Spawn tree, Webhooks, every global
Settings page), so the H1, its description, the actions and the tab bar sit at one place on all of
them. The geometry is `--portal-page-*` in `tokens/space.css`: the column top 18 (16 on compact), an
H1 row (H1, actions at its end), the description 6 under it spanning the column, an optional leading
tile (the agent avatar) beside both; with tabs the title block always reserves 68 (877:4166), so the
tab bar is at one y whether a page has a description or not, then 24 to the tabs, 32h tabs and 24 to
the content; without tabs the content follows after the page rhythm, 32. `PageTabs` is a WAI-ARIA
tab list (arrows, Home, End; roving focus) whose tabs all take the widest tab's width, and a count
reserves three digits while it loads, so switching tabs or a count arriving never moves a neighbour.
On compact the actions take their own row and the tabs keep their own widths in one row that scrolls sideways when it must (44px on touch).
A page that loads in fills its first paint with placeholders shaped like the loaded content
(`SettingsCardSkeleton`: the card's 64px rows and 48px add row), never arbitrary spacers.

### Routines (`app/routines/`)

The rail's Routines destination follows Figma "Routines — proposal + mock" (1081/1099/1100) and the
board's decisions of Nov 12: one list, two doors. The rail page lists every agent's routines on this
machine (the shared `$cronJobs` snapshot, which the redesign's sidebar sync keeps at every profile)
with **quick actions only**: Run now (Retry on a failure), Pause/Resume, Rename, Delete, and "Edit in
<agent> settings", which opens that routine, editor open, inside Agent › Settings › Routines
(`?agentProfile=…&tab=agent:schedules&routine=<id>&edit=1`). That agent page is the same list filtered
to the agent, read from its own machine, where the one editor changes what a routine does. Both doors
share the rows, detail and editor (`RoutineSource` is the difference); classic keeps `CronView` and
`AgentSchedulesSettings`. The rail page's panel and stage use the shared page-panel frame (panel at the
rail's edge, `--page-nav-width`; H1 where Settings puts it). The panel holds New routine (creation
opens "What kind of routine?" with an agent select), the overview row, and NEEDS ATTENTION / SCHEDULED
/ PAUSED (/ FINISHED) groups with a status dot and the agent's name as a tag. The overview is the
`PageHeader`, search and a status segment, a danger-tint banner per routine that needs attention (its
streak, its first plain error sentence, when it tries again, Retry) and a Routine · Agent · When · Last
run · Status table; on compact the schedule moves under the name. A routine needs attention when its
newest run failed or did not deliver, or its slot sits overdue past the scheduler's grace (a run in
flight supersedes the result); the rail button then carries the error-ink dot. The detail keeps the H1
at the page's y (the breadcrumb rides the top band); on the rail it shows what it does, when and where
read-only with an "Edit in <agent>" card, and Runs (each opens as a chat via `openCronRun`; chips only
where the backend recorded an outcome). In the agent's settings each row has Change, plus Advanced
(model, missed runs, run limit, remember last run; developer mode adds expression, script, id) and
Delete. "What kind of routine?" offers the agent's ready-made catalog, Ask in chat (a new chat with the
agent, its composer seeded with a starter sentence) and Start blank. When to run composes the
scheduler's own strings and reads them back in words. Send results to lists the agent's chat and its
connected channels; nothing checked is today's save-only `local`. In the chat sidebar each agent with
routines gets a Routines chip under its chats (count, error dot when one needs attention; 1100:8452)
that opens the Routines page searched to that agent; runs never join Chats. The sidebar's scheduled-jobs
section and the statusbar's Cron item are classic only.

### The chat sidebar is a variant

The chat sidebar (`app/chat/sidebar/portal/**`) is matched to "Prod - Desktop" (Panel 16292:15497).
Where it draws a state differently from the page language (row hover ink 4, the open chat marked by ink
alone, mono sentence-case captions, the r10 brand count pill, rimless mono keycaps, agent cards, its
inset focus rings), the sidebar keeps those values as its own named roles in `tokens/sidebar.css`
(`--portal-sidebar-*`) instead of the page roles. That is deliberate, not drift: change the sidebar there, and never point
page components at `--portal-sidebar-*`.

### Still to unify

Phase B unified cards (U1: 2% fill, 5% hairline, r4), page rhythm (U2: 32), page tabs (U3), in-card
dividers (U4: ink 3, light ink 5), row hover outside the sidebar (U5), danger (U10), display-lg (U11),
the text ladder outside the sidebar (U12), dialogs and drawers (U13: 24 padding and gap, 44px footer
actions), page focus rings (U16), brand selection (U17: 25% / light 12%) and control and floating
states (U18). What remains, one reviewable change per row:

| # | Role | Today: value · where | Winner (Figma) |
|---|---|---|---|
| U6 | Neutral selection | the pane tab-strip's active tab (ink 5, no stroke) | ink 5 + 0.5px ink 5 stroke (1036:7409) |
| U7 | Caption | new-session picker label 0.08em/500/brand ink; gateway menu `tracking-[0.08em]` | 10/1.4, `--portal-tracking-caption` 0.06em, 400, tertiary (1036:7408) |
| U8 | Chip and Count | new-session tag 0.06em/500/r3; RUNNING/EMPTY chip line boxes | Chip h14 9/0.04em r2, Count h14 10/500 r2 (1036:7318, 856:3474) |
| U9 | Keycap | New agent row keycaps (0.5px ink 5 rim, ink 5 fill, `portal-pages.css`) vs `[data-slot=kbd]` field ink 4 + 1px control stroke | ink 4 fill, 1px ink 14, r3, 11px (1038:7401) |
| U14 | Small text in mixed TSX | 14 sizes between 0.6 and 0.82rem and 0.07–0.2em tracking in the classic/portal branches | caption / micro / meta / ui sizes |
| U15 | Radius by role | `0.5rem` / `0.375rem` (agent overview menu and items), Tailwind `rounded-*` in redesign TSX (G2 warnings) | the seven `--portal-radius-*` roles |
| U19 | Units | px font sizes, line heights and radii in `agent-settings.css`, `history.css`, capabilities `page.css` (the guard's baseline lists them) | rem through the type tokens |
| U20 | Spacing drift | 3/5/7/9/11/13/15/17/22/23px insets (catalog, history, settings editor) | the 4px scale; Figma optical offsets keep `/* token-exempt */` |

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
4. If the change unifies a "Still to unify" row, delete the legacy token and its row, and
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
- **Shell wording** — the base catalog is classic's wording. Wording that differs only
  in the redesigned shell goes in `src/i18n/redesign/<locale>.ts` for every locale
  (see Terminology above), never in the base catalog and never as a `portal ? a : b`
  string at the call site.

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
