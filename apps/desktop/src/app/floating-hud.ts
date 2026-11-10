// Shared chrome for the top-center floating HUDs (command palette + session
// switcher). They pin just under the title bar, centered, and lean on a crisp
// border + shadow to separate from the app — no dimming/blurring backdrop.
// Each caller layers on its own z-index, width, and overflow.
//
// Narrow screens: the centered HUD widens toward full-width and its top-left
// corner slides under the macOS traffic lights. Below ~44rem (where the overlap
// begins) drop the whole surface beneath the titlebar band so the search row
// always clears the window controls. These HUDs portal to <body>, outside the
// app-shell subtree that defines --titlebar-height, so the var needs a fallback.
export const HUD_POSITION =
  'fixed left-1/2 top-3 -translate-x-1/2 max-[44rem]:top-[calc(var(--titlebar-height,34px)+0.375rem)]'

// Matches the app's borderless-overlay surface (dialog, keybind panel, …):
// hairline `--stroke-nous` paired with the soft `--shadow-nous` float.
// `no-drag`: these HUDs overlap the titlebar's `[-webkit-app-region:drag]` band
// (app-shell.tsx), which wins hit-testing over DOM regardless of z-index — so
// without it the top of the surface (the search input) swallows clicks.
export const HUD_SURFACE =
  'rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous [-webkit-app-region:no-drag]'

// One row/text size for both HUDs (compact — two notches under `text-sm`).
export const HUD_TEXT = 'text-xs'

// Shared item layout + padding for both HUDs. Tight vertical rhythm so rows
// don't feel chunky; overrides the shadcn `CommandItem` default (`px-2 py-1.5`).
export const HUD_ITEM = 'gap-2 px-2 py-1'

// Section headings styled like the sidebar panel labels: brand-tinted, uppercase,
// tightly tracked — plain text, no sticky chrome bar. Targets the cmdk group
// heading via the universal-descendant variant.
export const HUD_HEADING =
  '**:[[cmdk-group-heading]]:static **:[[cmdk-group-heading]]:bg-transparent **:[[cmdk-group-heading]]:px-2.5 **:[[cmdk-group-heading]]:pb-1 **:[[cmdk-group-heading]]:pt-2.5 **:[[cmdk-group-heading]]:text-[0.64rem] **:[[cmdk-group-heading]]:font-semibold **:[[cmdk-group-heading]]:uppercase **:[[cmdk-group-heading]]:tracking-[0.16em] **:[[cmdk-group-heading]]:text-(--theme-primary)'

// A short note trailing a row's label — a version, a count, a live state. Sits
// closer than the row's icon-to-label `gap-2` because it reads as a suffix of
// the label, not as another item in the row.
export const HUD_NOTE = '-ml-1'

// `state` marks a note the row will CHANGE (a toggle's on/off) rather than a
// passive fact. It keeps the label's own color so the pair reads as one phrase,
// and earns its separation from a faint underline instead of going muted.
//
// That underline needs room, so `state` skips `truncate`: flex items are
// blockified, which makes its `overflow: hidden` bite, and the app's global
// 0.25rem underline offset (styles.css) already sits at the bottom edge of a
// `text-xs` line box — together they clip the rule away entirely. Pulling the
// offset to 2px keeps it inside the box; dropping truncate costs nothing
// because these notes are one word.
export const HUD_NOTE_VARIANT = {
  muted: 'truncate text-muted-foreground/80',
  state: 'shrink-0 underline decoration-current/50 underline-offset-2'
} as const
