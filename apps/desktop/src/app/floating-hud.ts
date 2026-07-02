// Shared chrome for the top-center floating HUDs (command palette + session
// switcher). They pin just under the title bar, centered, and lean on a crisp
// border + shadow to separate from the app — no dimming/blurring backdrop.
// Each caller layers on its own z-index, width, and overflow.
export const HUD_POSITION = 'fixed left-1/2 top-3 -translate-x-1/2'

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
