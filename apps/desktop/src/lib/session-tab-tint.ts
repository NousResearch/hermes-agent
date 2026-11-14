import type { CSSProperties } from 'react'

/**
 * A session TAB's colour, painted from the ONE colour the app already resolves
 * for a session (explicit override → group/tile tier → inherited project).
 *
 * The tab reads its surface through the CSS custom properties `PaneTab`
 * documents (`--pane-tab-strip-bg` for the idle surface, `--pane-tab-active-bg`
 * for the active one, `--pane-tab-active-accent` for the active tab's seam), so
 * a coloured tab needs no component surgery: the vars are inherited from a
 * wrapper and each tab state resolves its own. Two mixes, both capped: the tab
 * sits in a strip of eight, and a tint strong enough to name one tab is a tint
 * that turns the strip into confetti.
 */
export const TAB_TINT_IDLE_MIX = 12
export const TAB_TINT_ACTIVE_MIX = 20

/**
 * `#rgb`, `#rrggbb`, `#rrggbbaa` only. Every caller-supplied colour reaches
 * `color-mix()` through an inline custom property, and a malformed value there
 * makes the WHOLE declaration invalid at computed-value time — the tab would
 * silently lose its surface, hover wash and seam, not merely its tint. A plugin
 * (or a hand-edited localStorage entry) can put anything in the store, so the
 * guard lives here rather than at each call site.
 */
export function isTintColor(value: null | string | undefined): value is string {
  return typeof value === 'string' && /^#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(value)
}

export type TabTintVars = CSSProperties & Record<'--pane-tab-active-accent' | '--pane-tab-active-bg' | '--pane-tab-strip-bg', string>

/**
 * The custom properties for a tinted tab. Idle and active surfaces are separate
 * vars because `PaneTab` swaps which one it reads by state — one `--tab-bg`
 * inline would paint the active tab with the idle mix.
 */
export function tabTintVars(color: null | string | undefined): TabTintVars | undefined {
  if (!isTintColor(color)) {
    return undefined
  }

  return {
    '--pane-tab-active-accent': color,
    '--pane-tab-active-bg': `color-mix(in srgb, ${color} ${TAB_TINT_ACTIVE_MIX}%, var(--ui-editor-surface-background))`,
    '--pane-tab-strip-bg': `color-mix(in srgb, ${color} ${TAB_TINT_IDLE_MIX}%, var(--ui-sidebar-surface-background))`
  }
}
