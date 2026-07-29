import { cn } from '@/lib/utils'

/**
 * The composer surface and the status/queue stack paint ONE shared
 * `--composer-fill` var. The state ladder (rest / scrolled) lives in styles.css
 * on `[data-slot='composer-root']`, so the layers can never disagree.
 */
export const composerFill = 'bg-(--composer-fill)'

/** Backdrop treatment for the composer input surface. Harmless when the fill
 *  goes opaque (drawer open) — nothing shows through to blur. */
export const composerSurfaceGlass = cn(
  'backdrop-blur-[0.75rem] backdrop-saturate-[1.12] [-webkit-backdrop-filter:blur(0.75rem)_saturate(1.12)]',
  'transition-[background-color] duration-150 ease-out'
)

const composerDockEdge = (edge: 'bottom' | 'top') =>
  cn('border border-border/65', edge === 'top' ? 'rounded-t-2xl border-b-0' : 'rounded-b-2xl border-t-0')

/** Glassy docked card — the status stack / queue. Paints the SAME
 *  `--composer-fill` as the surface, so rest / scrolled / focused / drawer-open
 *  all match the composer by construction. */
export const composerDockCard = (edge: 'bottom' | 'top' = 'top') =>
  cn(composerDockEdge(edge), composerFill, composerSurfaceGlass)

/** Floating composer panel skin — the `/`·`@`·`?` completion drawer and the
 *  attach (`+`) menu. Glassy translucent card, hairline border, full radius,
 *  smallest type, soft nous shadow. Uses an explicit fill (not `--composer-fill`)
 *  so it renders identically whether mounted inside the composer or portaled out
 *  of it. Visual skin only — consumers add their own size/position/padding. */
export const composerPanelCard = cn(
  'rounded-2xl border border-border/65 shadow-nous text-[length:var(--conversation-tool-font-size)]',
  'bg-[color-mix(in_srgb,var(--dt-card)_72%,transparent)]',
  composerSurfaceGlass
)

/**
 * Shared grid for the chrome-free floating strips that bracket the composer —
 * the micro-action pills above the surface and the `composer.underside` slot
 * below it.
 *
 * Both strips are in-flow children of the SAME box (the composer root's
 * content box), which is the whole point: they previously lived in different
 * parents — the pills inside the status stack's absolute overlay lane, the
 * chip in the root — so "no padding" resolved to two different left edges and
 * they never lined up. Same parent, no inset, one constant: the left edges are
 * identical by construction, not by matching numbers in two places.
 *
 * `relative z-1` at the call sites is load-bearing, not styling. The pop-out
 * drag region is an `absolute` sibling, and positioned elements paint above
 * static in-flow ones whatever the DOM order — so without a stacking context
 * these strips sit UNDER it and their contents never receive hover or clicks
 * (the region does, and hatches).
 *
 * The strip is full-width so a contribution can push itself to the right
 * (`ml-auto`), but it is `pointer-events-none` with its CHILDREN re-enabled:
 * the chips are interactive, while the empty space between and beside them
 * falls through to the drag region and stays grab area. That combination is
 * why the composer is still draggable by the band its badges live in.
 */
export const composerFloatingStrip = cn(
  'relative z-1 flex w-full flex-wrap items-center gap-1.5',
  'pointer-events-none [&>*]:pointer-events-auto'
)
