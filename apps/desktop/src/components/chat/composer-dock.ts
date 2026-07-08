import { cn } from '@/lib/utils'

/**
 * The composer surface and everything docked to it (slash·@ popover, `?` help)
 * paint ONE shared `--composer-fill` var. The state ladder (rest / scrolled /
 * focused / drawer-open) lives in styles.css on `[data-slot='composer-root']`,
 * so the two layers can never disagree — drawer-open forces an opaque fill via
 * `:has()`, because translucent glass sampling different backdrops (thread vs
 * fade gradient) renders as different colors even with identical tints.
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

/** Fused docked card — completion drawers. Shares `--composer-fill` with the
 *  composer surface, which goes opaque while a drawer is open. */
export const composerFusedDockCard = (edge: 'bottom' | 'top' = 'top') => cn(composerDockEdge(edge), composerFill)
