import { cn } from '@/lib/utils'

/**
 * The composer surface and the status/queue stack paint ONE shared
 * `--composer-fill` var. The state ladder (rest / scrolled) lives in styles.css
 * on `[data-slot='composer-root']`, so the layers can never disagree.
 */
export const composerFill = 'bg-(--composer-fill)'

/** The reply field itself is intentionally quieter than the cards docked to
 * it. It carries a faint powder-blue tint over the live chat canvas without a
 * backdrop blur, so on the matching light-blue field it visually disappears
 * while text, controls, and the focus border remain crisp. */
export const composerReplySurface = cn(
  'bg-(--composer-reply-fill)',
  'transition-[background-color] duration-150 ease-out'
)

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
 * A quiet control floating over composer content — the micro-action pills above
 * the surface, the Open affordance on a hovered link inside it. Full radius,
 * hairline border, the composer's own fill behind a blur so the text underneath
 * never shows through. Sized against the composer's control height so a pill
 * lines up with the chrome it floats above.
 *
 * Skin and size only; the call site owns position, width caps, and disabled
 * state.
 */
export const composerFloatingPill = cn(
  'inline-flex h-(--composer-control-size) shrink-0 cursor-pointer items-center gap-1.5 rounded-full px-2.5',
  'border border-border/65 bg-(--composer-fill) backdrop-blur-[0.75rem] [-webkit-backdrop-filter:blur(0.75rem)]',
  'text-xs font-normal text-(--ui-text-secondary) transition-colors',
  'hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/**
 * Shared grid for the chrome-free floating strips that bracket the composer —
 * the micro-action pills above the surface and the `composer.underside` slot
 * below it.
 *
 * Both are in-flow children of the composer DOCK, siblings of the composer
 * itself rather than children of it. That's deliberate: the pop-out drag
 * region is `absolute inset-0` inside the composer, so anything rendered in
 * there is inside the grab area by construction. Living outside makes that
 * impossible instead of something the gesture has to exclude.
 *
 * One parent and one constant means the two strips share a left edge without
 * anyone matching numbers across files. Vertical spacing stays at the call
 * site; the horizontal inset matches the composer's 5px grab margin so the
 * strips line up with the surface rather than the margin's outer edge.
 */
export const composerFloatingStrip = 'flex flex-wrap items-center gap-1.5'
