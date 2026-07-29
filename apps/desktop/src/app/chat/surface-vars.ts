/**
 * Measured-height CSS vars, scoped to one chat surface.
 *
 * The composer and the out-of-flow status stack both publish their measured
 * height so the thread can reserve bottom clearance for them. Those vars used
 * to be written to `document.documentElement` — a single global slot — while
 * the components writing them mount once per chat surface. Session tiles
 * render a full ChatView (thread + composer + status stack) beside the
 * workspace pane, so N surfaces raced for one value: a background tab with a
 * tall stack inflated the foreground thread's padding and pushed its
 * jump-to-bottom button into mid-screen.
 *
 * Publishing onto the surface's own root element instead makes each ChatView
 * read its own measurement through normal CSS inheritance. Remeasuring on
 * session switch cannot fix this — several surfaces are visible at once, so
 * there is no single correct value to remeasure to.
 *
 * A measurement is only ever meaningful to the surface that owns it, so an
 * unowned node has nowhere to publish to and must publish NOWHERE. Writing to
 * the document root instead is not a graceful fallback: `:root` holds the
 * defaults every surface inherits until it publishes its own, so one stale
 * write there is a global floor under every thread's bottom clearance, on
 * every tab, until reload. That is the "randomly huge gap at the bottom of the
 * thread" bug — see surface-vars.test.ts.
 */

export const COMPOSER_HEIGHT_VAR = '--composer-measured-height'
export const COMPOSER_SURFACE_HEIGHT_VAR = '--composer-surface-measured-height'
export const STATUS_STACK_VAR = '--status-stack-measured-height'

/**
 * The surface owning `el`, or null when `el` is detached or outside one.
 *
 * Null is the honest answer, not a failure to be papered over. Every publisher
 * lives inside a `[data-chat-surface]` for its whole visible life (a
 * popped-out composer is `position: fixed`, still a descendant), so the only
 * way to miss is a node React already removed — whose measurement is stale by
 * definition.
 */
export function chatSurfaceRoot(el: Element | null): HTMLElement | null {
  return el?.closest<HTMLElement>('[data-chat-surface]') ?? null
}

/** Publish a measured-height var on the surface owning `el`. No owner, no write. */
export function setSurfaceVar(el: Element | null, name: string, value: string): void {
  chatSurfaceRoot(el)?.style.setProperty(name, value)
}

/** Clear a measured-height var from `root`. */
export function clearSurfaceVar(root: HTMLElement | null, name: string): void {
  root?.style.removeProperty(name)
}
