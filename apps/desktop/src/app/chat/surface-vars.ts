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
 */

export const COMPOSER_HEIGHT_VAR = '--composer-measured-height'
export const COMPOSER_SURFACE_HEIGHT_VAR = '--composer-surface-measured-height'
export const STATUS_STACK_VAR = '--status-stack-measured-height'

/**
 * The surface owning `el`, falling back to the document root so a composer
 * rendered outside a chat surface (the popped-out window) keeps behaving
 * exactly as before.
 */
export function chatSurfaceRoot(el: Element | null): HTMLElement {
  return el?.closest<HTMLElement>('[data-chat-surface]') ?? document.documentElement
}

/** Publish a measured-height var on the surface owning `el`. */
export function setSurfaceVar(el: Element | null, name: string, value: string): void {
  chatSurfaceRoot(el).style.setProperty(name, value)
}

/** Clear a measured-height var from the surface owning `el`. */
export function clearSurfaceVar(el: Element | null, name: string): void {
  chatSurfaceRoot(el).style.removeProperty(name)
}
