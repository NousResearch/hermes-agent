/**
 * Pane id namespace for preview tabs (`preview-tile:<tabId>`). A leaf module
 * with no imports: the layout tree store's load-time scrub reads it while
 * `$layoutTree` decodes during module evaluation, so it must not sit on an
 * import cycle with that store.
 */
export const PREVIEW_TILE_PREFIX = 'preview-tile'
