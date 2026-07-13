import { MACOS_TAHOE_DARWIN_MAJOR } from './titlebar-overlay-width'

export { MACOS_TAHOE_DARWIN_MAJOR }

/**
 * Chromium command-line switches to disable FontationsFontBackend on macOS
 * Tahoe (Darwin ≥ 25). Returns the switch name and value when the workaround
 * is needed, or `null` on unaffected platforms.
 *
 * Background: macOS 26's Apple Color Emoji.ttc ships a corrupt sbix PNG that
 * SIGBUS-crashes Apple's ImageIO when Chromium's Fontations (Rust) font
 * backend routes bitmap extraction through NSImage/NSPasteboard → ImageIO.
 * Disabling FontationsFontBackend forces Chromium onto the older
 * SkTypeface/CoreText path that decodes sbix PNGs through Skia's own libpng,
 * bypassing the broken ImageIO codepath entirely.
 *
 * @param darwinMajor  Darwin kernel major (e.g. 25 for Tahoe). Pass 0 for
 *                     non-macOS platforms.
 * @returns `{ switch: string, value: string }` when the workaround applies,
 *          or `null` otherwise.
 */
export function fontationsWorkaround(darwinMajor: number): { switch: string; value: string } | null {
  if (darwinMajor >= MACOS_TAHOE_DARWIN_MAJOR) {
    return { switch: 'disable-features', value: 'FontationsFontBackend' }
  }

  return null
}
