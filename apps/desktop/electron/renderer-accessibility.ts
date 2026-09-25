/**
 * renderer-accessibility.ts
 *
 * Chromium only builds the renderer's accessibility tree when it detects an
 * assistive technology — and dictation tools that insert text through the OS
 * accessibility APIs (Wispr Flow and similar) do not register the way a
 * screen reader does, so the tree never appears: the window exposes only its
 * chrome and the contenteditable composer is invisible to them, on macOS
 * (#118271) and on Windows (#92607) alike. Electron's manual switch for that
 * is `app.setAccessibilitySupportEnabled(true)` once the app is ready.
 *
 * Default ON for the platforms where the typed API exists (darwin, win32) —
 * the composer being reachable by every accessibility-driven input method is
 * the point of the bug fix; the tree's rendering cost is the price. A
 * perf-sensitive setup can opt out with `desktop.renderer_accessibility:
 * false` in config.yaml, which the `hermes desktop` launcher bridges as
 * HERMES_DESKTOP_RENDERER_ACCESSIBILITY=0 (same shape as desktop.disable_gpu);
 * the Electron side also honors that variable when set directly.
 *
 * Extracted dependency-free — the Electron app object is injected — so the
 * platform/env decision is unit-testable per branch without booting Electron,
 * per the windows-child-options.ts convention.
 */

/** Values that turn the feature OFF when HERMES_DESKTOP_RENDERER_ACCESSIBILITY
 *  carries them (same words the launcher accepts for false). */
const RENDERER_ACCESSIBILITY_OFF_WORDS = new Set(['0', 'false', 'no', 'off'])

/** Whether this boot must expose the renderer's accessibility tree to the OS.
 *
 *  @param env - environment to read HERMES_DESKTOP_RENDERER_ACCESSIBILITY
 *    from (defaults to the real process env).
 *  @param platform - defaults to the real platform; injectable for tests.
 */
export function shouldEnableRendererAccessibility(
  env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY?: string | undefined } = process.env,
  platform: NodeJS.Platform = process.platform
): boolean {
  if (platform !== 'darwin' && platform !== 'win32') {
    // The typed Electron API is darwin/win32. Linux accessibility flows
    // through the platform's own ATK stack and needs no manual enable here.
    return false
  }

  const raw = (env.HERMES_DESKTOP_RENDERER_ACCESSIBILITY ?? '').trim().toLowerCase()

  return !RENDERER_ACCESSIBILITY_OFF_WORDS.has(raw)
}

/** Minimal slice of Electron's app API this module needs — injected so the
 *  module stays importable without Electron. */
export interface RendererAccessibilityApp {
  setAccessibilitySupportEnabled(enabled: boolean): void
}

/** Enable the renderer accessibility tree when this boot's platform/env says
 *  to. Call after the app `ready` event (the Electron API's requirement). */
export function enableRendererAccessibility(options: {
  appApi: RendererAccessibilityApp
  env?: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY?: string | undefined }
  platform?: NodeJS.Platform
}): void {
  if (shouldEnableRendererAccessibility(options.env, options.platform)) {
    options.appApi.setAccessibilitySupportEnabled(true)
  }
}
