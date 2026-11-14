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
 * Leave Electron's assistive technology detection in control by default:
 * forcing the tree can interfere with macOS window managers and Windows
 * accessibility clients (#133652). Dictation tools that cannot register
 * can opt in with `desktop.renderer_accessibility: true` in config.yaml,
 * bridged as HERMES_DESKTOP_RENDERER_ACCESSIBILITY=1 by both the launcher
 * and main.ts's pre-ready config read. False skips forcing; it does not
 * disable accessibility requested by an actual assistive technology.
 *
 * Extracted dependency-free — the Electron app object is injected — so the
 * platform/env decision is unit-testable per branch without booting Electron,
 * per the windows-child-options.ts convention.
 */

/** Values that turn the feature OFF when HERMES_DESKTOP_RENDERER_ACCESSIBILITY
 *  carries them (same words the launcher accepts for false). */
const RENDERER_ACCESSIBILITY_OFF_WORDS = new Set(['0', 'false', 'no', 'off', 'disabled'])

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

  return raw.length > 0 && !RENDERER_ACCESSIBILITY_OFF_WORDS.has(raw)
}

/** Bridge the packaged launch config without overriding an explicit environment value. */
export function bridgeRendererAccessibilityConfig(
  configured: boolean | undefined,
  env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY?: string | undefined } = process.env
): void {
  if (configured !== undefined && env.HERMES_DESKTOP_RENDERER_ACCESSIBILITY === undefined) {
    env.HERMES_DESKTOP_RENDERER_ACCESSIBILITY = configured ? '1' : '0'
  }
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
