/**
 * Optional global input hook for the HUD's "ask about this" sheet.
 *
 * Electron cannot observe mouse buttons pressed in OTHER apps. `uiohook-napi`
 * can — it is an OS-level input hook — but it is a native optionalDependency
 * that may be absent (no prebuilt for the platform, install skipped), so it is
 * loaded lazily and its absence is a normal, reported state rather than an
 * error. It observes only; it cannot swallow the click, which is why the
 * trigger is Ctrl + right-click and never a bare right-click: the app under
 * the cursor still gets its own context menu, and Ctrl keeps the two from
 * colliding on every ordinary right-click of the day.
 */

import path from 'node:path'
import { pathToFileURL } from 'node:url'

interface UiohookMouseEvent {
  altKey: boolean
  button: number
  ctrlKey: boolean
  metaKey: boolean
  shiftKey: boolean
  x: number
  y: number
}

interface UiohookLike {
  on(event: 'mousedown', listener: (event: UiohookMouseEvent) => void): unknown
  off?(event: 'mousedown', listener: (event: UiohookMouseEvent) => void): unknown
  removeListener?(event: 'mousedown', listener: (event: UiohookMouseEvent) => void): unknown
  start(): void
  stop(): void
}

interface UiohookModule {
  uIOhook: UiohookLike
}

/** uiohook's button numbering: 1 = left, 2 = right, 3 = middle. */
const RIGHT_BUTTON = 2

/** Pure trigger decision, so the gesture is testable without a hook. */
export function isHudAskGesture(event: Pick<UiohookMouseEvent, 'button' | 'ctrlKey' | 'metaKey'>, platform: string): boolean {
  if (event.button !== RIGHT_BUTTON) {
    return false
  }

  // ⌘ on macOS, Ctrl elsewhere — the same modifier every other Hermes chord
  // reads as "the command key".
  return platform === 'darwin' ? event.metaKey : event.ctrlKey
}

export interface HudInputHook {
  /** Start observing. Returns false when the hook could not start. */
  start(onAsk: () => void): boolean
  stop(): void
  /** Whether the native module loaded at all. */
  readonly available: boolean
  /** Why it is unavailable, for Settings. */
  readonly reason: null | string
}

export type HudHookLoader = () => Promise<UiohookModule>

const describeError = (error: unknown): string => (error instanceof Error ? error.message : String(error ?? 'unknown'))

/**
 * Build the hook around a loader (the real one imports `uiohook-napi`; tests
 * inject a fake). `load()` is called once; a rejected import becomes
 * `available: false` with the reason kept for Settings.
 */
export async function createHudInputHook(load: HudHookLoader, platform = process.platform): Promise<HudInputHook> {
  let hook: UiohookLike | null = null
  let reason: null | string = null

  try {
    const mod = await load()

    if (!mod?.uIOhook || typeof mod.uIOhook.on !== 'function') {
      reason = 'uiohook-napi loaded but exposes no uIOhook instance'
    } else {
      hook = mod.uIOhook
    }
  } catch (error) {
    reason = `uiohook-napi is not installed or failed to load (${describeError(error)})`
  }

  let listener: ((event: UiohookMouseEvent) => void) | null = null

  const stop = () => {
    if (!hook) {
      return
    }

    if (listener) {
      try {
        ;(hook.off ?? hook.removeListener)?.call(hook, 'mousedown', listener)
      } catch {
        // best effort
      }

      listener = null
    }

    try {
      hook.stop()
    } catch {
      // Stopping a hook that never started is fine.
    }
  }

  return {
    get available() {
      return hook !== null
    },
    get reason() {
      return reason
    },
    start(onAsk) {
      if (!hook) {
        return false
      }

      stop()

      listener = event => {
        if (isHudAskGesture(event, platform)) {
          onAsk()
        }
      }

      try {
        hook.on('mousedown', listener)
        hook.start()

        return true
      } catch (error) {
        reason = `uiohook-napi could not start (${describeError(error)})`
        listener = null

        return false
      }
    },
    stop
  }
}

/**
 * The production loader. The STAGED copy under dist/node_modules is tried
 * first — the packaged app ships only `dist/**`, so that is the only place
 * the module exists there (scripts/stage-native-deps.mjs puts it beside
 * get-windows). A dev run without a staged copy falls back to the bare
 * import; a missing module rejects instead of failing the build.
 */
export const loadUiohook = (stagedRoot?: string): Promise<UiohookModule> => {
  const specifiers = stagedRoot
    ? [pathToFileURL(path.join(stagedRoot, 'dist', 'index.js')).href, 'uiohook-napi']
    : ['uiohook-napi']

  return specifiers.reduce<Promise<UiohookModule>>(
    (chain, specifier) => chain.catch(() => import(specifier as string) as Promise<UiohookModule>),
    Promise.reject(new Error('uiohook-napi: no loader candidate'))
  )
}
