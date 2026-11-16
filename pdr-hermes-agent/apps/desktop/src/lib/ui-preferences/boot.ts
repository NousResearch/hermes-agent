// Runs BEFORE the app's module graph (see src/entry.ts): the localStorage cache
// must match this machine's shared preferences before any store module reads
// it, or a plain-atom store keeps a stale value and its next write clobbers the
// host's. A static import order cannot promise that (a bundler hoists shared
// chunks above the entry body, and ESM evaluates sibling imports before a
// top-level await settles), so the app is loaded only after `bootRenderer`
// hydrated. Keep this module's imports free of stores.
// Native Desktop's preload hands the snapshot over synchronously. A browser
// asks the server that serves it, bounded so an unreachable host never holds
// the first paint for long (then it boots on its local cache).
import { browserApi, browserBootstrap } from '@/lib/browser-transport'

import { captureOwnPreferences, hydrateFromSnapshot, type UiPreferencesSnapshot } from './sync'

const BROWSER_FIRST_READ_MS = 1500

async function browserSnapshot(): Promise<null | UiPreferencesSnapshot> {
  const bootstrap = browserBootstrap()

  if (!bootstrap) {
    return null
  }

  const read = browserApi<UiPreferencesSnapshot>(bootstrap, {
    path: '/api/ui-preferences',
    timeoutMs: BROWSER_FIRST_READ_MS
  }).catch(() => null)

  const late = new Promise<null>(resolve => setTimeout(() => resolve(null), BROWSER_FIRST_READ_MS))

  return Promise.race([read, late])
}

/** Hydrate this window's shared preferences, then load the app. */
export async function bootRenderer<T>(loadApp: () => Promise<T>): Promise<T> {
  if (typeof window !== 'undefined') {
    captureOwnPreferences()
    hydrateFromSnapshot(window.hermesDesktop?.uiPreferences?.snapshot ?? (await browserSnapshot()))
  }

  return loadApp()
}
