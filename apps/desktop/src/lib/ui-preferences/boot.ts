// Side-effect module, imported first by main.tsx, so the localStorage cache
// matches this machine's shared preferences before any store module reads it
// (no default flash, and no plain-atom store seeded from a stale cache).
// Native Desktop's preload hands the snapshot over synchronously. A browser
// asks the server that serves it, bounded so an unreachable host never holds
// the first paint for long: the top-level await delays the modules main.tsx
// imports after this one.
import { browserApi, browserBootstrap } from '@/lib/browser-transport'

import { hydrateFromSnapshot, type UiPreferencesSnapshot } from './sync'

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

if (typeof window !== 'undefined') {
  hydrateFromSnapshot(window.hermesDesktop?.uiPreferences?.snapshot ?? (await browserSnapshot()))
}
