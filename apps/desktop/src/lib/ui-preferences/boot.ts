// Side-effect module, imported first by main.tsx: native Desktop's preload hands
// over this machine's shared preferences synchronously, so the localStorage
// cache matches the store before any store module reads it (no default flash).
// A browser has no synchronous door; its sync applies the store once fetched.
import { hydrateFromSnapshot } from './sync'

if (typeof window !== 'undefined') {
  hydrateFromSnapshot(window.hermesDesktop?.uiPreferences?.snapshot)
}
