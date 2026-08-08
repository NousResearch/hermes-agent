/**
 * LocalStorage-backed set of backend skin names that resolved to live skins
 * in a previous session.
 *
 * Backend skins (user YAML files in ~/.hermes/skins/) are registered AFTER
 * the desktop paints its first frame — the gateway hasn't connected yet.
 * Without a memory of which names were valid LAST session, `normalizeSkin`
 * would reject `trt`/`ares`/… from localStorage and reset every fresh
 * launch to `nous`. This set bridges the gap: a name that resolved to a
 * backend skin in the previous session is treated as valid even before
 * the gateway reconnects, so the user's skin choice survives a restart.
 * Truly unknown/junk names still fall back to the default.
 */

import { storedString } from '@/lib/storage'

const KNOWN_SKIN_NAMES_KEY = 'hermes-desktop-known-skin-names-v1'

// Legacy global skin key, used for auto-seeding.  Defined here rather
// than imported from context.tsx to keep this module cycle-free.
const SKIN_KEY = 'hermes-desktop-theme-v2'
const RETIRED_SKINS = new Set(['nous-light', 'default', 'gold'])

export function knownSkinNames(): Set<string> {
  try {
    const raw = window.localStorage.getItem(KNOWN_SKIN_NAMES_KEY)
    const names: string[] = raw ? (Array.isArray(JSON.parse(raw)) ? JSON.parse(raw) : []) : []

    // Auto-seed: include the currently persisted skin name from the legacy
    // global slot so the set is populated on the very first launch after
    // this fix — no manual `/skin` needed.
    const active = storedString(SKIN_KEY)
    if (active && !RETIRED_SKINS.has(active) && !names.includes(active)) {
      names.push(active)
    }

    return new Set(names.filter((v): v is string => typeof v === 'string'))
  } catch {
    return new Set()
  }
}

export function rememberSkinName(name: string): void {
  const names = knownSkinNames()
  if (names.has(name)) {
    return
  }
  names.add(name)
  try {
    window.localStorage.setItem(KNOWN_SKIN_NAMES_KEY, JSON.stringify([...names]))
  } catch {
    /* storage unavailable — next launch just flashes once */
  }
}
