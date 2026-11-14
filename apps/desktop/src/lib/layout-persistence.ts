import { atom, type WritableAtom } from 'nanostores'

import type { InterfaceMode } from '@/store/interface-mode'

import { type Codec, Codecs } from './persisted'
import { readKey, writeKey } from './storage'

interface LayoutEntry {
  capture(): void
  restore(): void
}

/** Which shell draws the window. The redesign merges developer tools into one side panel and
 *  marks them user-placed, so it keeps its own arrangement: turning it off must hand back the
 *  classic layout exactly as it was left. */
export type LayoutShell = 'classic' | 'portal'

/** One arrangement per interface mode × shell. */
export interface LayoutScope {
  mode: InterfaceMode
  shell: LayoutShell
}

export const LAYOUT_KEYS = {
  floating: 'hermes.desktop.floatingPanes.v1',
  tree: 'hermes.desktop.layoutTree.v2',
  preset: 'hermes.desktop.layoutPreset.active',
  panes: 'hermes.desktop.paneStates.v1',
  dismissed: 'hermes.desktop.dismissedPanes.v1',
  shares: 'hermes.desktop.paneShare.v1',
  sharePartners: 'hermes.desktop.paneSharePartners.v1',
  hiddenTabs: 'hermes.desktop.hiddenStripTabs.v1',
  placed: 'hermes.desktop.userPlacedPanes.v1',
  flipped: 'hermes.desktop.panesFlipped',
  collapsed: 'hermes.desktop.collapsedTreeSides.v1'
} as const

// The classic shell keeps the keys it shipped with (bare = Advanced, `.simple`), so classic
// users' layouts are untouched. The redesign never had a key of its own; it starts from the
// declared default rather than inheriting a classic arrangement it would immediately rewrite.
const SCOPE_SUFFIX: Record<LayoutShell, Record<InterfaceMode, string>> = {
  classic: { advanced: '', simple: '.simple' },
  portal: { advanced: '.portal', simple: '.portal.simple' }
}

const scopeId = (scope: LayoutScope) => SCOPE_SUFFIX[scope.shell][scope.mode]

// Only classic Simple has a pre-scope legacy record to fall back to (see migrateLayoutScopes).
const hasLegacyFallback = (scope: LayoutScope) => scope.shell === 'classic' && scope.mode === 'simple'

function migrateLayoutScopes(initial: LayoutScope) {
  const marker = 'hermes.desktop.layoutModeScopes.v1'
  const legacy = new Map<string, string>()

  if (readKey(marker) !== null) {
    return legacy
  }

  let saved = true

  if (initial.mode === 'simple') {
    for (const key of Object.values(LAYOUT_KEYS)) {
      const raw = readKey(key)

      if (raw === null) {
        continue
      }

      legacy.set(key, raw)

      if (readKey(`${key}.simple`) === null) {
        writeKey(`${key}.simple`, raw)
        saved = readKey(`${key}.simple`) === raw && saved
      }
    }
  }

  if (saved) {
    writeKey(marker, 'true')
  }

  return legacy
}

/** Layout state switches as a unit: changing scope must not edit either tree while the other
 *  stores are still loading. Each scope reads and writes only its own keys. */
export function createLayoutPersistence(initial: LayoutScope, persistent: boolean) {
  const entries: LayoutEntry[] = []
  const snapshots = new Set<string>()
  // Manual atoms changed in memory but not yet written (a resize mid-gesture).
  const unsaved = new Set<string>()
  const listeners = new Set<() => void>()
  let scope = initial
  let restoring = false
  let held = false

  const legacy = persistent ? migrateLayoutScopes(initial) : new Map<string, string>()

  const keyFor = (key: string) => `${key}${scopeId(scope)}`

  function write(key: string, raw: null | string) {
    if (!restoring) {
      unsaved.delete(key)
      snapshots.add(keyFor(key))

      if (persistent && !held) {
        // Explicit empties keep migration retries from reviving cleared state.
        writeKey(keyFor(key), hasLegacyFallback(scope) ? (raw ?? 'null') : raw)
      }
    }
  }

  function scopedAtom<T>(
    key: string,
    fallback: () => T,
    codec: Codec<T> = Codecs.json<T>(),
    manual = false
  ): WritableAtom<T> {
    const memory = new Map<string, T>()

    const load = (): T => {
      if (memory.has(scopeId(scope))) {
        return memory.get(scopeId(scope))!
      }

      const raw = persistent
        ? (readKey(keyFor(key)) ?? (hasLegacyFallback(scope) ? (legacy.get(key) ?? null) : null))
        : null

      if (raw !== null) {
        try {
          const value = codec.decode(raw)
          snapshots.add(keyFor(key))

          return value
        } catch {
          // Invalid storage falls back only within this scope.
        }
      }

      return fallback()
    }

    const $value = atom<T>(load())

    $value.listen(value => {
      if (!manual) {
        write(key, codec.encode(value))
      } else if (!restoring) {
        unsaved.add(key)
      }
    })

    entries.push({
      // Writes only what this window changed and has not saved yet. A scope switch driven by
      // another window's storage event must never write this window's values over that window's.
      capture() {
        const value = $value.get()
        memory.set(scopeId(scope), value)

        if (unsaved.has(key)) {
          write(key, codec.encode(value))
        }
      },
      restore() {
        $value.set(load())
      }
    })

    return $value
  }

  return {
    atom: scopedAtom,
    get mode() {
      return scope.mode
    },
    get scope(): LayoutScope {
      return scope
    },
    get restoring() {
      return restoring
    },
    /** While held, layout changes stay in memory: a borrowed layout (the first-run demo) is never
     *  written over the user's own, and a relaunch opens on the user's layout. */
    hold(next: boolean) {
      held = next
    },
    has(key: string) {
      return snapshots.has(keyFor(key))
    },
    onRestore(listener: () => void) {
      listeners.add(listener)

      return () => void listeners.delete(listener)
    },
    write,
    /** Capture every atom into the current scope, then load `next`'s. `publish` runs between the
     *  two so readers of the scope's inputs see the new value as the new layout loads. */
    change(next: LayoutScope, publish: () => void = () => {}) {
      if (scopeId(scope) === scopeId(next)) {
        return
      }

      entries.forEach(entry => entry.capture())
      scope = next
      restoring = true

      try {
        publish()
        entries.forEach(entry => entry.restore())
        listeners.forEach(listener => listener())
      } finally {
        restoring = false
      }
    }
  }
}
