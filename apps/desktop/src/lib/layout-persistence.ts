import { atom, type WritableAtom } from 'nanostores'

import type { InterfaceMode } from '@/store/interface-mode'

import { type Codec, Codecs } from './persisted'
import { readKey, writeKey } from './storage'

interface LayoutEntry {
  capture(): void
  restore(): void
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

function migrateLayoutScopes(initialMode: InterfaceMode) {
  const marker = 'hermes.desktop.layoutModeScopes.v1'
  const legacy = new Map<string, string>()

  if (readKey(marker) !== null) {
    return legacy
  }

  let saved = true

  if (initialMode === 'simple') {
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

/** Layout state switches as a unit: changing visibility must not edit either
 * tree while the other stores are still loading. Advanced keeps the legacy
 * keys; Simple writes only its own namespace. */
export function createLayoutPersistence(initialMode: InterfaceMode, persistent: boolean) {
  const entries: LayoutEntry[] = []
  const listeners = new Set<() => void>()
  const snapshots = new Set<string>()
  let mode = initialMode
  let restoring = false

  const legacy = persistent ? migrateLayoutScopes(initialMode) : new Map<string, string>()

  const keyFor = (key: string) => (mode === 'advanced' ? key : `${key}.simple`)

  function write(key: string, raw: null | string) {
    if (!restoring) {
      snapshots.add(keyFor(key))

      if (persistent) {
        // Explicit empties keep migration retries from reviving cleared state.
        writeKey(keyFor(key), mode === 'simple' ? (raw ?? 'null') : raw)
      }
    }
  }

  function scopedAtom<T>(
    key: string,
    fallback: () => T,
    codec: Codec<T> = Codecs.json<T>(),
    manual = false
  ): WritableAtom<T> {
    const memory = new Map<InterfaceMode, T>()

    const load = (): T => {
      if (memory.has(mode)) {
        return memory.get(mode)!
      }

      const raw = persistent ? (readKey(keyFor(key)) ?? (mode === 'simple' ? (legacy.get(key) ?? null) : null)) : null

      if (raw !== null) {
        try {
          const value = codec.decode(raw)
          snapshots.add(keyFor(key))

          return value
        } catch {
          // Invalid storage falls back only within this mode.
        }
      }

      return fallback()
    }

    const $value = atom<T>(load())

    if (!manual) {
      $value.listen(value => write(key, codec.encode(value)))
    }

    entries.push({
      capture() {
        const value = $value.get()
        memory.set(mode, value)
        write(key, codec.encode(value))
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
      return mode
    },
    get restoring() {
      return restoring
    },
    has(key: string) {
      return snapshots.has(keyFor(key))
    },
    onRestore(listener: () => void) {
      listeners.add(listener)

      return () => void listeners.delete(listener)
    },
    write,
    change(next: InterfaceMode, publishMode: () => void) {
      if (mode === next) {
        return
      }

      entries.forEach(entry => entry.capture())
      mode = next
      restoring = true

      try {
        publishMode()
        entries.forEach(entry => entry.restore())
        listeners.forEach(listener => listener())
      } finally {
        restoring = false
      }
    }
  }
}
