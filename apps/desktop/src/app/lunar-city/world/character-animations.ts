export interface ImportedCharacterClip {
  name: string
  isPlaying?: boolean
  hasMotion?: boolean
  start?(loop?: boolean): void
  stop?(): void
  reset?(): void
}

/** One actual imported clip per character; no synthesized missing states. */
export function createCharacterAnimations(requestRender: () => void, diagnostic: (message: string) => void = console.warn) {
  const registered = new Map<
    string,
    { groups: ReadonlyMap<string, ImportedCharacterClip>; continuous: ReadonlySet<string> }
  >()

  const active = new Map<string, ImportedCharacterClip>()
  const desired = new Map<string, string>()
  let reduced = false
  const reported = new Set<string>()

  function set(id: string, state: string) {
    desired.set(id, state)
    const entry = registered.get(id)
    const current = active.get(id)
    const next = entry?.groups.get(state)

    if (state === 'idle' && reduced) {
      current?.stop?.()
      next?.reset?.()
      active.delete(id)
    } else if (next?.hasMotion === false) {
      current?.stop?.()
      next.reset?.()
      next.stop?.()
      active.delete(id)
    } else if (!next) {
      current?.stop?.()
      current?.reset?.()
      entry?.groups.get('idle')?.reset?.()
      active.delete(id)
      const message = `Lunar City missing animation ${id}:${state}; animation stopped; imported idle pose used when available`

      if (state !== 'idle' && !reported.has(message)) { reported.add(message); diagnostic(message) }
    } else if (next) {
      if (current !== next) {
        current?.stop?.()
        active.set(id, next)
      }

      if (current !== next || next.isPlaying !== true) {
        next.start?.(!reduced && entry!.continuous.has(state))
      }
    }

    requestRender()
  }

  return {
    register(id: string, groups: ReadonlyMap<string, ImportedCharacterClip>, continuous: ReadonlySet<string>) {
      registered.set(id, { groups, continuous })
    },
    set,
    activeCount() {
      return active.size
    },
    isActive() {
      for (const [id, group] of active) {
        if (group.isPlaying !== true) {
          active.delete(id)
        }
      }

      return active.size > 0
    },
    setReducedMotion(value: boolean) {
      reduced = value

      for (const group of active.values()) {
        group.stop?.()
      }

      active.clear()

      if (!reduced) {
        for (const [id, state] of desired) {
          if (registered.get(id)?.continuous.has(state)) {
            set(id, state)
          }
        }
      }

      requestRender()
    },
    dispose() {
      for (const group of active.values()) {
        group.stop?.()
      }

      active.clear()
      desired.clear()
      registered.clear()
      reported.clear()
    }
  }
}
