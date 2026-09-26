import { describe, expect, test } from 'vitest'

import { registry } from './registry'
import type { Contribution } from './types'

const entry = (area: string, id: string, extra: Partial<Contribution> = {}): Contribution =>
  ({ area, id, ...extra }) as Contribution

describe('registry snapshot stability (getSnapshot contract)', () => {
  test('re-resolving an unchanged area returns the SAME reference', () => {
    const a = entry('stability.a', 'one')
    registry.register(a)

    const first = registry.getArea('stability.a')

    // Invalidation drops the cached snapshot. Re-resolving to the SAME entries
    // must reuse the previous array: `useSyncExternalStore` compares with
    // `Object.is`, so a fresh-but-identical array reads as a change and can
    // loop the workspace pane into "Maximum update depth exceeded".
    registry.register(a) // re-register the same contribution

    const second = registry.getArea('stability.a')

    expect(second).toBe(first)
  })

  test('an actual change still produces a new reference', () => {
    const a = entry('stability.b', 'one')
    registry.register(a)

    const first = registry.getArea('stability.b')

    const dispose = registry.register(entry('stability.b', 'two'))
    const second = registry.getArea('stability.b')

    expect(second).not.toBe(first)
    expect(second).toHaveLength(2)

    // ...and removing it returns to a stable, correct snapshot.
    dispose()
    const third = registry.getArea('stability.b')
    expect(third).toHaveLength(1)
    expect(registry.getArea('stability.b')).toBe(third)
  })

  test('empty areas stay referentially stable', () => {
    const first = registry.getArea('stability.empty')
    expect(registry.getArea('stability.empty')).toBe(first)
  })
})
