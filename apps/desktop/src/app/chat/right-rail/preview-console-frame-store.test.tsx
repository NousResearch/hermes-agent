import { act, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useFrameStore } from './preview-console'
import type { ConsoleEntry } from './preview-console-state'

const entry = (id: number): ConsoleEntry => ({ id, level: 0, message: `line-${id}` })

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('preview console frame coalescing', () => {
  it('renders once per frame for a burst of log appends', () => {
    const frames: FrameRequestCallback[] = []
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.push(callback)

      return frames.length
    })
    vi.stubGlobal('cancelAnimationFrame', () => {})

    const store = atom<ConsoleEntry[]>([])
    let renders = 0

    const { result } = renderHook(() => {
      renders += 1

      return useFrameStore(store)
    })

    const before = renders

    act(() => {
      for (let id = 1; id <= 5; id += 1) {
        store.set([...store.get(), entry(id)])
      }
    })

    // Nothing is painted until the frame callback runs.
    expect(renders - before).toBe(0)
    expect(frames).toHaveLength(1)

    act(() => {
      frames.splice(0).forEach(callback => callback(0))
    })

    expect(renders - before).toBe(1)
    expect(result.current.map(item => item.id)).toEqual([1, 2, 3, 4, 5])
  })
})
