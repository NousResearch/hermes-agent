import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { StarmapGraph } from '@/types/hermes'

vi.mock('@/hermes', () => ({
  getStarmapGraph: vi.fn()
}))

import { getStarmapGraph } from '@/hermes'

import { $starmapError, $starmapGraph, $starmapLoading, loadStarmapGraph, resetStarmapGraph } from './starmap'

function graph(tag: string): StarmapGraph {
  return {
    clusters: [],
    edges: [],
    memory: [],
    nodes: [
      {
        category: 'cat',
        createdBy: null,
        id: `node-${tag}`,
        kind: 'skill',
        label: tag,
        pinned: false,
        state: 'ok',
        useCount: 1
      }
    ],
    stats: {}
  }
}

beforeEach(() => {
  vi.mocked(getStarmapGraph).mockReset()
  resetStarmapGraph()
})

describe('starmap store', () => {
  it('refreshes cached data on a forced reload (panel reopen)', async () => {
    // First open fills the cache.
    vi.mocked(getStarmapGraph).mockResolvedValueOnce(graph('stale'))
    await loadStarmapGraph(true)
    expect($starmapGraph.get()?.nodes[0]?.id).toBe('node-stale')

    // The backend now serves fresher data. The store cache only resets on a
    // profile switch, so a reopen must not be served from the stale cache.
    vi.mocked(getStarmapGraph).mockResolvedValueOnce(graph('fresh'))
    await loadStarmapGraph(true)

    expect(getStarmapGraph).toHaveBeenCalledTimes(2)
    expect($starmapGraph.get()?.nodes[0]?.id).toBe('node-fresh')
    expect($starmapError.get()).toBeNull()
    expect($starmapLoading.get()).toBe(false)
  })

  it('still serves an unforced load from the cache without another scan', async () => {
    vi.mocked(getStarmapGraph).mockResolvedValueOnce(graph('cached'))
    await loadStarmapGraph(true)
    expect(getStarmapGraph).toHaveBeenCalledTimes(1)

    await loadStarmapGraph()

    expect(getStarmapGraph).toHaveBeenCalledTimes(1)
    expect($starmapGraph.get()?.nodes[0]?.id).toBe('node-cached')
  })
})
