import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import type { StarmapGraph } from '@/types/hermes'

import { $starmapGraph, $starmapLoading, evictStarmapNode, loadStarmapGraph, resetStarmapGraph } from './starmap'

interface Deferred<T> {
  promise: Promise<T>
  reject: (reason?: unknown) => void
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}

function graph(id: string): StarmapGraph {
  return {
    clusters: [],
    edges: [],
    memory: [],
    nodes: [
      {
        category: 'memory',
        createdBy: 'memory',
        id,
        kind: 'memory',
        label: id,
        pinned: false,
        state: 'active',
        useCount: 1
      }
    ],
    stats: {}
  }
}

describe('Starmap cache ownership', () => {
  const requests = new Map<string, Deferred<StarmapGraph>>()
  const api = vi.fn(({ connectionId, path }: { connectionId?: string; path: string }) => {
    if (path !== '/api/learning/graph') {
      throw new Error(`Unexpected request: ${path}`)
    }

    const request = requests.get(connectionId ?? 'local')

    if (!request) {
      throw new Error(`No graph response for ${connectionId ?? 'local'}`)
    }

    return request.promise
  })

  beforeEach(() => {
    vi.stubGlobal('window', { hermesDesktop: { api } })
    setApiRequestProfile('shared')
    setApiRequestConnection('connection-a')
    resetStarmapGraph()
    api.mockClear()
    requests.clear()
  })

  afterEach(() => {
    resetStarmapGraph()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    requests.clear()
    vi.unstubAllGlobals()
  })

  it('keeps stale loads and delete rollbacks out of a new connection with the same profile', async () => {
    const graphA = graph('a-node')
    const graphB = graph('b-node')

    $starmapGraph.set(graphA)
    const sameGenerationRollback = evictStarmapNode('a-node')
    expect($starmapGraph.get()?.nodes).toEqual([])
    sameGenerationRollback()
    expect($starmapGraph.get()).toBe(graphA)

    const rollbackA = evictStarmapNode('a-node')
    const deleteA = deferred<void>()
    const deleteSettled = deleteA.promise.catch(() => rollbackA())
    const requestA = deferred<StarmapGraph>()
    requests.set('connection-a', requestA)
    const loadA = loadStarmapGraph(true)

    setApiRequestConnection('connection-b')

    const requestB = deferred<StarmapGraph>()
    requests.set('connection-b', requestB)
    const loadB = loadStarmapGraph()

    requestB.resolve(graphB)
    await loadB
    expect($starmapGraph.get()).toBe(graphB)

    deleteA.reject(new Error('connection A delete failed'))
    await deleteSettled
    requestA.resolve(graphA)
    await loadA

    expect(api.mock.calls.map(([request]) => request.connectionId)).toEqual(['connection-a', 'connection-b'])
    expect($starmapGraph.get()).toBe(graphB)
    expect($starmapLoading.get()).toBe(false)
  })
})
