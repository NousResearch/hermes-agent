import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { refreshBackgroundProcesses, resetBackgroundPollingGuard } from './composer-status'
import { $gateway } from './gateway'
import { markRuntimeGone, noteRuntimeAlive, resetRuntimeGoneHealing } from './runtime-gone'
import { $activeSessionId, $sessionResumeRequest } from './session'
import { $sessionStates, $sessionTiles } from './session-states'

const STORED = 'stored-1'
const RUNTIME = 'runtime-dead'

/** Minimal cached state: only the durable-identity field this module reads. */
const cachedState = (storedSessionId: string) => ({ messages: [], storedSessionId }) as never

const tile = (storedSessionId: string, runtimeId?: string) => ({ runtimeId, storedSessionId }) as never

beforeEach(() => {
  resetRuntimeGoneHealing()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
})

afterEach(() => {
  $gateway.set(null as never)
  resetBackgroundPollingGuard()
  resetRuntimeGoneHealing()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
})

describe('markRuntimeGone', () => {
  it('unbinds the tile holding the dead runtime so its resume effect refires', () => {
    $sessionTiles.set([tile(STORED, RUNTIME), tile('stored-2', 'runtime-live')])

    expect(markRuntimeGone(RUNTIME)).toBe(true)

    const tiles = $sessionTiles.get()

    // The dead binding is cleared — SessionTilePane's resume effect is gated on
    // `!runtimeId`, so this is what re-arms it against the intact stored row.
    expect(tiles.find(t => t.storedSessionId === STORED)?.runtimeId).toBeUndefined()
    // A healthy neighbour must not be disturbed.
    expect(tiles.find(t => t.storedSessionId === 'stored-2')?.runtimeId).toBe('runtime-live')
  })

  it('asks the primary chat to resume when the dead runtime is the one on screen', () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    $activeSessionId.set(RUNTIME)

    expect(markRuntimeGone(RUNTIME)).toBe(true)

    // useRouteResume skips on `alreadyActive`, which stays true forever against
    // a dead id; only an explicit request bypasses it without a reconnect.
    expect($sessionResumeRequest.get()?.sessionId).toBe(STORED)
  })

  it('does not navigate the primary chat for a tile-only dead runtime', () => {
    $sessionTiles.set([tile(STORED, RUNTIME)])
    $activeSessionId.set('runtime-of-the-focused-chat')

    expect(markRuntimeGone(RUNTIME)).toBe(true)
    expect($sessionResumeRequest.get()).toBeNull()
  })

  it('heals a runtime id exactly once, however many pollers report it', () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    $activeSessionId.set(RUNTIME)

    expect(markRuntimeGone(RUNTIME)).toBe(true)

    const firstSequence = $sessionResumeRequest.get()?.sequence

    expect(markRuntimeGone(RUNTIME)).toBe(false)
    expect(markRuntimeGone(RUNTIME)).toBe(false)
    expect($sessionResumeRequest.get()?.sequence).toBe(firstSequence)
  })

  it('latches without resuming when the runtime has no durable identity', () => {
    // A never-persisted draft: no cached state, no tile. Nothing to resume.
    expect(markRuntimeGone('runtime-orphan')).toBe(false)
    expect($sessionResumeRequest.get()).toBeNull()
  })

  it('caps consecutive heals so a reap-on-sight backend cannot become a resume loop', () => {
    $activeSessionId.set(null)

    // Each cycle: the view rebinds a fresh runtime id, which is reaped again.
    for (let i = 0; i < 3; i += 1) {
      $sessionTiles.set([tile(STORED, `runtime-${i}`)])
      expect(markRuntimeGone(`runtime-${i}`)).toBe(true)
    }

    $sessionTiles.set([tile(STORED, 'runtime-3')])
    expect(markRuntimeGone('runtime-3')).toBe(false)
    expect($sessionTiles.get()[0]?.runtimeId).toBe('runtime-3')
  })

  it('refunds the budget once a binding proves healthy', () => {
    for (let i = 0; i < 3; i += 1) {
      $sessionTiles.set([tile(STORED, `runtime-${i}`)])
      markRuntimeGone(`runtime-${i}`)
    }

    // The next runtime answers a poll, so the earlier deaths are not a streak.
    $sessionStates.set({ 'runtime-healthy': cachedState(STORED) })
    noteRuntimeAlive('runtime-healthy')

    $sessionTiles.set([tile(STORED, 'runtime-4')])
    expect(markRuntimeGone('runtime-4')).toBe(true)
  })
})

describe('refreshBackgroundProcesses recovery', () => {
  it('carries the gateway 4001 verdict to the view instead of only going quiet', async () => {
    $sessionTiles.set([tile(STORED, RUNTIME)])
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    $activeSessionId.set(RUNTIME)
    $gateway.set({
      request: vi.fn(async () => {
        throw new Error('session not found')
      })
    } as never)

    await refreshBackgroundProcesses(RUNTIME)

    // Before this fix the poll latched and the window stayed bound to the
    // phantom runtime for the rest of its life — quiet, and still unusable.
    expect($sessionTiles.get()[0]?.runtimeId).toBeUndefined()
    expect($sessionResumeRequest.get()?.sessionId).toBe(STORED)
  })

  it('leaves a transient failure alone — the binding may still be alive', async () => {
    $sessionTiles.set([tile(STORED, RUNTIME)])
    $gateway.set({
      request: vi.fn(async () => {
        throw new Error('request timed out after 30s: process.list')
      })
    } as never)

    await refreshBackgroundProcesses(RUNTIME)

    expect($sessionTiles.get()[0]?.runtimeId).toBe(RUNTIME)
    expect($sessionResumeRequest.get()).toBeNull()
  })
})
