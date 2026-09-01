import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { refreshBackgroundProcesses, resetBackgroundPollingGuard } from './composer-status'
import { $gateway } from './gateway'
import {
  isSessionGone,
  isSessionGoneForBackgroundPolling,
  markRuntimeGone,
  markSessionGone,
  noteRuntimeAlive,
  resetBackgroundPollingGuardAfterRebind,
  resetRuntimeGoneHealing
} from './runtime-gone'
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

describe('gone-latch classifier and rebind seam', () => {
  it('recognizes structured 4001 and bare legacy text without misclassifying coded errors', () => {
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('gone', { code: 4001 }))).toBe(true)
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('session not found', { code: 5007 }))).toBe(false)
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('session not found'))).toBe(true)
    expect(isSessionGoneForBackgroundPolling(new Error("Error invoking remote method 'x': Error: session not found"))).toBe(
      true
    )
    expect(isSessionGoneForBackgroundPolling(new Error('tool failed: upstream said session not found'))).toBe(false)
  })

  it('clears the latch only for ids a successful resume/activate rebound', () => {
    markSessionGone('rt-dead')
    markSessionGone('rt-other')

    resetBackgroundPollingGuardAfterRebind('process.list', { session_id: 'rt-dead' }, { session_id: 'rt-dead' })
    expect(isSessionGone('rt-dead')).toBe(true)

    resetBackgroundPollingGuardAfterRebind('session.resume', { session_id: 'stored-1' }, { session_id: 'rt-dead' })
    expect(isSessionGone('rt-dead')).toBe(false)
    expect(isSessionGone('rt-other')).toBe(true)

    resetBackgroundPollingGuardAfterRebind('session.activate', { session_id: 'rt-other' }, undefined)
    expect(isSessionGone('rt-other')).toBe(false)
  })

  it('refunds the stored session heal budget on a successful rebind', () => {
    // Three reaps exhaust MAX_CONSECUTIVE_HEALS for STORED...
    for (const rt of ['rt-1', 'rt-2', 'rt-3']) {
      $sessionStates.set({ [rt]: cachedState(STORED) })
      $sessionTiles.set([tile(STORED, rt)])
      expect(markRuntimeGone(rt)).toBe(true)
    }

    $sessionStates.set({ 'rt-4': cachedState(STORED) })
    $sessionTiles.set([tile(STORED, 'rt-4')])
    expect(markRuntimeGone('rt-4')).toBe(false)

    // ...but a rebind of STORED proves it alive, so the next reap heals again.
    resetBackgroundPollingGuardAfterRebind('session.resume', { session_id: STORED }, { session_id: 'rt-5' })

    $sessionStates.set({ 'rt-5': cachedState(STORED) })
    $sessionTiles.set([tile(STORED, 'rt-5')])
    expect(markRuntimeGone('rt-5')).toBe(true)
  })
})
