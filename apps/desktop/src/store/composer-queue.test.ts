import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $pendingSteersBySession,
  $queuedPromptsBySession,
  addPendingSteer,
  clearQueuedPrompts,
  enqueueQueuedPrompt,
  getPendingSteers,
  getQueuedPrompts,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  setGatewayRequester,
  setSessionQueue,
  settlePendingSteer,
  updateQueuedPromptText
} from './composer-queue'

const SESSION_KEY = 'session-1'

describe('composer-queue (gateway-backed)', () => {
  const requester = vi.fn()

  beforeEach(() => {
    $queuedPromptsBySession.set({})
    $pendingSteersBySession.set({})
    requester.mockReset()
    requester.mockResolvedValue({ entry_id: 'gw-1', status: 'queued' })
    setGatewayRequester(requester as never)
  })

  afterEach(() => {
    setGatewayRequester(null)
  })

  it('enqueue fires session.queue.add and mirrors the entry locally', async () => {
    const entry = await enqueueQueuedPrompt(SESSION_KEY, { text: 'queued draft' })

    expect(requester).toHaveBeenCalledWith('session.queue.add', { session_id: SESSION_KEY, text: 'queued draft' })
    expect(entry?.id).toBe('gw-1')
    expect(getQueuedPrompts(SESSION_KEY).map(e => e.text)).toEqual(['queued draft'])
  })

  it('enqueue returns null (and keeps the mirror empty) when the gateway rejects', async () => {
    requester.mockRejectedValue(new Error('gateway down'))

    const entry = await enqueueQueuedPrompt(SESSION_KEY, { text: 'lost?' })

    expect(entry).toBeNull()
    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
  })

  it('enqueue returns null with no requester wired', async () => {
    setGatewayRequester(null)

    expect(await enqueueQueuedPrompt(SESSION_KEY, { text: 'no gateway' })).toBeNull()
  })

  it('setSessionQueue replaces the mirror with the gateway entry list', () => {
    setSessionQueue(SESSION_KEY, [
      { id: 'a', text: 'first', queued_at: 1700000000.5 },
      { id: 'b', text: 'second' }
    ])

    const entries = getQueuedPrompts(SESSION_KEY)
    expect(entries.map(e => e.text)).toEqual(['first', 'second'])
    expect(entries[0]?.queuedAt).toBe(1700000000500)

    setSessionQueue(SESSION_KEY, [])
    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
  })

  it('removeQueuedPrompt updates the mirror optimistically and fires the RPC', () => {
    setSessionQueue(SESSION_KEY, [
      { id: 'a', text: 'first' },
      { id: 'b', text: 'second' }
    ])

    expect(removeQueuedPrompt(SESSION_KEY, 'a')).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(e => e.id)).toEqual(['b'])
    expect(requester).toHaveBeenCalledWith('session.queue.remove', { session_id: SESSION_KEY, entry_id: 'a' })

    expect(removeQueuedPrompt(SESSION_KEY, 'missing')).toBe(false)
  })

  it('promoteQueuedPrompt moves the entry to the head and forwards interrupt', () => {
    setSessionQueue(SESSION_KEY, [
      { id: 'a', text: 'first' },
      { id: 'b', text: 'second' },
      { id: 'c', text: 'third' }
    ])

    expect(promoteQueuedPrompt(SESSION_KEY, 'c', { interrupt: true })).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(e => e.id)).toEqual(['c', 'a', 'b'])
    expect(requester).toHaveBeenCalledWith('session.queue.promote', {
      session_id: SESSION_KEY,
      entry_id: 'c',
      interrupt: true
    })

    // Head entry: still accepted (fires the RPC so an idle gateway can drain).
    expect(promoteQueuedPrompt(SESSION_KEY, 'c')).toBe(true)
    expect(promoteQueuedPrompt(SESSION_KEY, 'missing')).toBe(false)
  })

  it('updateQueuedPromptText edits in place and fires the RPC', () => {
    setSessionQueue(SESSION_KEY, [{ id: 'a', text: 'before' }])

    expect(updateQueuedPromptText(SESSION_KEY, 'a', 'after')).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY)[0]?.text).toBe('after')
    expect(requester).toHaveBeenCalledWith('session.queue.update', {
      session_id: SESSION_KEY,
      entry_id: 'a',
      text: 'after'
    })

    expect(updateQueuedPromptText(SESSION_KEY, 'a', 'after')).toBe(false)
  })

  it('clearQueuedPrompts wipes the mirror and fires the RPC', () => {
    setSessionQueue(SESSION_KEY, [{ id: 'a', text: 'first' }])
    addPendingSteer(SESSION_KEY, 'nudge')

    clearQueuedPrompts(SESSION_KEY)

    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
    expect(getPendingSteers(SESSION_KEY)).toEqual([])
    expect(requester).toHaveBeenCalledWith('session.queue.clear', { session_id: SESSION_KEY })
  })

  it('migrateQueuedPrompts re-keys the local mirror on a runtime id change', () => {
    setSessionQueue('rt-old', [{ id: 'a', text: 'stranded' }])
    setSessionQueue('rt-new', [{ id: 'b', text: 'already here' }])

    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(true)
    expect(getQueuedPrompts('rt-old')).toEqual([])
    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['already here', 'stranded'])

    expect(migrateQueuedPrompts('rt-new', 'rt-new')).toBe(false)
    expect(migrateQueuedPrompts('rt-empty', 'rt-new')).toBe(false)
  })

  describe('pending steers', () => {
    it('tracks a steer until steer.applied settles it', () => {
      addPendingSteer(SESSION_KEY, 'go left')
      addPendingSteer(SESSION_KEY, 'then right')

      expect(getPendingSteers(SESSION_KEY).map(s => s.text)).toEqual(['go left', 'then right'])

      // The agent concatenates queued steers with newlines before injecting —
      // one applied event can cover both.
      settlePendingSteer(SESSION_KEY, 'go left\nthen right')

      expect(getPendingSteers(SESSION_KEY)).toEqual([])
    })

    it('settles only matching entries', () => {
      addPendingSteer(SESSION_KEY, 'go left')
      addPendingSteer(SESSION_KEY, 'stay')

      settlePendingSteer(SESSION_KEY, 'go left')

      expect(getPendingSteers(SESSION_KEY).map(s => s.text)).toEqual(['stay'])
    })
  })
})
