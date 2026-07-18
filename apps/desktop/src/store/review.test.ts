import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $reviewLastTurnBaseRef,
  $reviewOpen,
  $reviewScope,
  bindReviewTurnBaseline,
  captureReviewTurnBaseline
} from './review'
import { $activeSessionId, $currentCwd, $selectedStoredSessionId } from './session'

const snapshot = vi.fn<(cwd: string, pin?: string) => Promise<null | string>>()
const releaseSnapshot = vi.fn<(cwd: string, pin: string) => Promise<{ ok: boolean }>>()

describe('last-turn review baseline', () => {
  beforeEach(() => {
    snapshot.mockReset()
    releaseSnapshot.mockReset()
    releaseSnapshot.mockResolvedValue({ ok: true })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      git: { review: { releaseSnapshot, snapshot } }
    }
    $currentCwd.set('/repo')
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $reviewOpen.set(true)
    $reviewScope.set('lastTurn')
  })

  afterEach(() => {
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $currentCwd.set('')
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = undefined
  })

  it('keeps baselines isolated when sessions share a repository', async () => {
    snapshot.mockResolvedValueOnce('tree-a').mockResolvedValueOnce('tree-b')

    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('stored-a')
    await captureReviewTurnBaseline()
    expect($reviewLastTurnBaseRef.get()).toBe('tree-a')

    $activeSessionId.set('runtime-b')
    $selectedStoredSessionId.set('stored-b')
    expect($reviewLastTurnBaseRef.get()).toBeNull()

    await captureReviewTurnBaseline()
    expect($reviewLastTurnBaseRef.get()).toBe('tree-b')

    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('stored-a')
    expect($reviewLastTurnBaseRef.get()).toBe('tree-a')
  })

  it('binds a fresh draft snapshot to its newly created runtime', async () => {
    snapshot.mockResolvedValueOnce('draft-tree')
    await captureReviewTurnBaseline()

    $activeSessionId.set('new-runtime')
    expect($reviewLastTurnBaseRef.get()).toBeNull()

    bindReviewTurnBaseline('new-runtime')
    expect($reviewLastTurnBaseRef.get()).toBe('draft-tree')

    $selectedStoredSessionId.set('new-stored')
    $activeSessionId.set(null)
    expect($reviewLastTurnBaseRef.get()).toBe('draft-tree')
  })

  it('does not hash the workspace when last-turn review tracking is inactive', async () => {
    $reviewOpen.set(false)

    await captureReviewTurnBaseline()

    expect(snapshot).not.toHaveBeenCalled()
    expect($reviewLastTurnBaseRef.get()).toBeNull()
  })

  it('reuses a session pin across turns and releases it when tracking clears', async () => {
    snapshot.mockResolvedValueOnce('tree-one').mockResolvedValueOnce('tree-two')
    $activeSessionId.set('runtime-pin-lifecycle')

    await captureReviewTurnBaseline()
    const pin = snapshot.mock.calls[0]?.[1]

    expect(pin).toBeTruthy()

    await captureReviewTurnBaseline()

    expect(snapshot.mock.calls[1]?.[1]).toBe(pin)
    expect(releaseSnapshot).not.toHaveBeenCalled()

    $reviewOpen.set(false)
    await captureReviewTurnBaseline()

    expect(releaseSnapshot).toHaveBeenCalledOnce()
    expect(releaseSnapshot).toHaveBeenCalledWith('/repo', pin)
  })
})
