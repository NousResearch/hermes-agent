import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $reviewLastTurnBaseRef,
  $reviewOpen,
  $reviewScope,
  bindReviewTurnBaseline,
  captureReviewTurnBaseline
} from './review'
import { $activeSessionId, $currentCwd, $selectedStoredSessionId } from './session'

const snapshot = vi.fn<(cwd: string) => Promise<null | string>>()

describe('last-turn review baseline', () => {
  beforeEach(() => {
    snapshot.mockReset()
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      git: { review: { snapshot } }
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
})
