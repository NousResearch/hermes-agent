import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { deriveProfileSessionStarted, useDraftProfileSelection } from './profile-selection'

describe('draft profile-selection lifecycle', () => {
  it('does not freeze an unsent blank preview merely because it has an internal runtime or stored id', () => {
    const preview = {
      hasMessages: false,
      hasPersistedSession: false,
      isSessionTile: false,
      runtimeId: 'preview-runtime',
      storedId: 'preview-stored-id'
    }

    expect(deriveProfileSessionStarted(preview)).toBe(false)

    const { result } = renderHook(() =>
      useDraftProfileSelection({
        draftKey: 'draft-preview-unsent',
        sessionStarted: deriveProfileSessionStarted(preview)
      })
    )

    expect(result.current.canSelectProfile).toBe(true)
  })

  it('disables selection during first submission and keeps it frozen once creation is attempted', () => {
    const { result } = renderHook(() =>
      useDraftProfileSelection({ draftKey: 'draft-first-send', sessionStarted: false })
    )

    expect(result.current.canSelectProfile).toBe(true)

    act(() => result.current.beginSubmission())
    expect(result.current.canSelectProfile).toBe(false)

    act(() => result.current.markStarted())
    expect(result.current.canSelectProfile).toBe(false)

    // A failed session.create must not reopen the selector after the draft
    // crossed the first-submission boundary; a new draft gets a new key.
    act(() => result.current.cancelBeforeStart())
    expect(result.current.canSelectProfile).toBe(false)

    const nextDraft = renderHook(() =>
      useDraftProfileSelection({ draftKey: 'draft-after-failure', sessionStarted: false })
    )

    expect(nextDraft.result.current.canSelectProfile).toBe(true)
  })

  it('keeps resumed sessions immutable even before their transcript is painted', () => {
    const { result } = renderHook(() =>
      useDraftProfileSelection({ draftKey: 'resumed-empty-session', sessionStarted: true })
    )

    expect(result.current.canSelectProfile).toBe(false)

    act(() => {
      result.current.beginSubmission()
      result.current.cancelBeforeStart()
    })

    expect(result.current.canSelectProfile).toBe(false)
  })

  it('retains the started boundary when the composer remounts during the request', () => {
    const original = renderHook(() =>
      useDraftProfileSelection({ draftKey: 'draft-remount-in-flight', sessionStarted: false })
    )

    act(() => original.result.current.markStarted())
    expect(original.result.current.canSelectProfile).toBe(false)
    original.unmount()

    const remounted = renderHook(() =>
      useDraftProfileSelection({ draftKey: 'draft-remount-in-flight', sessionStarted: false })
    )

    expect(remounted.result.current.canSelectProfile).toBe(false)
  })
})
