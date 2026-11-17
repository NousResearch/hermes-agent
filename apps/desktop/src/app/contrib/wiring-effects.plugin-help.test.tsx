import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { freshDraftScope, rotateFreshDraftKey, takeSessionDraft } from '@/store/composer'
import { $pendingPluginHelpDraft, queuePluginHelpDraft } from '@/store/plugin-install-request'
import { requestFreshSession } from '@/store/profile'

import { useFreshSessionRequest } from './wiring-effects'

afterEach(() => cleanup())

it('places user-requested help in the newly rotated fresh chat, not the previous draft', () => {
  const old = rotateFreshDraftKey()

  const startFresh = vi.fn(() => {
    rotateFreshDraftKey()
  })

  renderHook(() => useFreshSessionRequest(startFresh))
  act(() => {
    queuePluginHelpDraft('Help with Godot 4 on the backend')
    requestFreshSession()
  })
  expect(startFresh).toHaveBeenCalledTimes(1)
  expect($pendingPluginHelpDraft.get()).toBeNull()
  expect(takeSessionDraft(old).text).toBe('')
  expect(takeSessionDraft(freshDraftScope()).text).toBe('Help with Godot 4 on the backend')
})
