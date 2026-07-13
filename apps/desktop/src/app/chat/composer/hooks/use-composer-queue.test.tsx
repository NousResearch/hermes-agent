import { act, renderHook, waitFor } from '@testing-library/react'
import { type RefObject } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts
} from '@/store/composer-queue'
import { profileSessionKey } from '@/store/session-identity'

import type { QueueEditState } from '../composer-utils'
import { composerQueueKeys, resolveComposerQueueScope } from '../queue-scope'

import { useComposerQueue } from './use-composer-queue'

describe('useComposerQueue re-keying', () => {
  beforeEach(() => {
    $queuedPromptsBySession.set({})
  })

  it('moves a runtime queue to its first stored identity FIFO exactly once', async () => {
    const runtimeKey = profileSessionKey('default', 'runtime-a')
    const storedKey = profileSessionKey('default', 'stored-a')
    enqueueQueuedPrompt(storedKey, { attachments: [], text: 'already stored' })
    enqueueQueuedPrompt(runtimeKey, { attachments: [], text: 'runtime first' })
    enqueueQueuedPrompt(runtimeKey, { attachments: [], text: 'runtime second' })

    const draftRef = { current: '' }
    const queueEditRef: RefObject<QueueEditState | null> = { current: null }

    const baseArgs = {
      attachments: [] as ComposerAttachment[],
      busy: true,
      clearDraft: vi.fn(),
      draftRef,
      focusInput: vi.fn(),
      loadIntoComposer: vi.fn(),
      onCancel: vi.fn(),
      onSubmit: vi.fn(() => true),
      queueEditRef,
      sessionId: 'runtime-a'
    }

    const runtimeScope = resolveComposerQueueScope(composerQueueKeys('default', null, 'runtime-a'))
    const storedScope = resolveComposerQueueScope(composerQueueKeys('default', 'stored-a', 'runtime-a'))

    const { rerender } = renderHook(
      ({ queueScope }) =>
        useComposerQueue({
          ...baseArgs,
          queueScope
        }),
      { initialProps: { queueScope: runtimeScope } }
    )

    rerender({ queueScope: storedScope })

    await waitFor(() => {
      expect(getQueuedPrompts(runtimeKey)).toEqual([])
      expect(getQueuedPrompts(storedKey).map(entry => entry.text)).toEqual([
        'already stored',
        'runtime first',
        'runtime second'
      ])
    })

    const entryIds = getQueuedPrompts(storedKey).map(entry => entry.id)

    act(() => rerender({ queueScope: storedScope }))

    expect(getQueuedPrompts(storedKey).map(entry => entry.id)).toEqual(entryIds)
  })
})
