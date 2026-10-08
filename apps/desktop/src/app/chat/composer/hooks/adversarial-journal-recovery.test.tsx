import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { preparedSubmissionKey, readPreparedSubmission, writePreparedSubmission } from '@/app/session/hooks/use-prompt-actions/prepared-submissions'
import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'

import { useComposerSubmit } from './use-composer-submit'

const destination = { scopeKey: 'test-owner', owner: { profile: 'test' } } as never

function composer(onSubmit: (text: string, options?: SubmitTextOptions) => Promise<boolean>) {
  return renderHook(() => useComposerSubmit({
    activeQueueSessionKey: 's', activeQueueSessionKeyRef: { current: 's' }, attachments: [],
    busy: false, clearDraft: vi.fn(), disabled: false,
    draftScopeRef: { current: 's' }, draftRef: { current: 'same text' },
    drainNextQueued: vi.fn(async () => false), editorRef: { current: null },
    exitQueuedEdit: vi.fn(() => false), focusInput: vi.fn(), inputDisabled: false,
    loadIntoComposer: vi.fn(), onCancel: vi.fn(), onSteer: vi.fn(), onSteerHidden: vi.fn(),
    onSubmit, queueCurrentDraft: vi.fn(() => false), queueEdit: null, queuedPrompts: [],
    sessionId: 's', setComposerText: vi.fn(), stashAt: vi.fn()
  }))
}

afterEach(() => { cleanup(); localStorage.clear(); sessionStorage.clear() })

test.each([false, true])('restored failed draft reuses its durable admission after composer remount (queue fallback: %s)', async fromQueue => {
  let firstId = ''
  const admissions = new Set<string>()

  const onSubmit = vi.fn(async (text: string, options?: SubmitTextOptions) => {
    const key = preparedSubmissionKey('s', destination, text, [], options)
    const saved = await readPreparedSubmission(key)
    const id = saved?.id ?? options!.submission_id!

    if (!firstId) {firstId = id}
    await writePreparedSubmission(key, { id, owner: {} as never, text, attachments: [], params: { text } })
    admissions.add(id) // Owner admission succeeded but the response was lost.

    return false
  })

  const first = composer(onSubmit)
  await act(async () => { first.result.current.dispatchSubmit('same text', [], undefined,
    fromQueue ? { fromQueue: true, storedSessionId: 's', sessionId: 's' } : undefined) })
  first.unmount()
  const restored = composer(onSubmit)
  await act(async () => { restored.result.current.dispatchSubmit('same text') })
  expect(onSubmit.mock.calls[1][1]!.submission_id).toBe(firstId)
  expect(admissions.size).toBe(1)
})
