import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $clarifyRequests } from '@/store/clarify'
import {
  clearSessionDraft,
  type ComposerAttachment,
  SESSION_DRAFTS_STORAGE_KEY,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import { $gateway } from '@/store/gateway'

import { useComposerSubmit } from './use-composer-submit'

interface SubmitHarnessOptions {
  attachments?: ComposerAttachment[]
  busy?: boolean
  compacting?: boolean
  text?: string
  /** Defaults to a mock; pass a store-backed writer to exercise the real
   *  draft-stash merge protection end to end. */
  stashAt?: (
    scope: string | null,
    text?: string,
    attachments?: ComposerAttachment[],
    opts?: { pending?: boolean }
  ) => void
}

function renderSubmitHook({
  attachments = [],
  busy = false,
  compacting = false,
  text = '',
  stashAt = vi.fn()
}: SubmitHarnessOptions = {}) {
  const draftRef = { current: text }
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = text
  const editorRef = { current: editor }
  const onCancel = vi.fn()
  const onSteer = vi.fn(async () => true)
  const onSubmit = vi.fn(async () => true)
  const queueCurrentDraft = vi.fn(() => true)
  const loadIntoComposer = vi.fn()
  // Mutable so a test can simulate the user switching sessions while the
  // send is in flight (dispatchSubmit reads it at resolve time).
  const activeQueueSessionKeyRef = { current: 'stored-session' }

  const clearDraft = vi.fn(() => {
    draftRef.current = ''
    editorRef.current!.textContent = ''
  })

  const hook = renderHook(() =>
    useComposerSubmit({
      activeQueueSessionKey: 'stored-session',
      activeQueueSessionKeyRef,
      attachments,
      busy,
      compacting,
      clearDraft,
      disabled: false,
      draftRef,
      drainNextQueued: vi.fn(async () => false),
      editorRef,
      exitQueuedEdit: vi.fn(() => false),
      focusInput: vi.fn(),
      inputDisabled: false,
      loadIntoComposer,
      onCancel,
      onSteer,
      onSubmit,
      queueCurrentDraft,
      queueEdit: null,
      queuedPrompts: [],
      sessionId: 'runtime-session',
      setComposerText: vi.fn(),
      stashAt
    })
  )

  return {
    activeQueueSessionKeyRef,
    clearDraft,
    hook,
    loadIntoComposer,
    onCancel,
    onSteer,
    onSubmit,
    queueCurrentDraft,
    stashAt
  }
}

describe('useComposerSubmit busy-turn routing', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('steers a plain-text follow-up instead of queueing or stopping', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'change course'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('queues a plain-text follow-up while the active turn is compacting', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      compacting: true,
      text: 'wait for the summary'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('runs slash commands immediately while busy', async () => {
    const { clearDraft, hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: '/compress preserve context'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('/compress preserve context', { composerScope: 'stored-session' })
    )
    expect(clearDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('queues an attachment-bearing follow-up while busy', () => {
    const attachment: ComposerAttachment = { id: 'doc', kind: 'file', label: 'notes.txt' }

    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      attachments: [attachment],
      busy: true,
      text: 'read this'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('stops an active turn only with an empty composer', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ busy: true })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
  })

  it('submits a normal turn while idle', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ text: 'ordinary question' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('ordinary question', {
        attachments: [],
        composerScope: 'stored-session'
      })
    )
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('threads the loaded composer scope through onSubmit for the #59305 submit-time guard', async () => {
    const { hook, onSubmit } = renderSubmitHook({ text: 'hello' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('hello', expect.objectContaining({ composerScope: 'stored-session' }))
    )
  })
})

describe('useComposerSubmit with a clarify parked on the session', () => {
  const gatewayRequest = vi.fn(async () => ({ ok: true }))

  const parkClarify = (sessionId: string) => {
    $clarifyRequests.set({
      [sessionId]: { requestId: `req-${sessionId}`, question: 'which one?', choices: ['a', 'b'], sessionId }
    })
    $gateway.set({ request: gatewayRequest } as unknown as ReturnType<typeof $gateway.get>)
  }

  afterEach(() => {
    cleanup()
    gatewayRequest.mockClear()
    $clarifyRequests.set({})
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  it('skips the question and still sends the typed message on an idle session', async () => {
    parkClarify('runtime-session')
    const { hook, onSubmit } = renderSubmitHook({ text: 'actually do this instead' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(gatewayRequest).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-runtime-session',
        answer: ''
      })
    )
    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('actually do this instead', expect.objectContaining({ attachments: [] }))
    )
    expect($clarifyRequests.get()['runtime-session']).toBeUndefined()
  })

  it('skips the question before steering a busy turn', async () => {
    parkClarify('runtime-session')
    const { hook, onSteer } = renderSubmitHook({ busy: true, text: 'change course' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(gatewayRequest).toHaveBeenCalledWith('clarify.respond', { request_id: 'req-runtime-session', answer: '' })
  })

  it('leaves the question alone for an empty Enter (Stop, not an answer)', () => {
    parkClarify('runtime-session')
    const { hook, onCancel } = renderSubmitHook({ busy: true })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(gatewayRequest).not.toHaveBeenCalled()
    expect($clarifyRequests.get()['runtime-session']).toBeDefined()
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it("leaves another session's question alone", async () => {
    parkClarify('other-session')
    const { hook, onSubmit } = renderSubmitHook({ text: 'unrelated message' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    expect(gatewayRequest).not.toHaveBeenCalled()
    expect($clarifyRequests.get()['other-session']).toBeDefined()
  })
})

describe('useComposerSubmit uncertain / late send outcomes', () => {
  afterEach(() => {
    cleanup()
    clearSessionDraft('stored-session')
    vi.restoreAllMocks()
  })

  it("an 'uncertain' outcome stashes the draft for recovery but never resurrects it into the composer — and never clears the stored draft", async () => {
    // The gateway may have accepted the prompt (response lost to a transport
    // drop / timeout). Resurrecting the text into the live composer would be
    // the "sent AND still in the composer" bug — a duplicate-send trap. The
    // words must survive only in the draft stash.
    const { hook, loadIntoComposer, onSubmit, stashAt } = renderSubmitHook({ text: 'ordinary question' })

    // Seed the persisted draft BEFORE the send, and prove the component leaves
    // it untouched: a genuinely lost send stays recoverable on reload (the
    // reload-restores-the-draft behavior Riyaaz observed) — clearSessionDraft
    // must NOT fire on 'uncertain'.
    stashSessionDraft('stored-session', 'ordinary question', [])

    onSubmit.mockResolvedValueOnce('uncertain' as never)

    act(() => {
      hook.result.current.submitDraft()
    })

    // The pending marker is the ownership contract: the recovery stash is
    // protected from stale EMPTY cleanup writes until taken/cleared.
    await waitFor(() =>
      expect(stashAt).toHaveBeenCalledWith('stored-session', 'ordinary question', expect.arrayContaining([]), {
        pending: true
      })
    )
    expect(loadIntoComposer).not.toHaveBeenCalled()
    expect(takeSessionDraft('stored-session').text).toBe('ordinary question')
  })

  it('a successful send clears the persisted draft for good', async () => {
    // The draft store must not outlive a successful send — otherwise reload
    // resurrects text the user already sent (the cmd+R duplicate trap).
    const { hook, onSubmit } = renderSubmitHook({ text: 'sent words' })

    stashSessionDraft('stored-session', 'sent words', [])

    onSubmit.mockResolvedValueOnce(true)

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(takeSessionDraft('stored-session').text).toBe(''))
  })

  it('a definitive rejection after a session switch stashes under the submitted scope without painting into the new session', async () => {
    // The send is rejected AFTER the user moved to another session. The
    // rejected text must land in the SUBMITTED session's stash (recoverable
    // when the user returns) and must NOT be painted into the composer that
    // is now showing the other session's draft.
    const { activeQueueSessionKeyRef, hook, loadIntoComposer, onSubmit, stashAt } = renderSubmitHook({
      text: 'rejected words'
    })

    onSubmit.mockResolvedValueOnce(false)

    act(() => {
      hook.result.current.submitDraft()
    })

    activeQueueSessionKeyRef.current = 'other-session'

    await waitFor(() => expect(stashAt).toHaveBeenCalled())
    expect(stashAt).toHaveBeenCalledWith('stored-session', 'rejected words', expect.arrayContaining([]))
    expect(loadIntoComposer).not.toHaveBeenCalled()
  })

  it("an uncertain send survives the later session switch — the empty stash-on-leave cannot wipe the recovery stash", async () => {
    // Regression coverage: the uncertain path stashes
    // the words under the submitted scope; when the user then switches
    // sessions, the swap cleanup stashes the now-EMPTY composer over the same
    // key. Pre-fix that deleted the recovery stash and the user's words were
    // gone for good. The pending marker must protect it.
    const storeStashAt = (
      scope: string | null,
      text?: string,
      attachments?: ComposerAttachment[],
      opts?: { pending?: boolean }
    ) => stashSessionDraft(scope, text ?? '', attachments ?? [], opts)

    const { hook, loadIntoComposer, onSubmit } = renderSubmitHook({
      text: 'must remain recoverable',
      stashAt: storeStashAt
    })

    onSubmit.mockResolvedValueOnce('uncertain' as never)

    act(() => {
      hook.result.current.submitDraft()
    })

    // The recovery stash is persisted (take would consume the pending marker,
    // so probe the persisted store instead).
    await waitFor(() =>
      expect(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY)).toContain('must remain recoverable')
    )

    // The session switch: the swap cleanup writes the (empty) composer over
    // the same key — exactly what deleted the recovery stash pre-fix.
    stashSessionDraft('stored-session', '', [])

    expect(takeSessionDraft('stored-session').text).toBe('must remain recoverable')
    expect(loadIntoComposer).not.toHaveBeenCalled()
  })
})
