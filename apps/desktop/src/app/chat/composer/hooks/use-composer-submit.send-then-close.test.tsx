import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useLayoutEffect, useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ChatBarProps } from '../types'

// The page is real (draft DOM, persistence, submit hook); only the assistant-ui adapter is
// substituted. A "window close" is the page's own pagehide flush plus a fresh module graph, so
// nothing the dead page held in memory survives into the next load: only localStorage does.
const composer = vi.hoisted(() => {
  let text = ''
  const listeners = new Set<() => void>()

  const runtime = {
    getState: () => ({ text }),
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    }
  }

  const api = {
    setText: (value: string) => {
      text = value
      listeners.forEach(listener => listener())
    }
  }

  return { aui: { composer: () => api }, runtime }
})

// The stored transcript the owner serves for a session (the REST page the outbox reads on reopen).
// `gate` holds the owner's answer (the recovery check is asynchronous) until a test releases it.
const transcript = vi.hoisted(() => ({ gate: Promise.resolve(), rows: new Map<string, unknown[]>() }))

vi.mock('@/api/sessions', async importOriginal => ({
  ...(await importOriginal<object>()),
  getSessionMessages: async (id: string) => ({
    messages: await transcript.gate.then(() => transcript.rows.get(id) ?? []),
    pagination: { limit: 200, offset: 0, order: 'latest' },
    session_id: id
  })
}))

vi.mock('@assistant-ui/react', () => ({
  useAui: () => composer.aui,
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: composer.runtime.getState() }),
  useComposerRuntime: () => composer.runtime
}))

interface Handles {
  attach: (id: string) => void
  detach: (id: string) => void
  editor: () => string
  submitDraft: () => void
}

interface PageOptions {
  /** A turn is running: Enter goes through the busy path (steer by default, or the queue). */
  busy?: boolean
  onSteer?: ChatBarProps['onSteer']
  queueCurrentDraft?: (submissionId?: string) => boolean
}

let handles: Handles

// One app page: fresh imports of the composer hooks and stores (as a reload or relaunch gets).
// Opening the session resumes it: the owner's snapshot lists its pending admissions (`pending`).
async function openPage(onSubmit: ChatBarProps['onSubmit'], pending: unknown[] = [], options: PageOptions = {}) {
  const { reconcilePendingSubmissions } = await import('@/store/pending-submissions')
  const { useComposerDraft } = await import('./use-composer-draft')
  const { useComposerSubmit } = await import('./use-composer-submit')
  const { mainComposerScope } = await import('@/store/composer')
  const { composerPlainText } = await import('../rich-editor')

  function Page() {
    const attachments = useStore(mainComposerScope.$attachments)
    const queueEditRef = useRef(null)

    const draft = useComposerDraft({
      activeQueueSessionKey: 'stored-B',
      focusKey: 'rt-B',
      inputDisabled: false,
      queueEditRef,
      sessionId: 'rt-B'
    })

    const submit = useComposerSubmit({
      ...draft,
      activeQueueSessionKey: 'stored-B',
      attachments,
      busy: options.busy ?? false,
      disabled: false,
      inputDisabled: false,
      drainNextQueued: async () => false,
      exitQueuedEdit: () => false,
      onCancel: () => {},
      onSteer: options.onSteer ?? (async () => false),
      onSteerHidden: async () => false,
      onSubmit,
      queueCurrentDraft: options.queueCurrentDraft ?? (() => false),
      queueEdit: null,
      queuedPrompts: [],
      sessionId: 'rt-B'
    })

    useLayoutEffect(() => {
      handles = {
        attach: id => mainComposerScope.add({ id, kind: 'url', label: id, refText: `@url:${id}` }),
        detach: id => mainComposerScope.remove(id),
        editor: () => composerPlainText(draft.editorRef.current!).trim(),
        submitDraft: submit.submitDraft
      }
    })

    return <div contentEditable data-testid="editor" ref={draft.editorRef} />
  }

  const page = render(<Page />)
  reconcilePendingSubmissions('stored-B', pending)

  return page
}

function closeWindow() {
  act(() => window.dispatchEvent(new Event('pagehide')))
  cleanup()
  vi.resetModules()
}

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  transcript.rows.clear()
  transcript.gate = Promise.resolve()
  vi.resetModules()
})

// Storage.key(i), not Object.keys: the test environment's localStorage enumerates no stored keys.
const outboxKeys = () =>
  Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index) ?? '').filter(key =>
    key.startsWith('hermes:composer-outbox:')
  )

// The notice that a restored send came back without its attachments (this page's notification store).
const attachmentsNotice = async () =>
  (await import('@/store/notifications')).$notifications
    .get()
    .find(notice => notice.id === 'restored-draft-attachments')?.message

// Seeds the draft as a first launch reads it: the v3 list counts only before any page wrote the v4 stash.
const saveDraft = (text: string) => {
  window.localStorage.removeItem('hermes:composer-drafts:v4')
  window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ 'stored-B': text }))
}

// Enter on `text` (with a url attachment), then the window closes before prompt.submit: the id it sent.
async function sendThenClose(text: string): Promise<string | undefined> {
  saveDraft(text)
  const cutOff = vi.fn<ChatBarProps['onSubmit']>(() => new Promise<boolean>(() => {}))
  await openPage(cutOff)
  act(() => handles.attach('example.com/a'))
  act(() => handles.submitDraft())
  closeWindow()

  return cutOff.mock.calls[0][1]?.submission_id
}

it('a message sent right before the window closes is never lost and never sent twice', async () => {
  saveDraft('answered question')

  // Control: an acknowledged send leaves nothing behind for the next page.
  await openPage(async () => true)
  expect(handles.editor()).toBe('answered question')
  await act(async () => handles.submitDraft())
  closeWindow()

  const relaunched = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(relaunched)
  await act(async () => {})
  expect(handles.editor()).toBe('')
  expect(await attachmentsNotice()).toBeUndefined()
  closeWindow()

  // Enter, then the window closes before the send chain reaches prompt.submit.
  saveDraft('my long question')
  const cutOff = vi.fn<ChatBarProps['onSubmit']>(() => new Promise<boolean>(() => {}))
  await openPage(cutOff)
  act(() => handles.submitDraft())
  expect(cutOff).toHaveBeenCalledTimes(1)
  const firstId = cutOff.mock.calls[0][1]?.submission_id
  expect(handles.editor()).toBe('')
  closeWindow()

  const reopened = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(reopened)
  await waitFor(() => expect(handles.editor()).toBe('my long question'))

  // Sending it again is the explicit retry of the same admission, so the owner runs it once.
  await act(async () => handles.submitDraft())
  expect(reopened).toHaveBeenCalledTimes(1)
  expect(reopened.mock.calls[0][0].trim()).toBe('my long question')
  expect(reopened.mock.calls[0][1]?.submission_id).toBe(firstId)
}, 30_000)

it('a send the owner admitted before the window closed is never offered again as the draft', async () => {
  // Admitted and run (its stored user row names it), admitted and still queued (the resume snapshot
  // lists it), or listed only by a snapshot that came after the draft was offered.
  const queued = (id: string, text: string) => [{ admission_id: 'adm-1', input_id: id, status: 'queued', text }]

  const landed = {
    late: (id: string) => ({ late: queued(id, 'late one') }),
    queued: (id: string) => ({ pending: queued(id, 'queued one') }),
    ran: (id: string) => ({ row: { content: 'ran one', display_metadata: { submission_id: id }, role: 'user' } })
  }

  const cases = [
    ['ran one', landed.ran],
    ['queued one', landed.queued],
    ['late one', landed.late]
  ] as const

  for (const [text, owner] of cases) {
    // Enter, the frame reaches the owner, and the window closes before the ack comes back.
    saveDraft(text)
    const unacked = vi.fn<ChatBarProps['onSubmit']>(() => new Promise<boolean>(() => {}))
    await openPage(unacked)
    act(() => handles.submitDraft())
    const held = owner(unacked.mock.calls[0][1]?.submission_id ?? '')
    closeWindow()

    transcript.rows.set('stored-B', 'row' in held ? [held.row] : [])
    const reopened = vi.fn<ChatBarProps['onSubmit']>(async () => true)
    await openPage(reopened, 'pending' in held ? held.pending : [])
    await waitFor(() => expect(outboxKeys()).toEqual([]))
    await act(async () => {})

    if ('late' in held) {
      expect(handles.editor()).toBe(text)
      const { reconcilePendingSubmissions } = await import('@/store/pending-submissions')
      act(() => reconcilePendingSubmissions('stored-B', held.late))
    }

    // Shown once (in the chat), never also back in the composer.
    expect(handles.editor()).toBe('')
    expect(reopened).not.toHaveBeenCalled()
    closeWindow()
    window.localStorage.clear()
  }
})

it('an interrupted send with attachments comes back once, under its id, even when the saved draft kept its words', async () => {
  // Offered back as text (the attachment did not outlive the page): Enter is still the retry.
  const offeredId = await sendThenClose('offered words')
  const offered = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(offered)
  await waitFor(() => expect(handles.editor()).toBe('offered words'))
  expect(await attachmentsNotice()).toBe(
    "Attachments from your unsent message weren't restored. Attach them again before you send it."
  )
  await act(async () => handles.submitDraft())
  expect(offered.mock.calls[0][1]?.submission_id).toBe(offeredId)
  closeWindow()
  window.localStorage.clear()

  // The saved draft persisted the words before the send cleared the composer (the close raced it).
  const savedId = await sendThenClose('saved words')
  saveDraft('saved words')
  const reopened = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(reopened)
  expect(handles.editor()).toBe('saved words')
  expect(await attachmentsNotice()).toBeTruthy()
  await act(async () => handles.submitDraft())
  expect(reopened.mock.calls[0][1]?.submission_id).toBe(savedId)
  closeWindow()

  // Never offered a second time beside the copy that was sent.
  const later = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(later)
  await waitFor(() => expect(outboxKeys()).toEqual([]))
  await act(async () => {})
  expect(handles.editor()).toBe('')
  closeWindow()
  window.localStorage.clear()

  // A send whose outcome is unknown (timed out) is given back with its id; after a reload the saved
  // draft has its words but not its attachment, and Enter on it still retries that send.
  saveDraft('unknown words')

  const timedOut = vi.fn<ChatBarProps['onSubmit']>(async () => {
    throw new Error('request timed out')
  })

  await openPage(timedOut)
  act(() => handles.attach('example.com/b'))
  await act(async () => handles.submitDraft())
  await waitFor(() => expect(handles.editor()).toBe('unknown words'))
  closeWindow()

  const retried = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(retried)
  expect(handles.editor()).toBe('unknown words')
  expect(await attachmentsNotice()).toBeTruthy()
  await act(async () => handles.submitDraft())
  expect(retried.mock.calls[0][1]?.submission_id).toBe(timedOut.mock.calls[0][1]?.submission_id)
})

it('a restored send retried while a turn runs is queued under its id, never steered as a new message', async () => {
  saveDraft('busy words')
  const cutOff = vi.fn<ChatBarProps['onSubmit']>(() => new Promise<boolean>(() => {}))
  await openPage(cutOff)
  act(() => handles.submitDraft())
  const firstId = cutOff.mock.calls[0][1]?.submission_id
  closeWindow()

  const onSteer = vi.fn(async () => true)
  const queueCurrentDraft = vi.fn((_submissionId?: string) => true)
  await openPage(async () => true, [], { busy: true, onSteer, queueCurrentDraft })
  await waitFor(() => expect(handles.editor()).toBe('busy words'))
  await act(async () => handles.submitDraft())
  expect(onSteer).not.toHaveBeenCalled()
  expect(queueCurrentDraft).toHaveBeenCalledWith(firstId)
})

it('recovery never discards a file the user attached while it was running', async () => {
  const attached = async () =>
    (await import('@/store/composer')).mainComposerScope.$attachments.get().map(attachment => attachment.id)

  // Attached while the owner check is still pending: the offer would replace the composer.
  await sendThenClose('held words')

  let answer = () => {}
  transcript.gate = new Promise<void>(resolve => (answer = resolve))
  await openPage(vi.fn<ChatBarProps['onSubmit']>(async () => true))
  act(() => handles.attach('new-file.txt'))
  await act(async () => answer())
  await act(async () => {})
  expect(await attached()).toEqual(['new-file.txt'])
  closeWindow()
  window.localStorage.clear()

  // Attached to the offered draft, then a late snapshot lists the original send: the take-back
  // would clear the whole composer.
  const lateId = await sendThenClose('late words')
  await openPage(vi.fn<ChatBarProps['onSubmit']>(async () => true))
  await waitFor(() => expect(handles.editor()).toBe('late words'))
  act(() => handles.attach('new-file.txt'))
  const { reconcilePendingSubmissions } = await import('@/store/pending-submissions')
  act(() =>
    reconcilePendingSubmissions('stored-B', [
      { admission_id: 'adm-1', input_id: lateId, status: 'queued', text: 'late words' }
    ])
  )
  await act(async () => {})
  expect(await attached()).toEqual(['new-file.txt'])
  expect(handles.editor()).toBe('late words')
})

it('equal words typed again after an interrupted send are a new message, never that send', async () => {
  const firstId = await sendThenClose('same words')
  const { at } = JSON.parse(window.localStorage.getItem(outboxKeys()[0])!) as { at: number }

  // The draft holding those words was saved after the send: the user typed them again.
  window.localStorage.setItem(
    'hermes:composer-drafts:v4',
    JSON.stringify({ drafts: { 'stored-B': { at: at + 1000, text: 'same words' } } })
  )
  const reopened = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(reopened)
  await act(async () => {})
  expect(handles.editor()).toBe('same words')
  await act(async () => handles.submitDraft())
  expect(reopened.mock.calls[0][1]?.submission_id).not.toBe(firstId)
}, 30_000)

it('a file removed from a given-back draft is never resent under the old send', async () => {
  saveDraft('trimmed words')

  const timedOut = vi.fn<ChatBarProps['onSubmit']>(async () => {
    throw new Error('request timed out')
  })

  await openPage(timedOut)
  act(() => handles.attach('example.com/c'))
  await act(async () => handles.submitDraft())
  await waitFor(() => expect(handles.editor()).toBe('trimmed words'))
  act(() => handles.detach('example.com/c'))
  closeWindow()

  const reopened = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  await openPage(reopened)
  await act(async () => handles.submitDraft())
  expect(reopened.mock.calls[0][1]?.submission_id).not.toBe(timedOut.mock.calls[0][1]?.submission_id)
  expect(await attachmentsNotice()).toBeUndefined()
}, 30_000)
