import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  clearSessionDraft,
  mainComposerScope,
  rotateFreshDraftKey,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'

import { DRAFT_PERSIST_DEBOUNCE_MS, type QueueEditState } from '../composer-utils'
import { composerPlainText, RICH_INPUT_SLOT } from '../rich-editor'

import { useComposerDraft } from './use-composer-draft'
import { useComposerSubmit } from './use-composer-submit'

// Only the assistant-ui adapter is substituted; the draft hook, the submit
// hook and the shared stash are the real ones.
const runtime = vi.hoisted(() => {
  let text = ''
  const listeners = new Set<() => void>()

  const setText = vi.fn((next: string) => {
    text = next
    listeners.forEach(listener => listener())
  })

  return {
    composer: { setText },
    getState: () => ({ text }),
    reset: () => {
      text = ''
      listeners.clear()
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    }
  }
})

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => runtime.composer }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: runtime.getState() }),
  useComposerRuntime: () => runtime
}))

const SCOPE = 'accept-clear-session'
const OTHER = 'accept-clear-other'
const queueEditRef = { current: null as QueueEditState | null }
let submitDraft: () => void
let accept: (accepted: boolean) => void

function Harness({ scope }: { scope: string }) {
  const draft = useComposerDraft({
    activeQueueSessionKey: scope,
    focusKey: null,
    inputDisabled: false,
    queueEditRef,
    sessionId: scope
  })

  const submit = useComposerSubmit({
    ...draft,
    activeQueueSessionKey: scope,
    attachments: [],
    busy: false,
    disabled: false,
    drainNextQueued: async () => false,
    exitQueuedEdit: () => false,
    inputDisabled: false,
    onCancel: () => {},
    onSteer: async () => true,
    onSteerHidden: async () => true,
    onSubmit: () => new Promise<boolean>(resolve => (accept = resolve)),
    queueCurrentDraft: () => false,
    queueEdit: null,
    queuedPrompts: [],
    sessionId: scope
  })

  submitDraft = submit.submitDraft

  return (
    <div
      contentEditable
      data-slot={RICH_INPUT_SLOT}
      data-testid="editor"
      ref={draft.editorRef}
      suppressContentEditableWarning
    />
  )
}

function typeAndSave(editor: HTMLElement, text: string) {
  editor.textContent = text
  editor.dispatchEvent(new Event('input', { bubbles: true }))
  act(() => {
    runtime.composer.setText(text)
  })
  act(() => void vi.advanceTimersByTime(DRAFT_PERSIST_DEBOUNCE_MS + 50))
}

beforeEach(() => {
  vi.useFakeTimers()
  runtime.reset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  mainComposerScope.clear()
  clearSessionDraft(SCOPE)
  clearSessionDraft(OTHER)
})

it('a send accepted after the next message was typed and saved keeps that message', async () => {
  const { getByTestId, rerender } = render(<Harness scope={SCOPE} />)
  const editor = getByTestId('editor')

  typeAndSave(editor, 'first message')
  act(() => submitDraft())
  typeAndSave(editor, 'next message')
  expect(takeSessionDraft(SCOPE).text).toBe('next message')

  // The gateway's accept for "first message" lands only now.
  await act(async () => accept(true))

  rerender(<Harness scope={OTHER} />)
  rerender(<Harness scope={SCOPE} />)
  expect(takeSessionDraft(SCOPE).text).toBe('next message')
  expect(composerPlainText(editor).trim()).toBe('next message')
})

it('switching away and back during a live send does not offer its outbox entry as a draft', async () => {
  const { getByTestId, rerender } = render(<Harness scope={SCOPE} />)
  const editor = getByTestId('editor')
  typeAndSave(editor, 'sent in A')
  act(() => submitDraft())
  rerender(<Harness scope={OTHER} />)
  typeAndSave(editor, 'B own draft')
  rerender(<Harness scope={SCOPE} />)
  expect(composerPlainText(editor).trim()).toBe('')
  await act(async () => accept(true))
  rerender(<Harness scope={OTHER} />)
  expect(composerPlainText(editor).trim()).toBe('B own draft')
  rerender(<Harness scope={SCOPE} />)
  expect(composerPlainText(editor).trim()).toBe('')
})

it('a late fresh-chat rejection restores only the original fresh lifecycle', async () => {
  const first = rotateFreshDraftKey()
  const { getByTestId, rerender } = render(<Harness scope={first} />)
  const editor = getByTestId('editor')
  typeAndSave(editor, 'first chat message')
  act(() => submitDraft())
  const second = rotateFreshDraftKey()
  rerender(<Harness scope={second} />)
  typeAndSave(editor, 'second chat draft')
  await act(async () => accept(false))
  expect(composerPlainText(editor).trim()).toBe('second chat draft')
  expect(takeSessionDraft(second).text).toBe('second chat draft')
  expect(takeSessionDraft(first).text).toBe('first chat message')
  clearSessionDraft(first)
  clearSessionDraft(second)
})

it('restoring an old fresh draft does not change the active fresh draft save baseline', async () => {
  const first = rotateFreshDraftKey()
  const { getByTestId, rerender } = render(<Harness scope={first} />)
  const editor = getByTestId('editor')
  typeAndSave(editor, 'repeat these words')
  act(() => submitDraft())
  const second = rotateFreshDraftKey()
  rerender(<Harness scope={second} />)
  typeAndSave(editor, 'second chat draft')
  await act(async () => accept(false))
  // A new edit can legitimately repeat the old send. Closing before debounce
  // must still save it; an offscreen restore must not mark it already saved.
  editor.textContent = 'repeat these words'
  act(() => runtime.composer.setText('repeat these words'))
  act(() => window.dispatchEvent(new Event('pagehide')))
  expect(takeSessionDraft(second).text).toBe('repeat these words')
  clearSessionDraft(first)
  clearSessionDraft(second)
})

it('a late fresh-chat acceptance must not clear an identical draft in another fresh lifecycle', async () => {
  const first = rotateFreshDraftKey()
  const { getByTestId, rerender } = render(<Harness scope={first} />)
  const editor = getByTestId('editor')
  typeAndSave(editor, 'continue')
  act(() => submitDraft())
  const second = rotateFreshDraftKey()
  stashSessionDraft(second, 'continue', [])
  rerender(<Harness scope={second} />)
  await act(async () => accept(true))
  expect(takeSessionDraft(second).text).toBe('continue')
  clearSessionDraft(first)
  clearSessionDraft(second)
})
