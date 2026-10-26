import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearSessionDraft, mainComposerScope, stashSessionDraft, takeSessionDraft } from '@/store/composer'

import { TYPING_STORE_IDLE_MS } from '../composer-store-writer'
import { DRAFT_PERSIST_DEBOUNCE_MS, type QueueEditState } from '../composer-utils'
import { RICH_INPUT_SLOT } from '../rich-editor'

import { useComposerDraft } from './use-composer-draft'

// A composer runtime that behaves like assistant-ui's: setText updates the
// state and notifies subscribers synchronously.
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
      setText.mockClear()
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
  useComposerRuntime: () => ({ getState: runtime.getState, subscribe: runtime.subscribe })
}))

const SCOPE = 'session-persist-window'
let draft: ReturnType<typeof useComposerDraft>

function Harness() {
  draft = useComposerDraft({
    activeQueueSessionKey: SCOPE,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId: SCOPE
  })

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

const editorText = (editor: HTMLElement) =>
  editor.ownerDocument.createTreeWalker(editor, NodeFilter.SHOW_TEXT).nextNode() as Text

describe('useComposerDraft — the draft stash window starts at the keystroke', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    runtime.reset()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    mainComposerScope.clear()
    clearSessionDraft(SCOPE)
  })

  it('stashes typed text one persist debounce after the keystroke, while its store write still coalesces', () => {
    stashSessionDraft(SCOPE, 'hello', [])
    const { getByTestId } = render(<Harness />)
    const editor = getByTestId('editor')

    // A keystroke that flips no composer-text edge: the store write waits for a typing pause.
    const typed = editorText(editor)
    typed.appendData(' draft')
    act(() => {
      draft.syncDraftFromEditor(true)
    })
    expect(runtime.composer.setText).not.toHaveBeenCalledWith('hello draft')

    act(() => {
      vi.advanceTimersByTime(TYPING_STORE_IDLE_MS)
    })
    expect(runtime.composer.setText).toHaveBeenLastCalledWith('hello draft')

    // Closing the window right after the debounce must find the draft saved;
    // the coalesced store write landing later does not restart the window.
    act(() => {
      vi.advanceTimersByTime(DRAFT_PERSIST_DEBOUNCE_MS - TYPING_STORE_IDLE_MS)
    })
    expect(takeSessionDraft(SCOPE).text).toBe('hello draft')
  })

  it('stashes the latest keystroke synchronously when the window starts to close', () => {
    stashSessionDraft(SCOPE, 'hello', [])
    const { getByTestId } = render(<Harness />)
    const typed = editorText(getByTestId('editor'))

    // Typed, and not even the rAF flush has run: only the DOM has the text.
    typed.appendData(' closing')
    window.dispatchEvent(new Event('beforeunload'))

    expect(takeSessionDraft(SCOPE).text).toBe('hello closing')
  })

  it('does not write a draft another window sent back when this untouched window closes', () => {
    stashSessionDraft(SCOPE, 'sent from the other window', [])
    render(<Harness />)
    act(() => {
      vi.advanceTimersByTime(DRAFT_PERSIST_DEBOUNCE_MS)
    })

    // The other window sends it: the shared stash clears, this editor still shows the old text.
    clearSessionDraft(SCOPE)
    window.dispatchEvent(new Event('beforeunload'))

    expect(takeSessionDraft(SCOPE).text).toBe('')
  })
})
