import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearSessionDraft, type ComposerAttachment, mainComposerScope } from '@/store/composer'

import type { QueueEditState } from '../composer-utils'
import { caretOffsetInEditor, placeCaretAtOffset, renderComposerContents, RICH_INPUT_SLOT } from '../rich-editor'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerDraft } from './use-composer-draft'

const mockComposerApi = { setText: vi.fn() }

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

type Restore = (text: string, attachments: ComposerAttachment[]) => void

function Harness({ onReady }: { onReady: (editor: HTMLDivElement | null, restore: Restore) => void }) {
  const { editorRef, loadIntoComposer } = useComposerDraft({
    activeQueueSessionKey: 'session-caret',
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId: 'session-caret'
  })

  return (
    <div
      contentEditable
      data-slot={RICH_INPUT_SLOT}
      ref={el => {
        editorRef.current = el
        onReady(el, loadIntoComposer)
      }}
      suppressContentEditableWarning
    />
  )
}

/** Mount the hook against a real contenteditable, then hand back the editor and
 *  the restore path with the caret parked mid-text — the state the user is in
 *  when a background tick re-runs a draft restore under them. */
function mountWithCaretAt(text: string, offset: number) {
  let editor: HTMLDivElement | null = null
  let restore: Restore = () => undefined

  render(
    <ComposerScopeProvider value={MAIN_COMPOSER_SCOPE}>
      <Harness
        onReady={(el, fn) => {
          editor = el
          restore = fn
        }}
      />
    </ComposerScopeProvider>
  )

  const el = editor as unknown as HTMLDivElement

  act(() => {
    renderComposerContents(el, text, { trailingCommitted: true })
    el.focus()
    placeCaretAtOffset(el, offset)
  })

  return { editor: el, restore }
}

// A restore that re-runs under a focused editor used to call
// renderComposerContents + placeCaretEnd unconditionally: the composer flashed
// and the caret landed at the bottom. Anything that re-triggers the restore on
// a timer (sessions.changed is floored to 2s server-side; a gateway reconnect
// storm re-runs it far faster) turned that into a caret jump every couple of
// seconds while typing.
describe('composer restore does not disturb a live caret', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('session-caret')
  })

  it('leaves the DOM and the caret untouched when the restored text is what the editor already shows', () => {
    const { editor, restore } = mountWithCaretAt('hello world', 5)
    const before = Array.from(editor.childNodes)

    act(() => restore('hello world', []))

    expect(Array.from(editor.childNodes)).toEqual(before)
    expect(caretOffsetInEditor(editor)).toBe(5)
  })

  it('keeps the caret where the user left it when a restore does change the text', () => {
    const { editor, restore } = mountWithCaretAt('hello world', 5)

    act(() => restore('hello world and then some', []))

    expect(caretOffsetInEditor(editor)).toBe(5)
  })
})
