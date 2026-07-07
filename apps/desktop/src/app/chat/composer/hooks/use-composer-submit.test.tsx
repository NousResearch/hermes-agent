import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerSubmit } from './use-composer-submit'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

afterEach(cleanup)

function Harness({
  attachments = [],
  busy = false,
  onClarify,
  onQueue,
  onSubmit
}: {
  attachments?: unknown[]
  busy?: boolean
  onClarify?: (text: string) => boolean | Promise<boolean>
  onQueue: () => boolean
  onSubmit: (text: string) => boolean | Promise<boolean>
}) {
  const activeQueueSessionKeyRef = useRef<string | null>('stored-session')
  const draftRef = useRef('')
  const editorRef = useRef<HTMLDivElement | null>(null)

  const { submitDraft } = useComposerSubmit({
    activeQueueSessionKey: 'stored-session',
    activeQueueSessionKeyRef,
    attachments: attachments as never[],
    busy,
    canSteer: false,
    clearDraft: () => {
      draftRef.current = ''

      if (editorRef.current) {
        editorRef.current.textContent = ''
      }
    },
    disabled: false,
    draftRef,
    drainNextQueued: async () => false,
    editorRef,
    exitQueuedEdit: () => false,
    focusInput: () => undefined,
    implicitQueueDrainAllowed: () => false,
    inputDisabled: false,
    loadIntoComposer: text => {
      draftRef.current = text

      if (editorRef.current) {
        editorRef.current.textContent = text
      }
    },
    onCancel: vi.fn(),
    onSteer: undefined,
    onSubmit,
    onSubmitClarifyAnswer: onClarify,
    queueCurrentDraft: onQueue,
    queueEdit: null,
    sendBlocked: busy,
    sessionId: 'runtime-session',
    setComposerText: value => {
      draftRef.current = value
    },
    stashAt: vi.fn()
  })

  return (
    <>
      <div
        contentEditable
        data-testid="editor"
        onInput={event => {
          draftRef.current = event.currentTarget.textContent ?? ''
        }}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <button data-testid="submit" onClick={submitDraft} type="button" />
    </>
  )
}

describe('useComposerSubmit clarify routing', () => {
  it('answers an active clarify request from the main composer instead of queuing', async () => {
    const onClarify = vi.fn(async () => true)
    const onQueue = vi.fn(() => true)
    const onSubmit = vi.fn(() => true)

    const { getByTestId } = render(<Harness busy onClarify={onClarify} onQueue={onQueue} onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'כן, תמשיך'
      fireEvent.input(editor)
      fireEvent.click(getByTestId('submit'))
    })

    expect(onClarify).toHaveBeenCalledWith('כן, תמשיך')
    expect(onQueue).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps ordinary busy submits on the queue when there is no clarify request', async () => {
    const onQueue = vi.fn(() => true)
    const onSubmit = vi.fn(() => true)

    const { getByTestId } = render(<Harness busy onQueue={onQueue} onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'send after current run'
      fireEvent.input(editor)
      fireEvent.click(getByTestId('submit'))
    })

    expect(onQueue).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not route attachment submits through clarify answers', async () => {
    const onClarify = vi.fn(async () => true)
    const onQueue = vi.fn(() => true)
    const onSubmit = vi.fn(() => true)

    const { getByTestId } = render(
      <Harness attachments={[{ id: 'file-1' }]} busy onClarify={onClarify} onQueue={onQueue} onSubmit={onSubmit} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'see attached'
      fireEvent.input(editor)
      fireEvent.click(getByTestId('submit'))
    })

    expect(onClarify).not.toHaveBeenCalled()
    expect(onQueue).toHaveBeenCalledTimes(1)
  })
})
