import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer } from './focus'
import { useComposerEscCancel } from './hooks/use-composer-esc-cancel'
import { useDictationEscCancel } from './hooks/use-dictation-esc-cancel'

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

// Faithful mirror of index.tsx's Escape wiring, driven through REAL DOM
// keydown events dispatched from a focused contentEditable — not synthetic
// window events. Esc has four claimants in the composer: the trigger popover
// (local, role="listbox"), queue-edit exit (local), turn halt (local +
// useComposerEscCancel on window), and dictation discard (capture-phase
// window listener). DESIGN.md: "One cancel gesture does one thing." A
// capture-phase preventDefault stops neither the local React handler nor the
// bubble listener from RUNNING — both must check `defaultPrevented`, and the
// dictation hook must yield to an open listbox. These tests pin exactly one
// action per keypress across all live combinations.
function Harness({
  busy,
  cancelDictation,
  closeTrigger,
  exitQueuedEdit,
  haltRun,
  queueEdit,
  recording,
  triggerOpen
}: {
  busy: boolean
  cancelDictation: () => void
  closeTrigger: () => void
  exitQueuedEdit: () => void
  haltRun: () => void
  queueEdit: boolean
  recording: boolean
  triggerOpen: boolean
}) {
  // Same registration order as ChatBar: turn-cancel (bubble) first, then
  // dictation (capture).
  useComposerEscCancel({ awaitingInput: false, busy, onCancel: haltRun, target: 'main' })
  useDictationEscCancel({ onCancel: cancelDictation, recording, target: 'main' })

  // Mirror of handleEditorKeyDown's Escape branches in index.tsx.
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (triggerOpen && event.key === 'Escape') {
      event.preventDefault()
      closeTrigger()

      return
    }

    if (event.key === 'Escape') {
      if (event.defaultPrevented) {
        return
      }

      if (queueEdit) {
        event.preventDefault()
        exitQueuedEdit()

        return
      }

      if (busy) {
        event.preventDefault()
        haltRun()
      }
    }
  }

  return (
    <div>
      <div contentEditable data-testid="editor" onKeyDown={onKeyDown} suppressContentEditableWarning />
      {triggerOpen ? <div data-testid="trigger" role="listbox" /> : null}
    </div>
  )
}

function renderComposer(overrides: Partial<Parameters<typeof Harness>[0]> = {}) {
  const actions = {
    cancelDictation: vi.fn(),
    closeTrigger: vi.fn(),
    exitQueuedEdit: vi.fn(),
    haltRun: vi.fn()
  }

  markActiveComposer('main')
  const view = render(
    <Harness busy={false} queueEdit={false} recording={true} triggerOpen={false} {...actions} {...overrides} />
  )
  const editor = view.getByTestId('editor')
  editor.focus()

  return { ...actions, editor }
}

const pressEscape = (editor: HTMLElement) => fireEvent.keyDown(editor, { bubbles: true, cancelable: true, key: 'Escape' })

describe('composer Escape precedence with a live dictation', () => {
  it('recording + open completion listbox: the listbox keeps Esc, dictation survives', () => {
    const { cancelDictation, closeTrigger, editor, haltRun } = renderComposer({ triggerOpen: true })

    pressEscape(editor)

    expect(closeTrigger).toHaveBeenCalledTimes(1)
    expect(cancelDictation).not.toHaveBeenCalled()
    expect(haltRun).not.toHaveBeenCalled()
  })

  it('recording + queue edit: dictation is discarded, the edit stays open', () => {
    const { cancelDictation, editor, exitQueuedEdit, haltRun } = renderComposer({ queueEdit: true })

    pressEscape(editor)

    expect(cancelDictation).toHaveBeenCalledTimes(1)
    expect(exitQueuedEdit).not.toHaveBeenCalled()
    expect(haltRun).not.toHaveBeenCalled()
  })

  it('recording + busy turn: dictation is discarded, the turn keeps running', () => {
    const { cancelDictation, editor, haltRun } = renderComposer({ busy: true })

    pressEscape(editor)

    expect(cancelDictation).toHaveBeenCalledTimes(1)
    expect(haltRun).not.toHaveBeenCalled()
  })

  it('no recording: local Escape handling is untouched', () => {
    const { cancelDictation, editor, exitQueuedEdit } = renderComposer({ queueEdit: true, recording: false })

    pressEscape(editor)

    expect(exitQueuedEdit).toHaveBeenCalledTimes(1)
    expect(cancelDictation).not.toHaveBeenCalled()
  })
})
