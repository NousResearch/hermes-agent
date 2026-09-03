import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer } from '../focus'

import { useComposerEscCancel } from './use-composer-esc-cancel'
import { useDictationEscCancel } from './use-dictation-esc-cancel'

// Esc is claimed by two composer handlers. DESIGN.md: "One cancel gesture does
// one thing: cancel the active interaction, or close the topmost dismissable
// surface — never both." These tests pin that contract.

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

function pressEscape() {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Escape' })
  window.dispatchEvent(event)

  return event
}

afterEach(() => {
  document.body.innerHTML = ''
})

describe('useDictationEscCancel', () => {
  it('discards the recording on Esc', () => {
    markActiveComposer('main')
    const onCancel = vi.fn()
    renderHook(() => useDictationEscCancel({ onCancel, recording: true, target: 'main' }))

    const event = pressEscape()

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(event.defaultPrevented).toBe(true)
  })

  it('ignores Esc when not recording', () => {
    markActiveComposer('main')
    const onCancel = vi.fn()
    renderHook(() => useDictationEscCancel({ onCancel, recording: false, target: 'main' }))

    pressEscape()

    expect(onCancel).not.toHaveBeenCalled()
  })

  it('ignores Esc for a composer that is not the active one', () => {
    markActiveComposer('main')
    const onCancel = vi.fn()
    renderHook(() => useDictationEscCancel({ onCancel, recording: true, target: 'popout' }))

    pressEscape()

    expect(onCancel).not.toHaveBeenCalled()
  })

  // The edit composer ('edit') is a real active target that has no dictation of
  // its own — no recorder, no VoiceActivity panel — and it already claims Esc to
  // close its trigger popover / cancel the edit. Bailing here is the correct
  // routing decision, not a silent no-op: there is no recording to discard, and
  // Esc must stay with the surface the user is actually in.
  //
  // The edit-composer root must exist in the DOM for the claim to hold: the
  // focus bus heals a claim whose surface is gone back to the visible chat
  // composer, so marking 'edit' without rendering it would resolve to 'main'.
  it('leaves Esc to the edit composer when it owns focus', () => {
    const editRoot = document.createElement('div')
    editRoot.setAttribute('data-slot', 'aui_edit-composer-root')
    document.body.appendChild(editRoot)
    markActiveComposer('edit')

    const onCancel = vi.fn()
    renderHook(() => useDictationEscCancel({ onCancel, recording: true, target: 'main' }))

    const event = pressEscape()

    expect(onCancel).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false)
  })

  it('lets an open dialog keep Esc', () => {
    markActiveComposer('main')
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.appendChild(dialog)

    const onCancel = vi.fn()
    renderHook(() => useDictationEscCancel({ onCancel, recording: true, target: 'main' }))

    pressEscape()

    expect(onCancel).not.toHaveBeenCalled()
  })
})

describe('Esc precedence between dictation and the running turn', () => {
  it('cancels the dictation and NOT the agent turn when both are live', () => {
    markActiveComposer('main')
    const haltRun = vi.fn()
    const cancelDictation = vi.fn()

    renderHook(() =>
      useComposerEscCancel({ awaitingInput: false, busy: true, onCancel: haltRun, target: 'main' })
    )
    renderHook(() => useDictationEscCancel({ onCancel: cancelDictation, recording: true, target: 'main' }))

    pressEscape()

    expect(cancelDictation).toHaveBeenCalledTimes(1)
    // The capture-phase preventDefault must stop the turn from being halted too.
    expect(haltRun).not.toHaveBeenCalled()
  })

  it('still halts the turn when no dictation is recording', () => {
    markActiveComposer('main')
    const haltRun = vi.fn()
    const cancelDictation = vi.fn()

    renderHook(() =>
      useComposerEscCancel({ awaitingInput: false, busy: true, onCancel: haltRun, target: 'main' })
    )
    renderHook(() => useDictationEscCancel({ onCancel: cancelDictation, recording: false, target: 'main' }))

    pressEscape()

    expect(cancelDictation).not.toHaveBeenCalled()
    expect(haltRun).toHaveBeenCalledTimes(1)
  })
})
