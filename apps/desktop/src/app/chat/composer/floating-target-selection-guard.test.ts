import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'

function mount() {
  const transcript = document.createElement('div')
  transcript.id = 'transcript'
  transcript.textContent = 'some transcript text to select'
  document.body.appendChild(transcript)

  const host = document.createElement('div')
  host.dataset.composerOwner = 'surface-1'

  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  return { transcript, editor }
}

function selectRange(el: HTMLElement) {
  const selection = window.getSelection()!
  const text = el.firstChild!
  const range = document.createRange()
  range.setStart(text, 0)
  range.setEnd(text, text.textContent!.length - 1)
  selection.removeAllRanges()
  selection.addRange(range)
}

function movePointerOver(target: Element) {
  target.dispatchEvent(
    new PointerEvent('pointermove', {
      bubbles: true,
      cancelable: true,
      clientX: 43,
      clientY: 44,
    })
  )
}

describe('floating-target pointer selection guard', () => {
  let unregister: (() => void) | undefined

  afterEach(() => {
    unregister?.()
    unregister = undefined
    document.body.innerHTML = ''
    window.getSelection()?.removeAllRanges()
  })

  it('preserves a non-collapsed transcript selection on pointermove over a composer surface', () => {
    const { transcript, editor } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    selectRange(transcript)
    const before = window.getSelection()!
    expect(before.rangeCount).toBe(1)
    expect(before.anchorNode).toBe(transcript.firstChild)

    movePointerOver(editor)

    const after = window.getSelection()!
    expect(after.rangeCount).toBe(1)
    expect(after.anchorNode).toBe(transcript.firstChild)
  })

  it('still focuses the composer when nothing is selected outside it', () => {
    const { editor } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    expect(window.getSelection()!.rangeCount).toBe(0)
    movePointerOver(editor)

    // Focus ran: the composer editor became the active element.
    expect(document.activeElement).toBe(editor)
  })
})
