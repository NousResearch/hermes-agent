import { afterEach, describe, expect, it, vi } from 'vitest'

import { RICH_INPUT_SLOT } from './rich-editor'
import { performEditHistory } from './perform-edit-history'

describe('performEditHistory', () => {
  afterEach(() => {
    document.body.replaceChildren()
    vi.restoreAllMocks()
  })

  it('dispatches historyUndo on the focused rich composer', () => {
    const editor = document.createElement('div')
    editor.contentEditable = 'true'
    editor.tabIndex = 0
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    editor.focus()

    const seen: string[] = []
    editor.addEventListener('beforeinput', event => {
      seen.push((event as InputEvent).inputType)
      event.preventDefault()
    })

    expect(performEditHistory('undo')).toBe(true)
    expect(seen).toEqual(['historyUndo'])
  })

  it('falls back to execCommand for ordinary fields', () => {
    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    const exec = vi.fn(() => true)
    Object.defineProperty(document, 'execCommand', { configurable: true, value: exec, writable: true })

    expect(performEditHistory('redo')).toBe(true)
    expect(exec).toHaveBeenCalledWith('redo')
  })
})
