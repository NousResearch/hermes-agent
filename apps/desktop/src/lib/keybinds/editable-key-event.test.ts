import { afterEach, describe, expect, it } from 'vitest'

import { isEditableKeyEvent } from './combo'

function captureSpace(target: EventTarget): KeyboardEvent {
  let seen: KeyboardEvent | null = null

  const listener = (event: KeyboardEvent) => {
    seen = event
  }

  window.addEventListener('keydown', listener, { capture: true })
  target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, code: 'Space', key: ' ' }))
  window.removeEventListener('keydown', listener, { capture: true })

  if (!seen) {
    throw new Error('keydown was not observed')
  }

  return seen
}

describe('isEditableKeyEvent', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('claims Space typed into an input', () => {
    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    expect(isEditableKeyEvent(captureSpace(input))).toBe(true)
  })

  it('claims Space when a portal element is the target but an input holds focus', () => {
    const input = document.createElement('textarea')
    const portal = document.createElement('div')
    document.body.append(input, portal)
    input.focus()

    expect(isEditableKeyEvent(captureSpace(portal))).toBe(true)
  })

  it('leaves Space over non-editable chrome to push-to-talk', () => {
    const button = document.createElement('button')
    document.body.append(button)
    button.focus()

    expect(isEditableKeyEvent(captureSpace(button))).toBe(false)
  })
})
