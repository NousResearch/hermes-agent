import { afterEach, describe, expect, it, vi } from 'vitest'

// The find bar is a plain overlay input, not a dialog/terminal, so the
// composer's surface guard did not know about it. Typing into Ctrl+F therefore
// raced the composer's focus bus, which refocuses three times (sync, rAF,
// setTimeout) and always won — the find input lost the caret and swallowed
// every keystroke.
import { findBarOwnsTyping } from './find-bar-focus'

function mountFindBar(): HTMLInputElement {
  const bar = document.createElement('div')

  bar.setAttribute('data-slot', 'find-bar')
  const input = document.createElement('input')

  bar.appendChild(input)
  document.body.appendChild(bar)

  return input
}

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('findBarOwnsTyping', () => {
  it('is false when the bar is closed', () => {
    expect(findBarOwnsTyping(false)).toBe(false)
  })

  it('is true whenever the bar is open', () => {
    mountFindBar()

    expect(findBarOwnsTyping(true)).toBe(true)
  })

  it('is true even before the input has taken focus', () => {
    // The regression window: the bar has rendered but its rAF focus has not
    // run yet. A keystroke landing here must still not reach the composer.
    mountFindBar()
    expect(document.activeElement).not.toBeInstanceOf(HTMLInputElement)

    expect(findBarOwnsTyping(true)).toBe(true)
  })

  it('is true while the find input holds focus', () => {
    const input = mountFindBar()

    input.focus()

    expect(findBarOwnsTyping(true)).toBe(true)
  })

  it('does not depend on the bar being in the DOM yet', () => {
    // Store flips active a commit before the element mounts; the guard must
    // hold from the store alone or the first keystroke still escapes.
    expect(findBarOwnsTyping(true)).toBe(true)
  })
})
