import { afterEach, describe, expect, it } from 'vitest'

import { hasTextSelection } from './user-message'

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
})

describe('hasTextSelection', () => {
  it('is false with nothing highlighted', () => {
    expect(hasTextSelection()).toBe(false)
  })

  it('is true once the user has a live range', () => {
    const node = document.createElement('span')
    node.textContent = 'copy me'
    document.body.appendChild(node)

    const range = document.createRange()
    range.selectNodeContents(node)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    expect(hasTextSelection()).toBe(true)
  })
})
