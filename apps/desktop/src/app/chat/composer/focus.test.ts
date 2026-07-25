import { afterEach, describe, expect, it } from 'vitest'

import { blurComposerInput } from './focus'
import { RICH_INPUT_SLOT } from './rich-editor'

/**
 * Inactive tabs keep their composer mounted, so an unscoped lookup can blur a
 * background input and leave the one the user is typing in focused.
 */

/** A composer input inside its own pane layer, hidden or not. */
function mountInput(hidden = false) {
  const layer = document.createElement('div')
  const input = document.createElement('div')
  input.dataset.slot = RICH_INPUT_SLOT
  input.tabIndex = 0
  layer.toggleAttribute('data-pane-hidden', hidden)
  layer.append(input)
  document.body.append(layer)

  return input
}

afterEach(() => {
  document.body.innerHTML = ''
})

describe('blurComposerInput', () => {
  it('blurs the foreground composer while a hidden tab matches first', () => {
    const background = mountInput(true)
    const foreground = mountInput()

    foreground.focus()
    blurComposerInput()

    expect(document.activeElement).not.toBe(foreground)
    expect(document.activeElement).not.toBe(background)
  })

  it('leaves focus alone when the composer does not hold it', () => {
    const outside = document.createElement('button')
    document.body.append(outside)
    mountInput()

    outside.focus()
    blurComposerInput()

    expect(document.activeElement).toBe(outside)
  })
})
