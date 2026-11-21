// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'
import { getActiveComposer } from './focus'

/** The owner's composer host, a chat surface with a plain control, and a
 * body-portaled Radix popper layer holding a control — the shapes the
 * window-level focus-follow has to tell apart. */
function mount(id = 'surface-1') {
  const host = document.createElement('div')
  host.dataset.composerOwner = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  const surface = document.createElement('div')
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = id
  const plainButton = document.createElement('button')
  surface.appendChild(plainButton)
  document.body.appendChild(surface)

  const layer = document.createElement('div')
  layer.setAttribute('data-radix-popper-content-wrapper', '')
  const layerButton = document.createElement('button')
  layer.appendChild(layerButton)
  document.body.appendChild(layer)

  return { editor, layer, layerButton, plainButton, surface }
}

/** Button-up movement over the chat surface — every hover of the transcript. */
function movePointerOver(target: Element, x: number) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: x, clientY: 40 }))
}

/** Radix moves focus into a popover/menu when it opens and dismisses it as soon
 * as focus leaves. The focus-follow used to pull that focus back into the pane
 * composer on the next pointermove, so the message reaction picker closed
 * before the pointer could reach it. */
describe('floating composer focus-follow vs an open floating layer', () => {
  let unregister: (() => void) | undefined

  afterEach(() => {
    unregister?.()
    unregister = undefined
    document.body.innerHTML = ''
  })

  it('leaves focus inside a floating layer, but still follows the pointer otherwise', () => {
    const { editor, layerButton, plainButton, surface } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    layerButton.focus()
    movePointerOver(surface, 10)
    expect(document.activeElement).toBe(layerButton)

    // Control: focus on an ordinary control in the surface is not "owned" —
    // hovering the transcript hands the caret to the composer as before.
    plainButton.focus()
    movePointerOver(surface, 20)
    expect(document.activeElement).toBe(editor)
  })

  it.each(['connected', 'detached'] as const)('preserves return focus from a %s layer and selects its chat', state => {
    const first = mount()
    const second = mount('surface-2')
    const disposeFirst = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })
    const disposeSecond = registerFloatingComposer('surface-2', { groupId: 'g2', target: 'tile:second' })

    unregister = () => {
      disposeFirst()
      disposeSecond()
    }

    first.editor.focus()
    second.layerButton.focus()
    second.layerButton.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' }))

    if (state === 'detached') {
      second.layer.remove()
    }

    second.plainButton.focus()
    expect(document.activeElement).toBe(second.plainButton)
    expect(getActiveComposer()).toBe('tile:second')

    // Return-focus protection is consumed, not a persistent redirect bypass.
    second.editor.focus()
    first.plainButton.focus()
    expect(document.activeElement).toBe(second.editor)
  })
})
