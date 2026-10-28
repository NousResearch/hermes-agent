// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'

let unregister: (() => void) | undefined

afterEach(() => {
  unregister?.()
  unregister = undefined
  document.body.innerHTML = ''
})

describe('session tab keyboard focus ownership', () => {
  it('keeps roving focus in zone tabs but still hands typing focus back on deliberate pointer movement', () => {
    const zone = document.createElement('div')
    zone.dataset.treeGroup = 'chat-zone'
    const list = document.createElement('div')
    list.setAttribute('role', 'tablist')

    const tabs = ['Current session', 'Other session'].map((name, index) => {
      const tab = document.createElement('div')
      tab.setAttribute('role', 'tab')
      tab.textContent = name
      tab.tabIndex = index === 0 ? 0 : -1
      list.appendChild(tab)

      return tab
    })

    const host = document.createElement('div')
    host.dataset.composerOwner = 'chat-composer'
    const editor = document.createElement('div')
    editor.dataset.slot = 'composer-rich-input'
    editor.tabIndex = -1
    host.appendChild(editor)
    const transcript = document.createElement('div')
    transcript.dataset.chatSurface = ''
    transcript.dataset.composerSurfaceId = 'chat-composer'
    zone.append(list, transcript, host)
    document.body.appendChild(zone)
    unregister = registerFloatingComposer('chat-composer', { groupId: 'chat-zone', target: 'main' })

    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }))
    tabs[0]!.focus()
    expect(document.activeElement).toBe(tabs[0])

    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }))
    tabs[1]!.focus()
    expect(document.activeElement).toBe(tabs[1])

    transcript.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: 40, clientY: 40 }))
    expect(document.activeElement).toBe(editor)
  })
})
