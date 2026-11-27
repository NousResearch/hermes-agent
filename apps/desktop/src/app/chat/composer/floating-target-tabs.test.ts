// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'
import { getActiveComposer } from './focus'

const unregister: Array<() => void> = []

afterEach(() => {
  unregister.splice(0).forEach(dispose => dispose())
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
    unregister.push(registerFloatingComposer('chat-composer', { groupId: 'chat-zone', target: 'main' }))

    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }))
    tabs[0]!.focus()
    expect(document.activeElement).toBe(tabs[0])

    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }))
    tabs[1]!.focus()
    expect(document.activeElement).toBe(tabs[1])

    transcript.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: 40, clientY: 40 }))
    expect(document.activeElement).toBe(editor)
  })

  it('moves the typing recipient to the focused split without taking focus from its tabs', () => {
    const mountZone = (id: string, target: string) => {
      const zone = document.createElement('div')
      zone.dataset.treeGroup = id
      const tab = document.createElement('div')
      tab.setAttribute('role', 'tab')
      tab.tabIndex = 0
      const host = document.createElement('div')
      host.dataset.composerOwner = id
      const editor = document.createElement('div')
      editor.dataset.slot = 'composer-rich-input'
      editor.tabIndex = 0
      host.append(editor)
      zone.append(tab, host)
      document.body.append(zone)
      unregister.push(registerFloatingComposer(id, { groupId: id, target }))

      return { editor, tab }
    }

    const first = mountZone('first', 'main')
    const second = mountZone('second', 'tile:second')
    first.editor.focus()

    first.editor.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }))
    second.tab.focus()
    expect(document.activeElement).toBe(second.tab)
    expect(getActiveComposer()).toBe('tile:second')

    second.tab.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab', shiftKey: true }))
    first.tab.focus()
    expect(document.activeElement).toBe(first.tab)
    expect(getActiveComposer()).toBe('main')
  })
})
