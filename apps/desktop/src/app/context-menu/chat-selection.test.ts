import { afterEach, describe, expect, it } from 'vitest'

import { captureChatSelection, chatSelectionIsCurrent } from './chat-selection'

function select(from: Node, to = from) {
  const range = document.createRange()
  range.setStart(from, 0)
  range.setEnd(to, to.textContent!.length)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.innerHTML = ''
})

describe('chat selection ownership', () => {
  it('binds selection to one rendered chat and rejects later text, session, or DOM changes', () => {
    document.body.innerHTML =
      '<div data-selection-session-id="chat-a"><div data-slot="aui_assistant-message-root">words</div></div>'
    const message = document.querySelector('[data-slot]')!
    select(message.firstChild!)
    const captured = captureChatSelection(message)!
    expect(captured.sessionId).toBe('chat-a')
    expect(chatSelectionIsCurrent(captured)).toBe(true)
    message.parentElement!.setAttribute('data-selection-session-id', 'chat-b')
    expect(chatSelectionIsCurrent(captured)).toBe(false)
    message.parentElement!.setAttribute('data-selection-session-id', 'chat-a')
    window.getSelection()!.removeAllRanges()
    expect(chatSelectionIsCurrent(captured)).toBe(false)
    select(message.firstChild!)
    message.remove()
    expect(chatSelectionIsCurrent(captured)).toBe(false)
  })

  it('rejects cross-message, editable, non-chat and unbound selections', () => {
    document.body.innerHTML =
      '<div data-selection-session-id="chat-a"><div data-slot="aui_user-message-root">one</div><div data-slot="aui_assistant-message-root">two</div><p>outside</p></div>'
    const [a, b] = document.querySelectorAll('[data-slot]')
    select(a.firstChild!, b.firstChild!)
    expect(captureChatSelection(a)).toBeNull()
    select(a.firstChild!)
    a.setAttribute('contenteditable', 'true')
    expect(captureChatSelection(a)).toBeNull()
    a.removeAttribute('contenteditable')
    a.parentElement!.removeAttribute('data-selection-session-id')
    expect(captureChatSelection(a)).toBeNull()
    const outside = document.querySelector('p')!
    select(outside.firstChild!)
    expect(captureChatSelection(outside)).toBeNull()
  })
})
