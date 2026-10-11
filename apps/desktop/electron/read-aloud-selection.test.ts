// @vitest-environment jsdom
import type { BrowserWindow, WebFrameMain } from 'electron'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { lookupChatSelection, registerSelectionMenuIpc } from './selection-context-menu'

function fixture() {
  document.body.innerHTML =
    '<div data-selection-session-id="chat-a"><div data-slot="aui_assistant-message-root">selected words</div></div>'
  const node = document.querySelector('[data-slot]')!.firstChild!
  const range = document.createRange()
  range.selectNodeContents(node)
  window.getSelection()!.removeAllRanges()
  window.getSelection()!.addRange(range)
  const frame = { isDestroyed: () => false, executeJavaScript: vi.fn(async (script: string) => window.eval(script)) }
  const showDefinitionForSelection = vi.fn()
  const win = { isDestroyed: () => false, webContents: { mainFrame: frame, showDefinitionForSelection } }

  return { frame: frame as unknown as WebFrameMain, win: win as unknown as BrowserWindow, showDefinitionForSelection }
}

afterEach(() => {
  document.body.innerHTML = ''
  window.getSelection()?.removeAllRanges()
})

describe('sender-bound dictionary selection', () => {
  it('uses only a matching live selection in the sending main frame and source chat', async () => {
    const f = fixture()
    expect(await lookupChatSelection(f.win, f.frame, { text: 'selected words', sessionId: 'chat-a' }, true)).toBe(true)
    expect(f.showDefinitionForSelection).toHaveBeenCalledOnce()
    expect(await lookupChatSelection(f.win, f.frame, { text: 'stale words', sessionId: 'chat-a' }, true)).toBe(false)
    expect(await lookupChatSelection(f.win, f.frame, { text: 'selected words', sessionId: 'chat-b' }, true)).toBe(false)
    expect(f.showDefinitionForSelection).toHaveBeenCalledOnce()
  })

  it('rejects child frames, editables, non-Mac hosts, and oversized payloads', async () => {
    const f = fixture()
    const payload = { text: 'selected words', sessionId: 'chat-a' }
    expect(await lookupChatSelection(f.win, {} as WebFrameMain, payload, true)).toBe(false)
    expect(await lookupChatSelection(f.win, f.frame, payload, false)).toBe(false)
    expect(await lookupChatSelection(f.win, f.frame, { ...payload, text: 'x'.repeat(16_001) }, true)).toBe(false)
    document.querySelector('[data-slot]')!.setAttribute('contenteditable', 'true')
    expect(await lookupChatSelection(f.win, f.frame, payload, true)).toBe(false)
    expect(f.showDefinitionForSelection).not.toHaveBeenCalled()
  })

  it('keeps the existing edit-command whitelist when registering the new dictionary capability', () => {
    const handlers = new Map<string, Function>()
    registerSelectionMenuIpc(
      {
        handle: (name: string, handler: Function) => {
          handlers.set(name, handler)
        }
      } as never,
      () => null,
      true
    )
    const sender = { copy: vi.fn(), cut: vi.fn(), paste: vi.fn(), selectAll: vi.fn() }
    handlers.get('hermes:context-menu:edit')!({ sender }, 'copy')
    handlers.get('hermes:context-menu:edit')!({ sender }, 'constructor')
    expect(sender.copy).toHaveBeenCalledOnce()
    expect(sender.cut).not.toHaveBeenCalled()
  })

  it('discards an older dictionary request after a newer selection action wins', async () => {
    const f = fixture()
    let finish!: (value: boolean) => void
    vi.mocked(f.frame.executeJavaScript).mockReturnValueOnce(
      new Promise(resolve => {
        finish = resolve
      })
    )
    const payload = { text: 'selected words', sessionId: 'chat-a' }
    const older = lookupChatSelection(f.win, f.frame, payload, true)
    expect(await lookupChatSelection(f.win, f.frame, payload, true)).toBe(true)
    finish(true)
    expect(await older).toBe(false)
    expect(f.showDefinitionForSelection).toHaveBeenCalledOnce()
  })
})
