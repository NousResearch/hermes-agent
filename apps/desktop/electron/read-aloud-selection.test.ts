import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createIpcSelectionSubscription } from './preload-selection'
import {
  buildSelectionActionItems,
  createChatSelectionAuthorizer,
  probeChatMessageSelection,
  shouldOfferSelectionActions
} from './selection-context-menu'

function makeDomSelection(text: string, sameRoot = true) {
  const anchorRoot = {}
  const focusRoot = sameRoot ? anchorRoot : {}
  const anchorNode = { nodeType: 1, closest: () => anchorRoot }
  const focusNode = { nodeType: 1, closest: () => focusRoot }

  return {
    anchorNode,
    focusNode,
    isCollapsed: false,
    rangeCount: 1,
    toString: () => text
  }
}

describe('selected-text native actions', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('authorizes only the exact live selection inside one rendered message and captures its locale', async () => {
    const previousDocument = globalThis.document
    const previousGetSelection = globalThis.getSelection
    const previousNode = globalThis.Node
    let selection = makeDomSelection('selected words only')

    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: { documentElement: { lang: 'ar' } }
    })
    Object.defineProperty(globalThis, 'Node', {
      configurable: true,
      value: { ELEMENT_NODE: 1 }
    })
    Object.defineProperty(globalThis, 'getSelection', {
      configurable: true,
      value: () => selection
    })

    const frame = {
      executeJavaScript: vi.fn(async (script: string) => (0, eval)(script)),
      isDestroyed: () => false
    }

    try {
      await expect(probeChatMessageSelection(frame, 'selected words only')).resolves.toEqual({
        authorized: true,
        locale: 'ar'
      })
      await expect(probeChatMessageSelection(frame, 'older settings text')).resolves.toEqual({
        authorized: false,
        locale: 'en'
      })

      selection = makeDomSelection('selected words only', false)
      await expect(probeChatMessageSelection(frame, 'selected words only')).resolves.toEqual({
        authorized: false,
        locale: 'en'
      })
    } finally {
      Object.defineProperty(globalThis, 'document', { configurable: true, value: previousDocument })
      Object.defineProperty(globalThis, 'Node', { configurable: true, value: previousNode })
      Object.defineProperty(globalThis, 'getSelection', { configurable: true, value: previousGetSelection })
    }
  })

  it('denies malformed or thrown probe output and falls back unknown locales to English', async () => {
    const executeJavaScript = vi
      .fn()
      .mockResolvedValueOnce({ authorized: 'true', locale: 'ar' })
      .mockResolvedValueOnce({ authorized: true, locale: '__proto__' })
      .mockRejectedValueOnce(new Error('renderer unavailable'))

    const frame = { executeJavaScript, isDestroyed: () => false }

    await expect(probeChatMessageSelection(frame, 'selected words only')).resolves.toEqual({
      authorized: false,
      locale: 'en'
    })
    await expect(probeChatMessageSelection(frame, 'selected words only')).resolves.toEqual({
      authorized: true,
      locale: 'en'
    })
    await expect(probeChatMessageSelection(frame, 'selected words only')).resolves.toEqual({
      authorized: false,
      locale: 'en'
    })
  })

  it('rejects an older same-frame authorization after a newer menu request begins', async () => {
    let resolveFirst!: (result: { authorized: boolean; locale: 'ar' }) => void
    let resolveSecond!: (result: { authorized: boolean; locale: 'ja' }) => void

    const probe = vi
      .fn()
      .mockImplementationOnce(
        () => new Promise<{ authorized: boolean; locale: 'ar' }>(resolve => (resolveFirst = resolve))
      )
      .mockImplementationOnce(
        () => new Promise<{ authorized: boolean; locale: 'ja' }>(resolve => (resolveSecond = resolve))
      )

    const frame = { executeJavaScript: vi.fn(), isDestroyed: () => false }

    const window = {
      isDestroyed: () => false,
      webContents: { isDestroyed: () => false, mainFrame: frame }
    }

    const authorize = createChatSelectionAuthorizer(window, probe)

    const older = authorize(frame, 'older settings text')
    const newer = authorize(frame, 'selected chat text')

    resolveSecond({ authorized: true, locale: 'ja' })
    await expect(newer).resolves.toEqual({ authorized: true, current: true, locale: 'ja' })
    resolveFirst({ authorized: true, locale: 'ar' })
    await expect(older).resolves.toEqual({ authorized: false, current: false, locale: 'en' })
    expect(probe).toHaveBeenNthCalledWith(1, frame, 'older settings text')
    expect(probe).toHaveBeenNthCalledWith(2, frame, 'selected chat text')
  })

  it('rejects a subframe selection even when a stale main-frame selection would authorize', async () => {
    const probe = vi.fn().mockResolvedValue({ authorized: true, locale: 'ar' })
    const mainFrame = { executeJavaScript: vi.fn(), isDestroyed: () => false }
    const subframe = { executeJavaScript: vi.fn(), isDestroyed: () => false }

    const window = {
      isDestroyed: () => false,
      webContents: { isDestroyed: () => false, mainFrame }
    }

    const authorize = createChatSelectionAuthorizer(window, probe)

    await expect(authorize(subframe, 'iframe selection')).resolves.toEqual({
      authorized: false,
      current: true,
      locale: 'en'
    })
    expect(probe).not.toHaveBeenCalled()
  })

  it('keeps editable selections on the ordinary editing path', () => {
    expect(shouldOfferSelectionActions({ isEditable: true, selectionText: 'credential-like text' }, true)).toBe(false)
    expect(shouldOfferSelectionActions({ isEditable: false, selectionText: 'message text' }, true)).toBe(true)
    expect(shouldOfferSelectionActions({ isEditable: false, selectionText: 'message text' }, false)).toBe(false)
  })

  it('uses exact localized labels and falls back unknown locales to English', () => {
    const window = {
      isDestroyed: () => false,
      webContents: { send: vi.fn(), showDefinitionForSelection: vi.fn() }
    }

    const cases = [
      ['en', ['Read Aloud', 'Look Up', 'Translate…']],
      ['ar', ['قراءة بصوت عال', 'بحث', 'ترجمة…']],
      ['ja', ['読み上げ', '調べる', '翻訳…']],
      ['zh', ['朗读', '查询', '翻译…']],
      ['zh-hant', ['朗讀', '查詢', '翻譯…']]
    ] as const

    for (const [locale, labels] of cases) {
      expect(buildSelectionActionItems(window, 'selected words only', true, locale).map(item => item.label)).toEqual(
        labels
      )
    }

    expect(
      buildSelectionActionItems(window, 'selected words only', true, '__proto__').map(item => item.label)
    ).toEqual(cases[0][1])
  })

  it('routes localized labels only to fixed actions and omits Look Up off macOS', () => {
    const send = vi.fn()
    const showDefinitionForSelection = vi.fn()

    const window = {
      isDestroyed: () => false,
      webContents: { send, showDefinitionForSelection }
    }

    const items = buildSelectionActionItems(window, 'selected words only', true, 'ar')
    const byLabel = new Map(items.map(item => [item.label, item]))

    byLabel.get('قراءة بصوت عال')?.click?.()
    byLabel.get('بحث')?.click?.()
    byLabel.get('ترجمة…')?.click?.()

    expect(send).toHaveBeenNthCalledWith(1, 'hermes:selection-speech:read', 'selected words only')
    expect(send).toHaveBeenNthCalledWith(2, 'hermes:selection-translate:open', 'selected words only')
    expect(showDefinitionForSelection).toHaveBeenCalledOnce()

    send.mockClear()
    showDefinitionForSelection.mockClear()
    const nonMacItems = buildSelectionActionItems(window, 'selected words only', false, 'ja')

    expect(nonMacItems.map(item => item.label)).toEqual(['読み上げ', '翻訳…'])
    nonMacItems.forEach(item => item.click?.())
    expect(send).toHaveBeenNthCalledWith(1, 'hermes:selection-speech:read', 'selected words only')
    expect(send).toHaveBeenNthCalledWith(2, 'hermes:selection-translate:open', 'selected words only')
    expect(showDefinitionForSelection).not.toHaveBeenCalled()
  })

  it('registers removable IPC listeners that forward only the payload text', () => {
    const on = vi.fn()
    const removeListener = vi.fn()
    const subscribe = createIpcSelectionSubscription({ on, removeListener }, 'selection:open')
    const callback = vi.fn()

    const dispose = subscribe(callback)
    const listener = on.mock.calls[0][1]
    listener({}, 'selected words only', 'ignored')

    expect(callback).toHaveBeenCalledWith('selected words only')
    dispose()
    expect(removeListener).toHaveBeenCalledWith('selection:open', listener)
  })
})