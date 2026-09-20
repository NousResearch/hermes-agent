// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import {
  $ideLayout,
  IDE_BROWSER_MIN_HEIGHT,
  IDE_CHAT_MAX_WIDTH,
  IDE_EXPLORER_MAX_WIDTH,
  IDE_LAYOUT_DEFAULTS,
  IDE_LAYOUT_STORAGE_KEY,
  setIdeBrowserHeight,
  setIdeChatWidth,
  setIdeExplorerWidth,
  toggleIdeBrowser,
  toggleIdeChat,
  toggleIdeExplorer
} from './ide-layout'

beforeEach(() => {
  window.localStorage.clear()
  $ideLayout.set(IDE_LAYOUT_DEFAULTS)
})

describe('ide layout store', () => {
  it('clamps sizes to their ranges', () => {
    setIdeExplorerWidth(9999)
    expect($ideLayout.get().explorerWidth).toBe(IDE_EXPLORER_MAX_WIDTH)

    setIdeBrowserHeight(0)
    expect($ideLayout.get().browserHeight).toBe(IDE_BROWSER_MIN_HEIGHT)

    setIdeChatWidth(9999)
    expect($ideLayout.get().chatWidth).toBe(IDE_CHAT_MAX_WIDTH)
  })

  it('ignores non-finite sizes instead of persisting NaN', () => {
    setIdeExplorerWidth(Number.NaN)

    expect(Number.isFinite($ideLayout.get().explorerWidth)).toBe(true)
  })

  it('persists changes under the IDE-owned key', () => {
    setIdeExplorerWidth(300)

    expect(window.localStorage.getItem(IDE_LAYOUT_STORAGE_KEY)).toContain('"explorerWidth":300')
  })

  it('toggles region visibility independently', () => {
    expect($ideLayout.get().explorerOpen).toBe(true)

    toggleIdeExplorer()
    expect($ideLayout.get().explorerOpen).toBe(false)
    expect($ideLayout.get().chatOpen).toBe(true)

    toggleIdeChat()
    expect($ideLayout.get().chatOpen).toBe(false)

    toggleIdeBrowser()
    expect($ideLayout.get().browserOpen).toBe(false)
    expect($ideLayout.get().explorerOpen).toBe(false)
  })
})
