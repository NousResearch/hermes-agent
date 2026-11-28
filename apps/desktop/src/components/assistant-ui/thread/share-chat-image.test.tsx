/* eslint-disable no-restricted-globals -- testing the live DOM selection chrome, bottom bar portal, and SVG stage */
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'

import { stubThreadEnvironment } from '../test-utils'

import {
  collectDocumentCssText,
  collectMessageBlocks,
  copyImageToSystemClipboard,
  dataUrlToBlob,
  downloadDataUrl,
  enterShareChatMode,
  exitShareChatMode,
  findTurnPeers,
  isShareChatModeActive,
  originSessionRef,
  prepareCloneForSnapshot,
  quoteSelectedToNewChat,
  renderSelectedToDataUrl,
  runShareExport,
  SHARE_BAR_ID,
  SHARE_BLOCK_ROLE_ATTR,
  SHARE_CHECK_ATTR,
  SHARE_MODE_ATTR,
  SHARE_SELECTABLE_ATTR,
  SHARE_SELECTED_ATTR,
  toggleBlock,
  toggleSelectAll
} from './share-chat-image'

import { Thread } from '.'

const requestFreshSession = vi.hoisted(() => vi.fn())
const requestComposerAttachImages = vi.hoisted(() => vi.fn())
const requestComposerSetDraft = vi.hoisted(() => vi.fn<(...args: unknown[]) => Promise<boolean>>(async () => true))
const requestComposerInsertAcked = vi.hoisted(() => vi.fn<(...args: unknown[]) => Promise<boolean>>(async () => true))
const requestComposerFocus = vi.hoisted(() => vi.fn())

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestFreshSession: () => requestFreshSession()
}))

vi.mock('@/app/chat/composer/focus', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestComposerAttachImages: (...args: unknown[]) => requestComposerAttachImages(...args),
  requestComposerSetDraft: (...args: unknown[]) => requestComposerSetDraft(...args),
  requestComposerInsertAcked: (...args: unknown[]) => requestComposerInsertAcked(...args),
  requestComposerFocus: (...args: unknown[]) => requestComposerFocus(...args)
}))

stubThreadEnvironment()

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function makeUserMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: { timelineTimestamp: createdAt.getTime() / 1000 } }
  } as unknown as ThreadMessage
}

function makeAssistantMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: { timelineTimestamp: createdAt.getTime() / 1000 }
    }
  } as unknown as ThreadMessage
}

function MultiTurnHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [
      makeUserMessage('u-1', 'first question'),
      makeAssistantMessage('a-1', 'first answer'),
      makeUserMessage('u-2', 'second question'),
      makeAssistantMessage('a-2', 'second answer')
    ],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

beforeEach(() => {
  exitShareChatMode()
  clearNotifications()
  requestFreshSession.mockClear()
  requestComposerAttachImages.mockClear()
  requestComposerSetDraft.mockReset().mockResolvedValue(true)
  requestComposerInsertAcked.mockReset().mockResolvedValue(true)
  requestComposerFocus.mockClear()
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $activeGatewayProfile.set('default')
  $sessions.set([])
})

afterEach(() => {
  exitShareChatMode()
  cleanup()
})

describe('share-chat-image core action bar & selection mode', () => {
  it('renders the Share conversation as image button in AssistantActionBar and enters selection mode for the clicked turn', async () => {
    render(<MultiTurnHarness />)

    await screen.findByText('second answer')
    const shareButtons = document.querySelectorAll<HTMLButtonElement>('button[data-share-chat-btn="true"]')
    expect(shareButtons.length).toBe(2)

    // Click the second turn's share button
    fireEvent.click(shareButtons[1]!)

    expect(isShareChatModeActive()).toBe(true)

    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    expect(viewport.getAttribute(SHARE_MODE_ATTR)).toBe('true')

    const blocks = collectMessageBlocks(viewport)
    expect(blocks.length).toBe(4)

    // Only the second turn (u-2 + a-2) should be pre-selected
    expect(blocks[0]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('false')
    expect(blocks[1]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('false')
    expect(blocks[2]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('true')
    expect(blocks[3]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('true')

    const bar = document.getElementById(SHARE_BAR_ID)
    expect(bar).toBeTruthy()
    expect(bar?.querySelector('[data-role="selected-count"]')?.textContent).toMatch(/2/)
  })

  it('toggles individual blocks, Select All switch, and exits cleanly on Escape or Cancel', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('second answer')

    enterShareChatMode(null)
    expect(isShareChatModeActive()).toBe(true)

    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const blocks = collectMessageBlocks(viewport)
    expect(blocks.length).toBe(4)

    // Toggle block 0 on
    toggleBlock(blocks[0]!)
    expect(blocks[0]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('true')

    // Click Select All to select all 4 blocks
    toggleSelectAll()

    for (const block of blocks) {
      expect(block.getAttribute(SHARE_SELECTED_ATTR)).toBe('true')
    }

    const bar = document.getElementById(SHARE_BAR_ID)!
    expect(bar.querySelector('[data-role="select-all-switch"]')?.getAttribute('data-checked')).toBe('true')

    // Click Select All again to deselect all
    toggleSelectAll()

    for (const block of blocks) {
      expect(block.getAttribute(SHARE_SELECTED_ATTR)).toBe('false')
    }

    expect(bar.querySelector('[data-role="select-all-switch"]')?.getAttribute('data-checked')).toBe('false')

    // Escape exits share mode and cleans up attributes and bottom bar
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(isShareChatModeActive()).toBe(false)
    expect(viewport.hasAttribute(SHARE_MODE_ATTR)).toBe(false)
    expect(document.getElementById(SHARE_BAR_ID)).toBeNull()

    for (const block of blocks) {
      expect(block.hasAttribute(SHARE_SELECTABLE_ATTR)).toBe(false)
      expect(block.querySelector(`[${SHARE_CHECK_ATTR}]`)).toBeNull()
    }
  })

  it('intercepts clicks on message blocks during share mode to toggle selection without opening edit composer', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('first answer')

    enterShareChatMode(null)
    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const blocks = collectMessageBlocks(viewport)

    expect(blocks[0]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('false')
    fireEvent.click(blocks[0]!)
    expect(blocks[0]?.getAttribute(SHARE_SELECTED_ATTR)).toBe('true')
  })

  it('un-virtualizes content-visibility on message groups during selection mode and restores on exit', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('first answer')

    const group = document.querySelector<HTMLElement>('[data-slot="aui_message-group"]')!
    group.style.contentVisibility = 'auto'

    enterShareChatMode(null)
    expect(group.style.contentVisibility).toBe('visible')

    exitShareChatMode()
    expect(group.style.contentVisibility).toBe('')
  })
})

describe('findTurnPeers', () => {
  it('groups an assistant block with its preceding user prompt', () => {
    const u1 = document.createElement('div')
    u1.setAttribute('data-slot', 'aui_user-message-root')
    const a1 = document.createElement('div')
    a1.setAttribute('data-slot', 'aui_response-group')
    const u2 = document.createElement('div')
    u2.setAttribute('data-slot', 'aui_user-message-root')
    const a2 = document.createElement('div')
    a2.setAttribute('data-slot', 'aui_response-group')

    expect(findTurnPeers(a1, [u1, a1, u2, a2])).toEqual([u1, a1])
    expect(findTurnPeers(null, [u1, a1, u2, a2])).toEqual([u2, a2])
  })
})

describe('collectDocumentCssText & prepareCloneForSnapshot', () => {
  it('aliases :root and :root.dark to [data-share-stage] and strips selection chrome from clones', async () => {
    document.documentElement.style.setProperty('--dt-background', '#181818')
    const css = collectDocumentCssText()
    expect(css).toContain('[data-share-stage]')
    expect(css).toContain('content-visibility: visible !important')

    const source = document.createElement('div')
    source.setAttribute('data-slot', 'aui_user-message-root')
    source.setAttribute(SHARE_SELECTABLE_ATTR, '0')
    source.setAttribute(SHARE_SELECTED_ATTR, 'true')
    source.setAttribute(SHARE_BLOCK_ROLE_ATTR, 'user')
    source.style.contentVisibility = 'auto'
    source.style.clipPath = 'inset(10px)'

    const checkBtn = document.createElement('button')
    checkBtn.setAttribute(SHARE_CHECK_ATTR, '0')
    source.appendChild(checkBtn)

    const actions = document.createElement('div')
    actions.setAttribute('data-slot', 'aui_msg-actions')
    source.appendChild(actions)

    const textSpan = document.createElement('span')
    textSpan.setAttribute('data-slot', 'aui_directive-text')
    textSpan.textContent =
      '[The user sent an image. Use vision_analyze with image_url: C:\\test\\shot.png] Explain this diagram'
    source.appendChild(textSpan)

    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { readFileDataUrl }

    try {
      const cloned = await prepareCloneForSnapshot(source)
      expect(cloned.querySelector(`[${SHARE_CHECK_ATTR}]`)).toBeNull()
      expect(cloned.querySelector('[data-slot="aui_msg-actions"]')).toBeNull()
      // Vision wrapper stripped, real prompt preserved, and thumbnail restored
      expect(cloned.textContent).toBe('Explain this diagram')
      expect(readFileDataUrl).toHaveBeenCalledWith('C:\\test\\shot.png')
      expect(cloned.querySelectorAll('img').length).toBe(1)
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })
})

describe('rasterization, clipboard copy, download, and quote-to-new-chat', () => {
  const sampleDataUrl =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='

  let origSrcDescriptor: PropertyDescriptor | undefined
  let getContextSpy: ReturnType<typeof vi.spyOn>
  let toDataURLSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    origSrcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src')
    Object.defineProperty(HTMLImageElement.prototype, 'src', {
      configurable: true,
      get() {
        return this.getAttribute('src') || ''
      },
      set(val: string) {
        this.setAttribute('src', val)
        queueMicrotask(() => {
          if (typeof this.onload === 'function') {
            this.onload(new Event('load'))
          }
        })
      }
    })

    getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      scale: vi.fn(),
      fillRect: vi.fn(),
      drawImage: vi.fn(),
      fillStyle: ''
    } as unknown as CanvasRenderingContext2D)

    toDataURLSpy = vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(sampleDataUrl)
  })

  afterEach(() => {
    if (origSrcDescriptor) {
      Object.defineProperty(HTMLImageElement.prototype, 'src', origSrcDescriptor)
    }

    getContextSpy.mockRestore()
    toDataURLSpy.mockRestore()
  })

  it('renders selected blocks to a PNG dataUrl via SVG foreignObject', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('second answer')

    enterShareChatMode(null)
    const { dataUrl, canvas } = await renderSelectedToDataUrl()

    expect(dataUrl).toBe(sampleDataUrl)
    expect(canvas).toBeInstanceOf(HTMLCanvasElement)
    // Cleanup removed the temporary stage wrapper
    expect(document.querySelector('[data-share-stage]')).toBeNull()
  })

  it('copies image to clipboard via capture-phase copy listener and execCommand', async () => {
    let capturedHtml = ''
    let capturedText = 'not-empty'

    const origExecCommand = document.execCommand
    document.execCommand = vi.fn((cmd: string) => {
      if (cmd === 'copy') {
        const evt = new Event('copy', { bubbles: true, cancelable: true }) as ClipboardEvent
        Object.defineProperty(evt, 'clipboardData', {
          value: {
            setData: (type: string, val: string) => {
              if (type === 'text/html') {capturedHtml = val}

              if (type === 'text/plain') {capturedText = val}
            }
          }
        })
        window.dispatchEvent(evt)

        return true
      }

      return false
    })

    try {
      const ok = await copyImageToSystemClipboard(sampleDataUrl)
      expect(ok).toBe(true)
      expect(capturedText).toBe('')
      expect(capturedHtml).toContain(`src="${sampleDataUrl}"`)
      expect(document.querySelector('[data-share-copy-holder]')).toBeNull()
    } finally {
      document.execCommand = origExecCommand
    }
  })

  it('saves image via window.hermesDesktop.saveImageFromUrl when available', async () => {
    const saveImageFromUrl = vi.fn(async () => true)

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { saveImageFromUrl }

    try {
      const saved = await downloadDataUrl(sampleDataUrl)
      expect(saved).toBe(true)
      expect(saveImageFromUrl).toHaveBeenCalledWith(sampleDataUrl)
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('quotes selected conversation image and @session:<profile>/<id> into a fresh chat', async () => {
    $selectedStoredSessionId.set('20261031_120000_abcdef')
    $activeGatewayProfile.set('default')

    expect(originSessionRef()).toBe('@session:default/20261031_120000_abcdef')

    const blob = dataUrlToBlob(sampleDataUrl)
    expect(blob).toBeInstanceOf(Blob)
    expect(blob?.type).toBe('image/png')

    render(<MultiTurnHarness />)
    await screen.findByText('second answer')

    enterShareChatMode(null)
    await runShareExport('quote-new')

    expect(isShareChatModeActive()).toBe(false)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(requestComposerAttachImages).toHaveBeenCalledTimes(1)
    expect(requestComposerSetDraft).toHaveBeenCalledWith([], '@session:default/20261031_120000_abcdef\n', {
      active: true
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('main')
    expect($notifications.get()[0]?.kind).toBe('success')
  })

  it('falls back to requestComposerInsertAcked when requestComposerSetDraft returns false', async () => {
    $selectedStoredSessionId.set('sess_999')
    requestComposerSetDraft.mockResolvedValueOnce(false)

    await quoteSelectedToNewChat(sampleDataUrl)

    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(requestComposerInsertAcked).toHaveBeenCalledWith('@session:default/sess_999', {
      mode: 'block',
      target: 'main'
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('main')
  })

  it('caps canvas pixel dimensions below 15,800px when rendering ultra-tall conversations', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('second answer')

    enterShareChatMode(null)

    const scrollHeightSpy = vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(20_000)

    try {
      const { canvas } = await renderSelectedToDataUrl()
      expect(canvas.height).toBeLessThanOrEqual(15_800)
    } finally {
      scrollHeightSpy.mockRestore()
    }
  })

  it('shows an info toast and skips rasterization when runShareExport is triggered with 0 selected blocks', async () => {
    render(<MultiTurnHarness />)
    await screen.findByText('second answer')

    enterShareChatMode(null)
    // Select all then deselect all -> 0 selected
    toggleSelectAll()
    toggleSelectAll()

    await runShareExport('copy')

    expect(isShareChatModeActive()).toBe(true)
    expect($notifications.get()[0]?.kind).toBe('info')
  })

  it('clones sibling attachment containers and converts embedded canvases and SVGs in prepareCloneForSnapshot', async () => {
    const parent = document.createElement('div')
    const userRoot = document.createElement('div')
    userRoot.setAttribute('data-slot', 'aui_user-message-root')

    const canvas = document.createElement('canvas')
    canvas.width = 50
    canvas.height = 30
    userRoot.appendChild(canvas)

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      width: 24,
      height: 24,
      top: 0,
      left: 0,
      right: 24,
      bottom: 24,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })
    userRoot.appendChild(svg)

    const siblingAtt = document.createElement('div')
    const attImg = document.createElement('img')
    attImg.setAttribute('src', sampleDataUrl)
    siblingAtt.appendChild(attImg)

    parent.appendChild(userRoot)
    parent.appendChild(siblingAtt)

    const cloned = await prepareCloneForSnapshot(userRoot)
    // Canvas replaced with <img>, plus the sibling attachment <img>
    expect(cloned.querySelectorAll('canvas').length).toBe(0)
    expect(cloned.querySelectorAll('img').length).toBe(2)
    const clonedSvg = cloned.querySelector('svg')
    expect(clonedSvg?.getAttribute('width')).toBe('24')
    expect(clonedSvg?.getAttribute('height')).toBe('24')
  })
})
