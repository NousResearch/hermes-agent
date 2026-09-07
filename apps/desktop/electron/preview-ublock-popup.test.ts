import { describe, expect, it, vi } from 'vitest'

import type { PreviewUblockState } from './preview-ublock'
import {
  createPreviewUblockPopupManager,
  type PreviewUblockPopupBounds,
  type PreviewUblockPopupWindow
} from './preview-ublock-popup'

const bounds: PreviewUblockPopupBounds = { height: 900, width: 1200, x: 40, y: 20 }

function eventTarget() {
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>()

  return {
    emit(event: string, ...args: unknown[]) {
      for (const listener of listeners.get(event) ?? []) {
        listener(...args)
      }
    },
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener])
    },
    once(event: string, listener: (...args: unknown[]) => void) {
      const wrapped = () => {
        listener()
        const current = listeners.get(event) ?? []
        listeners.set(
          event,
          current.filter(candidate => candidate !== wrapped)
        )
      }
      listeners.set(event, [...(listeners.get(event) ?? []), wrapped])
    }
  }
}

function makeParent() {
  const target = eventTarget()
  const value = {
    ...target,
    destroyed: false,
    getBounds: () => bounds,
    isDestroyed: () => value.destroyed,
    webContents: { once: target.once }
  }

  return value
}

function makePopup(load: () => Promise<void> = async () => undefined) {
  const target = eventTarget()
  const popup = {
    close: vi.fn(() => target.emit('closed')),
    focus: vi.fn(),
    isDestroyed: vi.fn(() => popup.destroyed),
    loadURL: vi.fn(load),
    on: target.on,
    setContentSize: vi.fn(),
    setPosition: vi.fn(),
    show: vi.fn(),
    destroyed: false,
    webContents: {
      isDestroyed: vi.fn(() => popup.destroyed),
      on: target.on,
      executeJavaScript: vi.fn(async () => undefined),
      setWindowOpenHandler: vi.fn()
    }
  }

  return { popup: popup as typeof popup & PreviewUblockPopupWindow, target }
}

function readyState(popupUrl = 'chrome-extension://ublock/popup.html'): PreviewUblockState {
  return {
    available: true,
    dashboardUrl: 'chrome-extension://ublock/dashboard.html',
    enabled: true,
    extensionId: 'ublock',
    operation: {
      failure: null,
      operationId: 'operation',
      phase: 'ready' as const,
      receivedBytes: 0,
      totalBytes: null
    },
    popupUrl,
    rulesetsReady: true,
    version: '2026.825.1619'
  }
}

function controller(initial: PreviewUblockState = readyState()) {
  let state = initial
  let listener: ((next: PreviewUblockState) => void) | null = null

  return {
    controller: {
      getState: () => state,
      subscribe: (next: (state: PreviewUblockState) => void) => {
        listener = next
        next(state)
        return () => {
          listener = null
        }
      }
    },
    setState(next: PreviewUblockState) {
      state = next
      listener?.(next)
    }
  }
}

describe('preview uBlock popup manager', () => {
  it('opens only the validated extension popup URL and reuses one live popup per parent', async () => {
    const owner = controller()
    const created = makePopup()
    const createWindow = vi.fn().mockReturnValue(created.popup)
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow,
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: { name: 'preview' }
    })
    const invokingWindow = makeParent()

    await manager.open(invokingWindow)
    await manager.open(invokingWindow)

    expect(created.popup.loadURL).toHaveBeenCalledWith('chrome-extension://ublock/popup.html')
    expect(createWindow).toHaveBeenCalledOnce()
    expect(created.popup.show).toHaveBeenCalledOnce()
    expect(created.popup.focus).toHaveBeenCalledTimes(2)
  })

  it('measures the loaded document, adds a small allowance, and keeps the popup on-screen', async () => {
    const owner = controller()
    const created = makePopup()
    created.popup.webContents.executeJavaScript = vi.fn(async () => ({ height: 304, valid: true, width: 362 }))
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: () => created.popup,
      getWorkArea: () => ({ height: 500, width: 700, x: 10, y: 20 }),
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: {}
    })

    await manager.open(makeParent())

    expect(created.popup.webContents.executeJavaScript).toHaveBeenCalledOnce()
    expect(created.popup.setContentSize).toHaveBeenCalledWith(378, 320)
    expect(created.popup.setPosition).toHaveBeenCalledWith(316, 68)
  })

  it('closes on blur and can be reopened as a fresh popup', async () => {
    const owner = controller()
    const first = makePopup()
    const second = makePopup()
    const createWindow = vi.fn().mockReturnValueOnce(first.popup).mockReturnValueOnce(second.popup)
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow,
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: {}
    })
    const invokingWindow = makeParent()

    await manager.open(invokingWindow)
    first.target.emit('blur')
    expect(first.popup.close).toHaveBeenCalledOnce()

    await manager.open(invokingWindow)
    expect(createWindow).toHaveBeenCalledTimes(2)
    expect(second.popup.show).toHaveBeenCalledOnce()
    manager.dispose()
  })

  it('rejects arbitrary popup URLs and does not create a window', async () => {
    const owner = controller(readyState('https://example.com/evil.html'))
    const createWindow = vi.fn()
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow,
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: {}
    })

    await expect(manager.open(makeParent())).rejects.toThrow('could not be opened')
    expect(createWindow).not.toHaveBeenCalled()
  })

  it('closes on parent destruction and on controller disable without changing controller state', async () => {
    const owner = controller()
    const created = makePopup()
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: () => created.popup,
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: {}
    })
    const invokingWindow = makeParent()

    await manager.open(invokingWindow)
    invokingWindow.emit('closed')
    expect(created.popup.close).toHaveBeenCalledOnce()

    const second = makePopup()
    const secondManager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: () => second.popup,
      log: vi.fn(),
      openExternal: vi.fn(() => true),
      session: {}
    })
    await secondManager.open(invokingWindow)
    owner.setState({ ...readyState(), enabled: false, available: false, popupUrl: null, rulesetsReady: false })

    expect(second.popup.close).toHaveBeenCalledOnce()
    expect(owner.controller.getState().enabled).toBe(false)
    manager.dispose()
    secondManager.dispose()
  })

  it('keeps a failed or destroyed popup hidden and retryable', async () => {
    const owner = controller()
    let first = true
    const log = vi.fn()
    const firstPopup = makePopup(async () => {
      first = false
      throw new Error('load failed')
    })
    const retryPopup = makePopup()
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: vi.fn().mockReturnValueOnce(firstPopup.popup).mockReturnValueOnce(retryPopup.popup),
      log,
      openExternal: vi.fn(() => true),
      session: {}
    })
    const invokingWindow = makeParent()

    await expect(manager.open(invokingWindow)).rejects.toThrow('could not be opened')
    expect(firstPopup.popup.show).not.toHaveBeenCalled()
    expect(first).toBe(false)
    expect(log).toHaveBeenCalledWith('[preview] uBlock popup resource failed to load')
    await manager.open(invokingWindow)
    expect(retryPopup.popup.show).toHaveBeenCalledOnce()
  })

  it('uses a stable creation diagnostic without recording the thrown error', async () => {
    const owner = controller()
    const log = vi.fn()
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: () => {
        throw new Error('secret filesystem path')
      },
      log,
      openExternal: vi.fn(() => true),
      session: {}
    })

    await expect(manager.open(makeParent())).rejects.toThrow('could not be opened')
    expect(log).toHaveBeenCalledWith('[preview] uBlock popup creation failed')
    expect(log).not.toHaveBeenCalledWith(expect.stringContaining('secret filesystem path'))
  })

  it('denies every window.open request but forwards only safe external schemes', async () => {
    const owner = controller()
    const created = makePopup()
    const openExternal = vi.fn(() => true)
    const manager = createPreviewUblockPopupManager({
      controller: owner.controller,
      createWindow: () => created.popup,
      log: vi.fn(),
      openExternal,
      session: {}
    })
    const invokingWindow = makeParent()

    await manager.open(invokingWindow)
    const windowOpenHandler = created.popup.webContents.setWindowOpenHandler.mock.calls[0][0]

    expect(windowOpenHandler({ url: 'chrome-extension://ublock/dashboard.html' })).toEqual({ action: 'deny' })
    expect(windowOpenHandler({ url: 'https://example.com/help' })).toEqual({ action: 'deny' })
    expect(windowOpenHandler({ url: 'mailto:support@example.com' })).toEqual({ action: 'deny' })
    expect(windowOpenHandler({ url: 'file:///tmp/secret' })).toEqual({ action: 'deny' })
    expect(windowOpenHandler({ url: 'javascript:alert(1)' })).toEqual({ action: 'deny' })
    expect(openExternal).toHaveBeenCalledWith('https://example.com/help')
    expect(openExternal).toHaveBeenCalledWith('mailto:support@example.com')
    expect(openExternal).not.toHaveBeenCalledWith('file:///tmp/secret')

    const internalNavigation = { preventDefault: vi.fn() }
    created.target.emit('will-navigate', internalNavigation, 'chrome-extension://ublock/dashboard.html')
    expect(internalNavigation.preventDefault).not.toHaveBeenCalled()

    const externalNavigation = { preventDefault: vi.fn() }
    created.target.emit('will-navigate', externalNavigation, 'https://example.com/help')
    expect(externalNavigation.preventDefault).toHaveBeenCalledOnce()
    expect(openExternal).toHaveBeenCalledWith('https://example.com/help')

    const malformedNavigation = { preventDefault: vi.fn() }
    created.target.emit('will-navigate', malformedNavigation, 'javascript:alert(1)')
    expect(malformedNavigation.preventDefault).toHaveBeenCalledOnce()
    expect(openExternal).not.toHaveBeenCalledWith('javascript:alert(1)')
  })
})
