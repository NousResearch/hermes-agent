import { BrowserWindow, ipcRenderer } from 'electron'
import { afterEach, expect, it, vi } from 'vitest'

import { createIntroRevealWindowController, INTRO_REVEAL_WATCHDOG_MS } from './intro-reveal-window'

interface IntroPayload {
  hideMain?: boolean
  showMain?: boolean
}

interface IntroBridge {
  introReveal: {
    open: (payload?: IntroPayload) => Promise<{ ok: boolean }>
    close: (payload?: IntroPayload) => Promise<{ ok: boolean }>
  }
}

interface IntroInvokeEvent {
  channel: string
}

interface NativeTestState {
  bridge?: IntroBridge
  handlers: Map<string, (event: IntroInvokeEvent, payload?: IntroPayload) => { ok: boolean }>
}

const native = vi.hoisted(() => {
  const state: NativeTestState = { handlers: new Map() }

  return state
})

vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events')

  class WindowDouble extends EventEmitter {
    destroyed = false
    visible = false
    webContents = Object.assign(new EventEmitter(), { send: vi.fn() })
    isDestroyed = () => this.destroyed
    isVisible = () => this.visible
    setAlwaysOnTop = vi.fn()
    setVisibleOnAllWorkspaces = vi.fn()
    setVibrancy = vi.fn()
    setOpacity = vi.fn()
    show = () => {
      this.visible = true
    }
    hide = () => {
      this.visible = false
    }
    close = () => {
      this.destroyed = true
      this.emit('closed')
    }
    destroy = this.close
  }

  return {
    BrowserWindow: WindowDouble,
    contextBridge: {
      exposeInMainWorld: (_name: string, bridge: IntroBridge) => {
        native.bridge = bridge
      }
    },
    ipcMain: Object.assign(new EventEmitter(), {
      handle: (channel: string, handler: (event: IntroInvokeEvent, payload?: IntroPayload) => { ok: boolean }) => {
        native.handlers.set(channel, handler)
      }
    }),
    ipcRenderer: {
      sendSync: vi.fn(),
      invoke: vi.fn(async (channel: string, payload?: IntroPayload) =>
        native.handlers.get(channel)?.({ channel }, payload)
      )
    },
    screen: { getPrimaryDisplay: () => ({ bounds: { x: 0, y: 0, width: 1440, height: 900 } }) },
    webFrame: {},
    webUtils: {}
  }
})

afterEach(() => {
  vi.useRealTimers()
  native.handlers.clear()
})

it('keeps the independent native watchdog longer than the renderer deadman', async () => {
  // A dynamic path keeps the Electron TS project from compiling renderer sources.
  const rendererTimeline = '../src/components/intro-reveal/timeline.ts'
  const timeline: { INTRO_DEADMAN_MS: number } = await import(rendererTimeline)

  expect(INTRO_REVEAL_WATCHDOG_MS).toBeGreaterThan(timeline.INTRO_DEADMAN_MS)
})

it('forwards ownership payloads and restores the app on close or a stalled renderer', async () => {
  vi.useFakeTimers()
  await import('./preload')
  const main = new BrowserWindow()
  const showMain = vi.fn(() => main.show())

  const options = {
    enabled: false,
    isMac: process.platform === 'darwin',
    loadWindowUrl: vi.fn(),
    log: vi.fn(),
    mainWindow: () => main,
    preloadPath: '/test/preload.cjs',
    rendererIndex: () => '/test/index.html',
    showMain,
    wireWindow: vi.fn()
  }

  const disabled = createIntroRevealWindowController(options)

  expect(await native.bridge?.introReveal.open({ hideMain: true })).toEqual({ ok: false })
  expect(options.loadWindowUrl).not.toHaveBeenCalled()
  disabled.destroy()

  const controller = createIntroRevealWindowController({ ...options, enabled: true })

  // Main starts unshown; closing still has to reveal it.
  await native.bridge?.introReveal.open({ hideMain: true })
  expect(main.isVisible()).toBe(false)
  expect(ipcRenderer.invoke).toHaveBeenCalledWith('hermes:intro-reveal:open', { hideMain: true })
  expect(options.loadWindowUrl).toHaveBeenCalledWith(
    expect.anything(),
    expect.stringContaining('?win=intro'),
    'Intro reveal'
  )
  await native.bridge?.introReveal.close({ showMain: true })
  expect(ipcRenderer.invoke).toHaveBeenCalledWith('hermes:intro-reveal:close', { showMain: true })
  expect(main.isVisible()).toBe(true)
  vi.advanceTimersByTime(INTRO_REVEAL_WATCHDOG_MS)
  expect(showMain).toHaveBeenCalledTimes(1)

  await native.bridge?.introReveal.open({ hideMain: true })
  expect(main.isVisible()).toBe(false)
  vi.advanceTimersByTime(INTRO_REVEAL_WATCHDOG_MS)
  expect(main.isVisible()).toBe(true)
  expect(showMain).toHaveBeenCalledTimes(2)
  controller.destroy()
})
