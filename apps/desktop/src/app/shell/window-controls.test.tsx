import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { act } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import { setConnection } from '@/store/session'

import { WindowControls } from './window-controls'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const WCO_API = {
  visible: true,
  getTitlebarAreaRect: () => ({ right: 100, width: 100 }) as DOMRect,
  addEventListener: () => undefined,
  removeEventListener: () => undefined
}

function setWcoAvailable(available: boolean) {
  Object.defineProperty(navigator, 'windowControlsOverlay', {
    configurable: true,
    value: available ? WCO_API : undefined
  })
}

function connect(overrides: Partial<HermesConnection> = {}) {
  setConnection({
    baseUrl: 'http://box:9119',
    isFullscreen: false,
    mode: 'local',
    nativeOverlayWidth: 144,
    token: 't',
    wsUrl: 'ws://box:9119',
    logs: [],
    windowButtonPosition: null,
    windowChromeMode: 'overlay',
    ...overrides
  })
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
})

afterEach(() => {
  cleanup()
  setConnection(null)
  setWcoAvailable(true)
})

describe('WindowControls (app-drawn window chrome)', () => {
  it('renders nothing in default overlay mode with a native overlay present', () => {
    connect({ windowChromeMode: 'overlay' })
    setWcoAvailable(true)

    render(
      <I18nProvider>
        <WindowControls />
      </I18nProvider>
    )

    expect(screen.queryByRole('button', { name: /minimize/i })).toBeNull()
  })

  it('renders min/max/close in app-drawn mode and drives window controls over IPC', () => {
    connect({ windowChromeMode: 'app-drawn' })
    const minimize = vi.fn()
    const toggleMaximize = vi.fn()
    const close = vi.fn()
    vi.stubGlobal('hermesDesktop', { windowControls: { minimize, toggleMaximize, close } })

    render(
      <I18nProvider>
        <WindowControls />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: /minimize/i }))
    fireEvent.click(screen.getByRole('button', { name: /maximize/i }))
    fireEvent.click(screen.getByRole('button', { name: /close/i }))

    expect(minimize).toHaveBeenCalledTimes(1)
    expect(toggleMaximize).toHaveBeenCalledTimes(1)
    expect(close).toHaveBeenCalledTimes(1)
  })

  it('falls back to app-drawn controls when no native overlay exists (WSLg bug class)', () => {
    connect({ windowChromeMode: 'overlay', windowButtonPosition: null })
    setWcoAvailable(false)

    render(
      <I18nProvider>
        <WindowControls />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: /minimize/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /close/i })).toBeTruthy()
  })

  it('never falls back on macOS (traffic lights are native, windowButtonPosition set)', () => {
    connect({ windowChromeMode: 'overlay', windowButtonPosition: { x: 24, y: 10 } })
    setWcoAvailable(false)

    render(
      <I18nProvider>
        <WindowControls />
      </I18nProvider>
    )

    expect(screen.queryByRole('button', { name: /minimize/i })).toBeNull()
  })

  it('flips the center button to restore when the window is maximized', () => {
    connect({ windowChromeMode: 'app-drawn' })
    let onState: ((state: { isMaximized?: boolean }) => void) | undefined
    vi.stubGlobal('hermesDesktop', {
      windowControls: { minimize: vi.fn(), toggleMaximize: vi.fn(), close: vi.fn() },
      onWindowStateChanged: (callback: (state: { isMaximized?: boolean }) => void) => {
        onState = callback

        return () => undefined
      }
    })

    render(
      <I18nProvider>
        <WindowControls />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: /maximize/i })).toBeTruthy()

    act(() => onState?.({ isMaximized: true }))

    expect(screen.getByRole('button', { name: /restore/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /maximize/i })).toBeNull()
  })
})
