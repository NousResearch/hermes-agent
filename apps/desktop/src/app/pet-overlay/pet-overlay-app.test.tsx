// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { usePetZoomGesture } from '@/components/pet/use-pet-zoom-gesture'
import { en } from '@/i18n/en'
import { setPetInfo } from '@/store/pet'
import type { PetActionCenterState } from '@/store/pet-action-center'
import type { PetOverlayStatePayload } from '@/store/pet-overlay'

import { PetOverlayApp } from './pet-overlay-app'

vi.mock('@/components/chat/vibe-hearts', () => ({
  PetHeartField: () => null,
  playVibeHearts: vi.fn()
}))

vi.mock('@/components/pet/pet-bubble', () => ({
  PetBubble: () => <div data-testid="pet-bubble" />
}))

vi.mock('@/components/pet/pet-sprite', () => ({
  PetSprite: () => <canvas data-testid="pet-sprite" height={1} width={1} />
}))

vi.mock('@/components/pet/use-pet-zoom-gesture', () => ({
  usePetZoomGesture: vi.fn()
}))

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const composerPlaceholder = en.pet.composerPlaceholder
const ac = en.pet.actionCenter

let controlMock: ReturnType<typeof vi.fn>
let setBoundsMock: ReturnType<typeof vi.fn>
let setFocusableMock: ReturnType<typeof vi.fn>
let setIgnoreMouseMock: ReturnType<typeof vi.fn>
let stateListener: ((payload: PetOverlayStatePayload) => void) | undefined
let windowBounds = { height: 300, width: 240, x: 100, y: 200 }

const initialResizeObserver = globalThis.ResizeObserver

class FakeResizeObserver {
  static instances: FakeResizeObserver[] = []

  readonly disconnect = vi.fn()
  readonly observe = vi.fn<(target: Element) => void>()
  readonly unobserve = vi.fn<(target: Element) => void>()

  constructor(private readonly callback: ResizeObserverCallback) {
    FakeResizeObserver.instances.push(this)
  }

  emit(width: number, height: number): void {
    const target = this.observe.mock.calls[0]?.[0]

    if (!target) {
      throw new Error('ResizeObserver target not observed')
    }

    this.callback(
      [{ contentRect: { height, width }, target } as ResizeObserverEntry],
      this as unknown as ResizeObserver
    )
  }
}

function makeActionCenterState(): PetActionCenterState {
  return {
    action: null,
    actionableCount: 1,
    attentionCount: 1,
    blockingCount: 1,
    items: [
      {
        actionable: true,
        allowPermanent: false,
        allowedActions: ['approve-once', 'deny'],
        blocking: true,
        choices: null,
        command: 'npm test',
        description: 'Run the test suite',
        detail: 'npm test',
        id: 'approval-1',
        kind: 'approval',
        profile: 'default',
        profileLabel: 'default',
        receivedAt: 1,
        sessionId: 'session-1',
        sessionTitle: 'Test session',
        smartDenied: false,
        storedSessionId: 'stored-1',
        summary: 'Run the test suite'
      }
    ],
    secureInputCount: 0,
    selectedItemId: 'approval-1'
  }
}

function makeWorkingActionCenterState(): PetActionCenterState {
  return {
    action: null,
    actionableCount: 1,
    attentionCount: 0,
    blockingCount: 0,
    items: [
      {
        actionable: true,
        activityKind: 'tool',
        activityName: 'Terminal',
        allowedActions: ['steer', 'queue', 'stop', 'open-in-app'],
        blocking: false,
        connectionState: 'open',
        detail: null,
        id: 'live-1',
        kind: 'live-turn',
        profile: 'default',
        profileLabel: 'Default',
        queuedCount: 0,
        receivedAt: 1,
        sessionId: 'runtime-1',
        sessionTitle: 'Working session',
        status: 'working',
        storedSessionId: 'stored-1',
        summary: null,
        turnStartedAt: null
      }
    ],
    secureInputCount: 0,
    selectedItemId: 'live-1'
  }
}

function makePayload(actionCenter = makeActionCenterState()): PetOverlayStatePayload {
  return {
    actionCenter,
    activity: {},
    awaiting: false,
    busy: false,
    info: { enabled: true, spritesheetBase64: 'data:image/png;base64,AA==' },
    reaction: null,
    unread: false
  }
}

function pushState(payload = makePayload()) {
  act(() => stateListener?.(payload))
}

function installDesktopMock() {
  controlMock = vi.fn()
  setBoundsMock = vi.fn(async bounds => {
    windowBounds = bounds

    return { bounds, ok: true }
  })
  setFocusableMock = vi.fn()
  setIgnoreMouseMock = vi.fn()
  stateListener = undefined

  desktopWindow.hermesDesktop = {
    petOverlay: {
      close: vi.fn().mockResolvedValue({ ok: true }),
      control: controlMock,
      onControl: vi.fn(() => () => {}),
      onState: vi.fn(callback => {
        stateListener = callback

        return () => {
          stateListener = undefined
        }
      }),
      open: vi.fn().mockResolvedValue({ ok: true }),
      pushState: vi.fn(),
      setBounds: setBoundsMock,
      setFocusable: setFocusableMock,
      setIgnoreMouse: setIgnoreMouseMock
    }
  } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  vi.useFakeTimers()
  FakeResizeObserver.instances = []
  globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver
  installDesktopMock()
  setPetInfo({ enabled: true, spritesheetBase64: 'data:image/png;base64,AA==' })
  windowBounds = { height: 300, width: 240, x: 100, y: 200 }
  Object.defineProperties(window, {
    outerHeight: { configurable: true, get: () => windowBounds.height },
    outerWidth: { configurable: true, get: () => windowBounds.width },
    screenX: { configurable: true, get: () => windowBounds.x },
    screenY: { configurable: true, get: () => windowBounds.y }
  })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  globalThis.ResizeObserver = initialResizeObserver
  setPetInfo({ enabled: false })

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('PetOverlayApp action-center integration', () => {
  it('mirrors a zero-attention working session without auto-opening or stealing focus, then opens explicitly', () => {
    render(<PetOverlayApp />)

    pushState(makePayload(makeWorkingActionCenterState()))

    const trigger = screen.getByRole('button', { name: ac.open })
    const interactiveRoot = document.querySelector('[data-pet-overlay-interactive-root]')

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(setFocusableMock.mock.calls.some(([focusable]) => focusable === true)).toBe(false)
    expect(interactiveRoot?.contains(trigger)).toBe(true)

    fireEvent.click(trigger)

    expect(screen.getByRole('dialog')).not.toBeNull()
    expect(setFocusableMock).toHaveBeenCalledWith(true)
    expect(setIgnoreMouseMock).toHaveBeenCalledWith(false)
  })

  it('returns to non-focusable on action-center close when the composer is closed', () => {
    render(<PetOverlayApp />)
    pushState()

    fireEvent.click(screen.getByRole('button', { name: ac.open }))
    setFocusableMock.mockClear()
    fireEvent.click(screen.getByRole('button', { name: /close action center/i }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(setFocusableMock).toHaveBeenCalledWith(false)
  })

  it('keeps focusability through the composer-to-action-center handoff and disables it after panel close', () => {
    render(<PetOverlayApp />)
    pushState()

    const sprite = screen.getByTestId('pet-sprite')
    fireEvent.pointerDown(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    fireEvent.pointerUp(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    act(() => vi.advanceTimersByTime(251))

    expect(screen.getByPlaceholderText(composerPlaceholder)).not.toBeNull()
    setFocusableMock.mockClear()
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    expect(screen.queryByPlaceholderText(composerPlaceholder)).toBeNull()
    expect(setFocusableMock).toHaveBeenCalledWith(true)
    expect(setFocusableMock.mock.calls.some(([focusable]) => focusable === false)).toBe(false)

    setFocusableMock.mockClear()
    fireEvent.click(screen.getByRole('button', { name: /close action center/i }))

    expect(setFocusableMock).toHaveBeenCalledWith(false)
  })

  it('stops action-center pointer events before they reach pet gestures', () => {
    render(<PetOverlayApp />)
    pushState()
    controlMock.mockClear()

    const trigger = screen.getByRole('button', { name: ac.open })
    fireEvent.pointerDown(trigger, { button: 0, pointerId: 1, screenX: 10, screenY: 10 })
    fireEvent.pointerUp(trigger, { button: 0, pointerId: 1, screenX: 10, screenY: 10 })
    fireEvent.click(trigger)
    act(() => vi.advanceTimersByTime(251))

    const petGestureControlSent = controlMock.mock.calls.some(([control]) =>
      ['submit', 'toggle-app'].includes((control as { type?: string }).type ?? '')
    )

    expect(petGestureControlSent).toBe(false)
    expect(screen.queryByPlaceholderText(composerPlaceholder)).toBeNull()
    expect(screen.getByRole('dialog')).not.toBeNull()
  })

  it('suppresses the deferred composer single-click while the action center is open and keeps dialog focus', () => {
    render(<PetOverlayApp />)
    pushState()

    fireEvent.click(screen.getByRole('button', { name: ac.open }))
    act(() => vi.advanceTimersByTime(16))
    const dialog = screen.getByRole('dialog')
    expect(document.activeElement).toBe(dialog)

    const sprite = screen.getByTestId('pet-sprite')
    fireEvent.pointerDown(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    fireEvent.pointerUp(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    act(() => vi.advanceTimersByTime(251))

    expect(screen.queryByPlaceholderText(composerPlaceholder)).toBeNull()
    expect(dialog.contains(document.activeElement) || document.activeElement === dialog).toBe(true)
  })

  it('does not replay a panel-time sprite click as a composer open after a rapid panel close', () => {
    render(<PetOverlayApp />)
    pushState()

    fireEvent.click(screen.getByRole('button', { name: ac.open }))
    const sprite = screen.getByTestId('pet-sprite')
    fireEvent.pointerDown(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    fireEvent.pointerUp(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    fireEvent.click(screen.getByRole('button', { name: ac.close }))
    act(() => vi.advanceTimersByTime(251))

    expect(screen.queryByPlaceholderText(composerPlaceholder)).toBeNull()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('atomically closes the composer before opening the action center without dropping focusability', () => {
    render(<PetOverlayApp />)
    pushState()

    const sprite = screen.getByTestId('pet-sprite')
    fireEvent.pointerDown(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    fireEvent.pointerUp(sprite, { button: 0, pointerId: 1, screenX: 20, screenY: 20 })
    act(() => vi.advanceTimersByTime(267))
    expect(screen.getByPlaceholderText(composerPlaceholder)).not.toBeNull()

    setFocusableMock.mockClear()
    fireEvent.click(screen.getByRole('button', { name: ac.open }))
    act(() => vi.advanceTimersByTime(16))

    const dialog = screen.getByRole('dialog')
    expect(screen.queryByPlaceholderText(composerPlaceholder)).toBeNull()
    expect(document.activeElement).toBe(dialog)
    expect(setFocusableMock).toHaveBeenCalledWith(true)
    expect(setFocusableMock.mock.calls.some(([focusable]) => focusable === false)).toBe(false)
  })

  it('measures the expanded action center, grows without cropping, then collapses at the same feet anchor', async () => {
    render(<PetOverlayApp />)
    pushState()
    const observer = FakeResizeObserver.instances[0]!
    const originalCenter = windowBounds.x + windowBounds.width / 2
    const originalBottom = windowBounds.y + windowBounds.height

    observer.emit(180, 200)
    expect(setBoundsMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: ac.open }))
    await act(async () => observer.emit(340, 500))

    const expanded = setBoundsMock.mock.calls.at(-1)?.[0]
    expect(expanded.width).toBeGreaterThan(340)
    expect(expanded.height).toBeGreaterThan(500)
    expect(expanded.x + expanded.width / 2).toBe(originalCenter)
    expect(expanded.y + expanded.height).toBe(originalBottom)

    fireEvent.click(screen.getByRole('button', { name: ac.close }))
    await act(async () => observer.emit(180, 200))

    const collapsed = setBoundsMock.mock.calls.at(-1)?.[0]
    expect(collapsed).toEqual({ height: 300, width: 240, x: 100, y: 200 })
    expect(collapsed.x + collapsed.width / 2).toBe(originalCenter)
    expect(collapsed.y + collapsed.height).toBe(originalBottom)
  })

  it('ignores repeated identical measurements and disconnects the observer on unmount', async () => {
    const view = render(<PetOverlayApp />)
    pushState()
    const observer = FakeResizeObserver.instances[0]!

    await act(async () => observer.emit(340, 500))
    const callCount = setBoundsMock.mock.calls.length
    await act(async () => observer.emit(340, 500))

    expect(setBoundsMock).toHaveBeenCalledTimes(callCount)
    view.unmount()
    expect(observer.disconnect).toHaveBeenCalledTimes(1)
  })

  it('persists the actual clamped bounds returned by the main process', async () => {
    const actual = { height: 420, width: 360, x: 0, y: 0 }

    setBoundsMock.mockImplementationOnce(async () => {
      windowBounds = actual

      return { bounds: actual, ok: true }
    })
    render(<PetOverlayApp />)
    pushState()

    await act(async () => FakeResizeObserver.instances[0]!.emit(340, 500))

    expect(controlMock).toHaveBeenCalledWith({ bounds: actual, type: 'bounds' })
  })

  it('falls back to compact geometry when ResizeObserver is unavailable', () => {
    // @ts-expect-error feature-detection contract: older Chromium may omit it.
    delete globalThis.ResizeObserver

    expect(() => render(<PetOverlayApp />)).not.toThrow()
    pushState()
    expect(setBoundsMock).not.toHaveBeenCalled()
  })

  it('keeps the existing Alt+wheel cursor ratio anchor while resizing', async () => {
    render(<PetOverlayApp />)
    pushState()
    setBoundsMock.mockClear()

    const onScale = vi.mocked(usePetZoomGesture).mock.calls.at(-1)?.[1]
    await act(async () => onScale?.(0.5, { clientX: 60, clientY: 80, ratio: 1.5 }))

    expect(setBoundsMock).toHaveBeenCalledWith({ height: 304, width: 240, x: 130, y: 294 })
  })
})
