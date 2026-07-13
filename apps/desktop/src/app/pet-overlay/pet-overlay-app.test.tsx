// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
let setFocusableMock: ReturnType<typeof vi.fn>
let setIgnoreMouseMock: ReturnType<typeof vi.fn>
let stateListener: ((payload: PetOverlayStatePayload) => void) | undefined

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
      setBounds: vi.fn(),
      setFocusable: setFocusableMock,
      setIgnoreMouse: setIgnoreMouseMock
    }
  } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  vi.useFakeTimers()
  installDesktopMock()
  setPetInfo({ enabled: true, spritesheetBase64: 'data:image/png;base64,AA==' })
  Object.defineProperty(window, 'outerHeight', { configurable: true, value: 300 })
  Object.defineProperty(window, 'outerWidth', { configurable: true, value: 240 })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  setPetInfo({ enabled: false })

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('PetOverlayApp action-center integration', () => {
  it('mirrors pushed state without auto-opening or stealing focus, then opens explicitly inside the interactive root', () => {
    render(<PetOverlayApp />)

    pushState()

    const trigger = screen.getByRole('button', { name: /review pending actions/i })
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

    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))
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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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

    const trigger = screen.getByRole('button', { name: /review pending actions/i })
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
})
