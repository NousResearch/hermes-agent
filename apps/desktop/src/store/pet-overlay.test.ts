// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest } from './clarify'
import {
  $petOverlayActive,
  anchoredOverlayBounds,
  initPetOverlayBridge,
  overlayWindowSize,
  overlayWindowTargetSize,
  parsePetOverlayControl,
  popInPet,
  popOutPet,
  setPetOverlayActionCenterHandler
} from './pet-overlay'
import { clearAllPrompts, setApprovalRequest } from './prompts'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const close = vi.fn().mockResolvedValue({ ok: true })
const open = vi.fn().mockResolvedValue({ ok: true })
const pushState = vi.fn()
let onControl: ((payload: unknown) => void) | null = null

beforeEach(() => {
  clearAllPrompts()
  clearClarifyRequest()
  $petOverlayActive.set(false)
  close.mockClear()
  open.mockClear()
  pushState.mockClear()
  onControl = null
  desktopWindow.hermesDesktop = {
    petOverlay: {
      close,
      control: vi.fn(),
      onControl: vi.fn(callback => {
        onControl = callback

        return () => {
          onControl = null
        }
      }),
      onState: vi.fn(() => () => {}),
      open,
      pushState,
      setBounds: vi.fn(),
      setFocusable: vi.fn(),
      setIgnoreMouse: vi.fn()
    }
  } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  popInPet()
  setPetOverlayActionCenterHandler(null)
  clearAllPrompts()
  clearClarifyRequest()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('pet overlay control boundary', () => {
  it('parses exact typed action-center intents and rejects injected route identity fields', () => {
    expect(
      parsePetOverlayControl({
        type: 'action-center-approval',
        itemId: 'item-1',
        choice: 'deny',
        reason: 'unsafe',
        profile: 'attacker',
        sessionId: 'wrong',
        route: '/session/wrong'
      })
    ).toBeNull()
    expect(parsePetOverlayControl({ type: 'action-center-clarify', itemId: 'item-2', answer: 'Blue' })).toEqual({
      type: 'action-center-clarify',
      itemId: 'item-2',
      answer: 'Blue'
    })
    expect(parsePetOverlayControl({ type: 'action-center-select', itemId: 'item-3' })).toEqual({
      type: 'action-center-select',
      itemId: 'item-3'
    })
    expect(parsePetOverlayControl({ type: 'action-center-open-session', itemId: 'item-4' })).toEqual({
      type: 'action-center-open-session',
      itemId: 'item-4'
    })
    expect(parsePetOverlayControl({ type: 'action-center-submit', itemId: 'item-5', text: ' hello ' })).toEqual({
      type: 'action-center-submit',
      itemId: 'item-5',
      text: ' hello '
    })
    expect(parsePetOverlayControl({ type: 'action-center-steer', itemId: 'item-6', text: ' nudge ' })).toEqual({
      type: 'action-center-steer',
      itemId: 'item-6',
      text: ' nudge '
    })
    expect(parsePetOverlayControl({ type: 'action-center-queue', itemId: 'item-7', text: ' later ' })).toEqual({
      type: 'action-center-queue',
      itemId: 'item-7',
      text: ' later '
    })
    expect(parsePetOverlayControl({ type: 'action-center-stop', itemId: 'item-8' })).toEqual({
      type: 'action-center-stop',
      itemId: 'item-8'
    })
    expect(parsePetOverlayControl({ type: 'action-center-acknowledge', itemId: 'item-9' })).toEqual({
      type: 'action-center-acknowledge',
      itemId: 'item-9'
    })
  })

  it.each([
    null,
    {},
    { type: 'action-center-select', itemId: '' },
    { type: 'action-center-approval', itemId: 'item', choice: 'root' },
    { type: 'action-center-approval', itemId: 'item', choice: 'approve-once', reason: 'not-for-approve' },
    { type: 'action-center-clarify', itemId: 42, answer: 'Blue' },
    { type: 'action-center-open-session', itemId: null },
    { type: 'action-center-submit', itemId: 'item', text: 42 },
    { type: 'action-center-steer', itemId: '', text: 'nudge' },
    { type: 'action-center-queue', itemId: 'item' },
    { type: 'action-center-stop', itemId: 'item', text: 'extra' },
    { type: 'action-center-acknowledge', itemId: 'item', profile: 'work' },
    { type: 'action-center-submit', itemId: 'item', text: 'hello', sessionId: 'runtime' }
  ])('ignores malformed payload %j', payload => {
    expect(parsePetOverlayControl(payload)).toBeNull()
  })

  it('forwards only parsed action-center controls to the registered main-renderer handler', () => {
    const handler = vi.fn()

    setPetOverlayActionCenterHandler(handler)
    const dispose = initPetOverlayBridge()

    onControl?.({ type: 'action-center-select', itemId: '' })
    onControl?.({ type: 'action-center-select', itemId: 'item-1', profile: 'wrong' })
    onControl?.({ type: 'action-center-select', itemId: 'item-1' })

    expect(handler).toHaveBeenCalledTimes(1)
    expect(handler).toHaveBeenCalledWith({ type: 'action-center-select', itemId: 'item-1' })

    dispose()
  })
})

describe('pet overlay action-center state bridge', () => {
  it('includes the current projection and pushes request updates while open', async () => {
    setApprovalRequest({
      command: 'npm test',
      description: 'Run tests',
      profile: 'default',
      sessionId: 'runtime-1'
    })

    popOutPet({ height: 80, width: 80, x: 10, y: 20 })
    await Promise.resolve()

    expect(pushState).toHaveBeenCalled()
    expect(pushState.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        actionCenter: expect.objectContaining({
          items: [expect.objectContaining({ id: expect.any(String), kind: 'approval', sessionId: 'runtime-1' })]
        })
      })
    )

    pushState.mockClear()
    setApprovalRequest({
      command: 'npm run typecheck',
      description: 'Run TypeScript checks',
      profile: 'work',
      sessionId: 'runtime-2'
    })

    expect(pushState).toHaveBeenCalled()
    expect(pushState.mock.calls.at(-1)?.[0].actionCenter.items).toHaveLength(2)
  })
})

describe('pet overlay window geometry', () => {
  const compact = overlayWindowSize(192, 208, 0.33)
  const currentBounds = { height: compact.height, width: compact.width, x: 100, y: 200 }

  it('grows beyond compact bounds for a wide, tall measured action panel and collapses exactly', () => {
    const expanded = overlayWindowTargetSize(192, 208, 0.33, { height: 480, width: 321 })
    const collapsed = overlayWindowTargetSize(192, 208, 0.33, { height: 0, width: 0 })

    expect(expanded.width).toBeGreaterThan(compact.width)
    expect(expanded.height).toBeGreaterThan(compact.height)
    expect(expanded.width).toBeGreaterThan(321)
    expect(expanded.height).toBeGreaterThan(480)
    expect(collapsed).toEqual(compact)
  })

  it('keeps the pet feet bottom-center anchor exact through repeated expand/collapse cycles', () => {
    const expandedSize = overlayWindowTargetSize(192, 208, 0.33, { height: 480, width: 321 })
    const originalCenter = currentBounds.x + currentBounds.width / 2
    const originalBottom = currentBounds.y + currentBounds.height
    let bounds = currentBounds

    for (let index = 0; index < 10; index += 1) {
      bounds = anchoredOverlayBounds({ currentBounds: bounds, paddingBottom: 24, targetSize: expandedSize })
      expect(bounds.x + bounds.width / 2).toBe(originalCenter)
      expect(bounds.y + bounds.height).toBe(originalBottom)

      bounds = anchoredOverlayBounds({ currentBounds: bounds, paddingBottom: 24, targetSize: compact })
      expect(bounds).toEqual(currentBounds)
    }
  })

  it('sanitizes invalid measurements and dimensions to a safe compact integer size', () => {
    const target = overlayWindowTargetSize(Number.NaN, -100, Number.POSITIVE_INFINITY, {
      height: Number.NEGATIVE_INFINITY,
      width: -320
    })

    expect(target).toEqual({ height: 300, width: 240 })
    expect(Number.isInteger(target.width)).toBe(true)
    expect(Number.isInteger(target.height)).toBe(true)
  })

  it('preserves the existing wheel cursor/ratio anchor formula', () => {
    const targetSize = { height: 360, width: 300 }
    const wheelAnchor = { clientX: 60, clientY: 80, ratio: 1.5 }

    expect(anchoredOverlayBounds({ currentBounds, paddingBottom: 24, targetSize, wheelAnchor })).toEqual({
      height: targetSize.height,
      width: targetSize.width,
      x: Math.round(
        currentBounds.x +
          wheelAnchor.clientX -
          (wheelAnchor.clientX - currentBounds.width / 2) * wheelAnchor.ratio -
          targetSize.width / 2
      ),
      y: Math.round(
        currentBounds.y +
          wheelAnchor.clientY -
          (wheelAnchor.clientY - (currentBounds.height - 24)) * wheelAnchor.ratio -
          (targetSize.height - 24)
      )
    })
  })
})
