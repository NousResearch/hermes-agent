// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest } from './clarify'
import {
  $petOverlayActive,
  initPetOverlayBridge,
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
  it('parses typed action-center intents without trusting route identity fields', () => {
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
    ).toEqual({ type: 'action-center-approval', itemId: 'item-1', choice: 'deny', reason: 'unsafe' })
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
  })

  it.each([
    null,
    {},
    { type: 'action-center-select', itemId: '' },
    { type: 'action-center-approval', itemId: 'item', choice: 'root' },
    { type: 'action-center-approval', itemId: 'item', choice: 'approve-once', reason: 'not-for-approve' },
    { type: 'action-center-clarify', itemId: 42, answer: 'Blue' },
    { type: 'action-center-open-session', itemId: null }
  ])('ignores malformed payload %j', payload => {
    expect(parsePetOverlayControl(payload)).toBeNull()
  })

  it('forwards only parsed action-center controls to the registered main-renderer handler', () => {
    const handler = vi.fn()

    setPetOverlayActionCenterHandler(handler)
    const dispose = initPetOverlayBridge()

    onControl?.({ type: 'action-center-select', itemId: '' })
    onControl?.({ type: 'action-center-select', itemId: 'item-1', profile: 'wrong' })

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
