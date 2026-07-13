import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest } from './clarify'
import { $petOverlayActive, popInPet, popOutPet } from './pet-overlay'
import { clearAllPrompts, setApprovalRequest } from './prompts'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const close = vi.fn().mockResolvedValue({ ok: true })
const open = vi.fn().mockResolvedValue({ ok: true })
const pushState = vi.fn()

beforeEach(() => {
  clearAllPrompts()
  clearClarifyRequest()
  $petOverlayActive.set(false)
  close.mockClear()
  open.mockClear()
  pushState.mockClear()
  desktopWindow.hermesDesktop = {
    petOverlay: {
      close,
      control: vi.fn(),
      onControl: vi.fn(() => () => {}),
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
  clearAllPrompts()
  clearClarifyRequest()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
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
