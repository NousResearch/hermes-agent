import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeTip, $retiredTips, $tipsEnabled, dismissTip, resetTips, retireActiveTip } from '@/store/tips'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', () => ({ recordAgentReaction: vi.fn() }))
vi.mock('@/store/session', () => ({ setMessages: vi.fn() }))

const tipShow = (text: string): GatewayEventContext =>
  ({
    event: { type: 'tip.show' },
    isActiveEvent: true,
    payload: { selector: '[data-tour="model-pill"]', text }
  }) as unknown as GatewayEventContext

beforeEach(() => {
  dismissTip()
  resetTips()
  $tipsEnabled.set(true)
})

describe('tip.show bridge (#117216)', () => {
  it('a ✕-closed agent tip does not come back on the next tip.show of the same content', () => {
    handleDesktopBridgeEvent(tipShow('Choose a model here.'))
    const tipId = $activeTip.get()?.tipId

    expect(tipId).toMatch(/^agent:/)

    retireActiveTip()
    expect($retiredTips.get()).toContain(tipId)

    handleDesktopBridgeEvent(tipShow('Choose a model here.'))
    expect($activeTip.get()).toBeNull()

    // Different content is a different tip and still shows.
    handleDesktopBridgeEvent(tipShow('Attach files with the paperclip.'))
    expect($activeTip.get()?.text).toBe('Attach files with the paperclip.')
  })
})
