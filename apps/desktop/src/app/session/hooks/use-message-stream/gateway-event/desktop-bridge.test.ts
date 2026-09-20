import { beforeEach, expect, test, vi } from 'vitest'

// Only the tip.show branch is under test; its sibling branches' collaborators
// are stubbed so importing the module stays free of their side effects.
// `@/store/session` stays real: the gateway-store chain other modules load at
// import time subscribes to its `$connection`.
vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane_focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', () => ({ recordAgentReaction: vi.fn() }))

import { $activeTip, $retiredTips, $tipsEnabled, resetTips, retireActiveTip } from '@/store/tips'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

function tipShow(payload: Record<string, unknown>, isActiveEvent = true): boolean {
  const ctx = { event: { type: 'tip.show' }, isActiveEvent, payload } as unknown as GatewayEventContext

  return handleDesktopBridgeEvent(ctx)
}

beforeEach(() => {
  resetTips()
  $tipsEnabled.set(true)
  $activeTip.set(null)
})

// The tool derives `tip_id` from selector+text (#117216) so an agent tip has
// the stable identity the ✕-ledger keys on; the bridge just has to carry it.
test('an agent tip carries its content id onto the bubble', () => {
  expect(tipShow({ selector: '#composer', text: 'Type here', tip_id: 'agent:abc123' })).toBe(true)

  expect($activeTip.get()).toMatchObject({ text: 'Type here', tipId: 'agent:abc123' })
})

test('content the user already closed never comes back', () => {
  $retiredTips.set(['agent:abc123'])

  expect(tipShow({ selector: '#composer', text: 'Type here', tip_id: 'agent:abc123' })).toBe(true)
  expect($activeTip.get()).toBeNull()
})

test('the ✕ on an agent tip retires its content across conversations', () => {
  tipShow({ selector: '#composer', text: 'Type here', tip_id: 'agent:abc123' })
  retireActiveTip()

  expect($retiredTips.get()).toContain('agent:abc123')

  // Same content in the next conversation: the event is consumed (true), but
  // the bubble is refused exactly like any other tip that has no slot left.
  expect(tipShow({ selector: '#composer', text: 'Type here', tip_id: 'agent:abc123' })).toBe(true)
  expect($activeTip.get()).toBeNull()
})

test('a retirement only silences that content, not the tool', () => {
  $retiredTips.set(['agent:abc123'])

  tipShow({ selector: '#composer', text: 'Something else worth pointing at', tip_id: 'agent:def456' })

  expect($activeTip.get()?.tipId).toBe('agent:def456')
})

test('a legacy event without an id still shows — old backend, new renderer', () => {
  tipShow({ selector: '#composer', text: 'Type here' })

  expect($activeTip.get()).toMatchObject({ text: 'Type here' })
  expect($activeTip.get()?.tipId).toBeUndefined()
})
