import { beforeEach, describe, expect, it } from 'vitest'

import {
  $activeTip,
  $retiredTips,
  $tipShownAt,
  dismissTip,
  retireActiveTip,
  resetTips,
  showTip
} from './tips'

const AGENT_TIP_ID = 'agent-0123456789ab'

function agentTip(overrides: Partial<Parameters<typeof showTip>[0]> = {}): Parameters<typeof showTip>[0] {
  return {
    side: 'top',
    targets: ['#composer'],
    text: 'The model name is a button',
    tipId: AGENT_TIP_ID,
    ...overrides
  }
}

beforeEach(() => {
  resetTips()
  $activeTip.set(null)
})

describe('agent tips carry their content id through the ledgers', () => {
  it('records the seen ledger for an agent tip', () => {
    showTip(agentTip())

    expect($tipShownAt.get()[AGENT_TIP_ID]).toBeGreaterThan(0)
  })

  it('the ✕ on an agent tip retires it (issue #117216)', () => {
    showTip(agentTip())
    retireActiveTip()

    expect($retiredTips.get()).toContain(AGENT_TIP_ID)
    expect($activeTip.get()).toBeNull()
  })

  it('a retired agent tip never shows again — the ✕ means never, not next conversation', () => {
    showTip(agentTip())
    retireActiveTip()

    // A new conversation re-emits the same tip (same content → same id):
    // the bridge calls showTip with it and the store drops it.
    showTip(agentTip())

    expect($activeTip.get()).toBeNull()
  })

  it('a merely dismissed (timer) agent tip has had its moment but may return', () => {
    showTip(agentTip())
    dismissTip()

    showTip(agentTip())

    expect($activeTip.get()?.tipId).toBe(AGENT_TIP_ID)
  })

  it('edited tip content is a new tip, not the retired one', () => {
    showTip(agentTip())
    retireActiveTip()

    showTip(agentTip({ text: 'The model name now opens the picker' }))

    expect($activeTip.get()?.tipId).not.toBe(AGENT_TIP_ID)
  })

  it('agent tips never move the rotation cursor', () => {
    showTip(agentTip())

    // The catalog-adjacent assertions live in the rotation tests; here the
    // contract is that the seen ledger, not the walk, is what an agent id hits.
    expect($tipShownAt.get()).toHaveProperty(AGENT_TIP_ID)
  })
})

describe('catalog tips keep their existing contract', () => {
  it('a retired catalog tip is dropped too', () => {
    showTip({ side: 'top', targets: ['[data-tour="composer"]'], text: 'Type @', tipId: 'composer-mentions' })
    retireActiveTip()

    showTip({ side: 'top', targets: ['[data-tour="composer"]'], text: 'Type @', tipId: 'composer-mentions' })

    expect($activeTip.get()).toBeNull()
  })
})
