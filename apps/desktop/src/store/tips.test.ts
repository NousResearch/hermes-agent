import { beforeEach, describe, expect, it } from 'vitest'

import { $activeTip, $retiredTips, agentTipId, dismissTip, resetTips, retireActiveTip, showTip } from './tips'

const showAgentTip = (selector: string, text: string) => {
  showTip({
    side: 'top',
    targets: [selector],
    text,
    tipId: agentTipId(selector, text)
  })
}

beforeEach(() => {
  dismissTip()
  resetTips()
})

describe('agent tip retirement', () => {
  it('does not show the same agent tip again after the user retires it', () => {
    showAgentTip('[data-tour="model-pill"]', 'Choose a model here.')
    const tipId = $activeTip.get()?.tipId

    retireActiveTip()
    showAgentTip('[data-tour="model-pill"]', 'Choose a model here.')

    expect(tipId).toBe(agentTipId('[data-tour="model-pill"]', 'Choose a model here.'))
    expect(tipId).not.toContain('Choose a model here.')
    expect($retiredTips.get()).toContain(tipId)
    expect($activeTip.get()).toBeNull()
  })

  it('keeps distinct agent tips independently dismissible', () => {
    showAgentTip('[data-tour="model-pill"]', 'Choose a model here.')
    retireActiveTip()

    showAgentTip('[data-tour="composer"]', 'Write your prompt here.')

    expect($activeTip.get()?.tipId).toBe(agentTipId('[data-tour="composer"]', 'Write your prompt here.'))
  })
})
