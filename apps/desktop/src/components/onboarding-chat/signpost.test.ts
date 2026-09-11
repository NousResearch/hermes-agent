import { describe, expect, it } from 'vitest'

import { declinedLookAround } from '@/components/onboarding-chat/signpost'
import type { ChatMessage } from '@/lib/chat-messages'
import { TOUR_OPTIONS } from '@/store/onboarding-script'

const turn = (role: 'assistant' | 'user', text: string): ChatMessage => ({
  id: `${role}-${text}`,
  parts: [{ text, type: 'text' }],
  role,
  timestamp: 0
})

describe('the parting signpost', () => {
  // The pick is a real user turn in the guide transcript and the option text is
  // pinned by the runbook — both sides read TOUR_OPTIONS, so neither can drift.
  it('reads the opt-out off the guide transcript', () => {
    expect(declinedLookAround([turn('assistant', 'Want a look around first?'), turn('user', TOUR_OPTIONS.none)])).toBe(
      true
    )
  })

  it('leaves it on for the two answers that did want showing around', () => {
    for (const option of [TOUR_OPTIONS.basics, TOUR_OPTIONS.tour]) {
      expect(declinedLookAround([turn('user', option)])).toBe(false)
    }
  })

  // The runbook has Hermes quote the options back in prose nowhere, but a model
  // that narrates one must not switch the signpost off — only the USER declines.
  it('ignores the same words coming from Hermes', () => {
    expect(declinedLookAround([turn('assistant', TOUR_OPTIONS.none)])).toBe(false)
  })
})
