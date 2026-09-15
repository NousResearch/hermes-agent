import { describe, expect, it } from 'vitest'

import { buildFirstTaskRunbook } from '@/components/onboarding-chat/setup-profile'
import type { OnboardingAnswers } from '@/store/onboarding-answers'

const answers = (connectors: string[]): OnboardingAnswers => ({
  accent: null,
  committed: [],
  connectors,
  context: '',
  layout: 'basic',
  name: 'Sam'
})

describe('the first-build runbook', () => {
  it('connects the picked apps through one call on the operation and reads its per-app result', () => {
    const prompt = buildFirstTaskRunbook('Plan my week', answers(['gmail', 'googlecalendar']), 'build')

    expect(prompt).toContain('action="connect"')
    expect(prompt).toMatch(/gmail \(Gmail\), googlecalendar \(Google Calendar\)/)
    expect(prompt).toContain('not_connected')
    // The card owns waiting, Try again and the early exit; none of those is a model action any more.
    expect(prompt).not.toMatch(/the wait returns|action="wait"|Start with|Start without|already active|mints new links/)
  })

  it('keeps the no-account rule when nothing was picked', () => {
    const prompt = buildFirstTaskRunbook('Plan my week', answers([]), 'build')

    expect(prompt).toContain('NO external account')
    expect(prompt).not.toContain('CONNECT FIRST')
  })
})
