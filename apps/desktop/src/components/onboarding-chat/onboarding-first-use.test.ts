import { expect, it } from 'vitest'

import { buildFirstTaskRunbook } from '@/components/onboarding-chat/setup-profile'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { buildChatOnboardingSeedMessages, FIRST_USE_GUIDANCE } from '@/store/onboarding-script'

it('carries the same first-use guidance through the hidden guide and every first-task plan', () => {
  expect(FIRST_USE_GUIDANCE).toBeTruthy()

  const greeting = 'What should I call you?'
  const seeds = buildChatOnboardingSeedMessages(greeting)
  expect(seeds.filter(seed => seed.display_kind !== 'hidden')).toEqual([{ role: 'assistant', content: greeting }])

  const prompts = [
    seeds[0].content,
    ...(['build', 'machine-setup', 'plugin'] as const).map(plan =>
      buildFirstTaskRunbook('Organize my work', DEFAULT_ANSWERS, plan, '/tmp/example-plugins')
    )
  ]

  for (const prompt of prompts) {
    expect(prompt.split(FIRST_USE_GUIDANCE)).toHaveLength(2)
  }
})
