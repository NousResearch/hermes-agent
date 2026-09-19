import { expect, it } from 'vitest'

import { buildFirstTaskRunbook } from '@/components/onboarding-chat/setup-profile'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { buildChatOnboardingSeedMessages, FIRST_USE_GUIDANCE } from '@/store/onboarding-script'

it('carries the same first-use guidance through the hidden guide and every first-task plan', () => {
  expect(FIRST_USE_GUIDANCE).toBeTruthy()

  const greeting = 'What should I call you?'
  const seeds = buildChatOnboardingSeedMessages(greeting)
  // `buildChatOnboardingSeedMessages` returns typed `SeedMessage` rows now (the wire shape); compare the
  // fields this test is about, not the envelope.
  expect(seeds.filter(seed => seed.display_kind !== 'hidden').map(seed => ({ content: seed.content, role: seed.role })))
    .toEqual([{ content: greeting, role: 'assistant' }])

  // `SeedMessage.content` is nullable on the wire; this seed is text, and an empty string would
  // fail the length assertion below exactly as a null would.
  const prompts = [
    seeds[0].content ?? '',
    ...(['build', 'machine-setup', 'plugin'] as const).map(plan =>
      buildFirstTaskRunbook('Organize my work', DEFAULT_ANSWERS, plan, '/tmp/example-plugins')
    )
  ]

  for (const prompt of prompts) {
    expect(prompt.split(FIRST_USE_GUIDANCE)).toHaveLength(2)
  }
})
