import { expect, it } from 'vitest'

import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'

import { CONNECTORS } from './options'
import { buildFirstTaskRunbook } from './setup-profile'

it('ignores unoffered connector picks in the first task runbook', () => {
  for (const connectors of [[], CONNECTORS.map(connector => connector.id)]) {
    const answers = { ...DEFAULT_ANSWERS, connectors }
    expect(buildFirstTaskRunbook('Make a task tracker', {
      ...answers, connectors: [...connectors, 'retired-app', '', 'google-calendar']
    })).toBe(buildFirstTaskRunbook('Make a task tracker', answers))
  }
})
