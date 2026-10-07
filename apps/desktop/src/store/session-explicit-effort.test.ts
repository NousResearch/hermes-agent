import { afterEach, describe, expect, it } from 'vitest'

import { $currentReasoningEffort, $defaultReasoningEffort, explicitEffortPick } from './session'

describe('explicitEffortPick', () => {
  afterEach(() => {
    $currentReasoningEffort.set('')
    $defaultReasoningEffort.set('')
  })

  it('drops a composer effort that only mirrors the profile default', () => {
    $defaultReasoningEffort.set('high')
    $currentReasoningEffort.set(' High ')

    expect(explicitEffortPick()).toBe('')

    $currentReasoningEffort.set('')

    expect(explicitEffortPick()).toBe('')
  })

  it('ships a distinct pick, trimmed', () => {
    $defaultReasoningEffort.set('medium')
    $currentReasoningEffort.set(' high ')

    expect(explicitEffortPick()).toBe('high')

    $currentReasoningEffort.set('none')

    expect(explicitEffortPick()).toBe('none')
  })
})
