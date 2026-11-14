import { describe, expect, it } from 'vitest'

import { canonicalPromptParams } from './canonical-protocol'

// The guardian's escalation reason travels PendingControls -> approval.request event ->
// canonicalPromptParams -> the renderer's approval handler. A whitelist rebuild of the
// params used to drop smart_reason/smart_denied here, so the card rendered blank even
// though both ends carried the field (approval-explain).
describe('canonicalPromptParams approval projection', () => {
  const base = {
    choices: ['once', 'session', 'deny'],
    command: 'rm -rf /tmp/x',
    description: 'recursive delete',
    prompt_id: 'req-1'
  }

  it('forwards smart_reason and smart_denied to the renderer', () => {
    const params = canonicalPromptParams('approval', 's1', {
      ...base,
      smart_denied: true,
      smart_reason: 'prints a credential value to stdout'
    })

    expect(params.smart_reason).toBe('prints a credential value to stdout')
    expect(params.smart_denied).toBe(true)
  })

  it('omits the keys when the guardian produced no reason', () => {
    const params = canonicalPromptParams('approval', 's1', base)

    expect('smart_reason' in params).toBe(false)
    expect('smart_denied' in params).toBe(false)
  })

  it('ignores a non-string smart_reason', () => {
    const params = canonicalPromptParams('approval', 's1', { ...base, smart_reason: 42 })

    expect('smart_reason' in params).toBe(false)
  })
})
