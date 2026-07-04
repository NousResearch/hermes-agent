import { describe, expect, it } from 'vitest'

import { applyUltraModePrefix } from './skill-mode-prefix'

describe('applyUltraModePrefix', () => {
  it('prefixes normal prompts with active ULW and ULR modes in trigger order', () => {
    expect(applyUltraModePrefix('build and verify this', { ultrawork: true, ultraresearch: true })).toBe(
      'ulw ultraresearch build and verify this'
    )
  })

  it('prefixes Agent Fleet before other active skill modes', () => {
    expect(
      applyUltraModePrefix('build and verify this', {
        agentFleet: true,
        ultraresearch: true,
        ultrawork: true
      })
    ).toBe('agent fleet ulw ultraresearch build and verify this')
  })

  it('does not duplicate an explicit mode trigger already typed by the user', () => {
    expect(applyUltraModePrefix('ulr compare these papers', { agentFleet: true, ultrawork: true, ultraresearch: true })).toBe(
      'ulr compare these papers'
    )
  })

  it('leaves prompts unchanged when both toggles are off', () => {
    expect(applyUltraModePrefix('plain question', { ultrawork: false, ultraresearch: false })).toBe('plain question')
  })
})
