import type { BillingBlock } from '@rabbit/shared'
import { beforeEach, expect, test, vi } from 'vitest'

vi.mock('@/lib/external-link', () => ({ openExternalLink: vi.fn() }))

import { openExternalLink } from '@/lib/external-link'

import { $billingBlock, clearBillingBlock, runBillingRecovery, setBillingBlock } from './billing-block'

function makeBlock(overrides: Partial<BillingBlock> = {}): BillingBlock {
  return {
    billing_url: 'https://platform.openai.com/settings/organization/billing',
    message: 'You are out of credits.',
    model: 'gpt-5',
    provider: 'openai',
    provider_label: 'OpenAI',
    ...overrides
  }
}

beforeEach(() => {
  $billingBlock.set(null)
  vi.clearAllMocks()
})

test('clearBillingBlock scoped to a session leaves a different session block intact', () => {
  setBillingBlock('s1', makeBlock())
  clearBillingBlock('s2')
  expect($billingBlock.get()).not.toBeNull()

  clearBillingBlock('s1')
  expect($billingBlock.get()).toBeNull()
})

test('clearBillingBlock with no arg clears any active block', () => {
  setBillingBlock('s1', makeBlock())
  clearBillingBlock()
  expect($billingBlock.get()).toBeNull()
})

test('runBillingRecovery deep-links the provider to its billing page', () => {
  const block = makeBlock({ billing_url: 'https://openrouter.ai/settings/credits', provider: 'openrouter' })
  runBillingRecovery(block)
  expect(openExternalLink).toHaveBeenCalledWith('https://openrouter.ai/settings/credits')
})

test('runBillingRecovery is a no-op when a provider has no billing URL', () => {
  runBillingRecovery(makeBlock({ billing_url: null, provider: 'custom' }))
  expect(openExternalLink).not.toHaveBeenCalled()
})
