import type { BillingBlock } from '@hermes/shared'
import { beforeEach, expect, test, vi } from 'vitest'

vi.mock('@/lib/external-link', () => ({ openExternalLink: vi.fn() }))

import { openExternalLink } from '@/lib/external-link'

import {
  $billingBlock,
  $billingSettingsRequest,
  billingCtaLabel,
  clearBillingBlock,
  requestBillingSettings,
  runBillingRecovery,
  setBillingBlock
} from './billing-block'

function makeBlock(overrides: Partial<BillingBlock> = {}): BillingBlock {
  return {
    billing_url: 'https://platform.openai.com/settings/organization/billing',
    is_nous: false,
    message: 'You are out of credits.',
    model: 'gpt-5',
    provider: 'openai',
    provider_label: 'OpenAI',
    ...overrides
  }
}

beforeEach(() => {
  $billingBlock.set(null)
  $billingSettingsRequest.set(0)
  vi.clearAllMocks()
})

test('setBillingBlock stores the block against its session', () => {
  setBillingBlock('s1', makeBlock())
  expect($billingBlock.get()?.sessionId).toBe('s1')
  expect($billingBlock.get()?.block.provider).toBe('openai')
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

test('runBillingRecovery routes Nous to in-app Settings, never an external link', () => {
  runBillingRecovery(makeBlock({ is_nous: true, provider: 'nous', provider_label: 'Nous Portal' }))
  expect($billingSettingsRequest.get()).toBe(1)
  expect(openExternalLink).not.toHaveBeenCalled()
})

test('runBillingRecovery deep-links a third-party provider to its billing page', () => {
  const block = makeBlock({ billing_url: 'https://openrouter.ai/settings/credits', provider: 'openrouter' })
  runBillingRecovery(block)
  expect(openExternalLink).toHaveBeenCalledWith('https://openrouter.ai/settings/credits')
  expect($billingSettingsRequest.get()).toBe(0)
})

test('runBillingRecovery falls back to in-app settings when a provider has no URL', () => {
  runBillingRecovery(makeBlock({ billing_url: null, provider: 'custom' }))
  expect(openExternalLink).not.toHaveBeenCalled()
  expect($billingSettingsRequest.get()).toBe(1)
})

test('requestBillingSettings increments the intent counter', () => {
  requestBillingSettings()
  requestBillingSettings()
  expect($billingSettingsRequest.get()).toBe(2)
})

test('billingCtaLabel picks the right verb per route', () => {
  const copy = { addCredits: 'Add credits', openBilling: 'Open billing' }
  expect(billingCtaLabel(makeBlock({ is_nous: true }), copy)).toBe('Open billing')
  expect(billingCtaLabel(makeBlock({ is_nous: false }), copy)).toBe('Add credits')
})
