import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@/components/onboarding-chat/setup-profile', async () => ({
  $setupHandoff: (await import('nanostores')).atom(null)
}))

import type { HandoffReceipt } from './handoff-leg'
import { handoffReceiptKey, readHandoffReceipt, saveHandoffReceipt } from './handoff-receipt'

const receipt: HandoffReceipt = {
  owner: { connectionId: 'source-a', profile: 'default' },
  runtimeId: 'runtime',
  storedId: 'stored',
  task: 'Tracker',
  brief: 'Build tracker',
  plan: 'build',
  status: 'created'
}

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

it('rejects corrupt receipts instead of treating them as permission to create again', () => {
  const key = handoffReceiptKey('source-a', 'guide')
  expect(key).not.toBe(handoffReceiptKey('source-b', 'guide'))
  localStorage.setItem(key, '{broken')
  expect(() => readHandoffReceipt(key)).toThrow('could not be read')
})

it('retains the identity in memory if disk persistence fails so retry cannot recreate', () => {
  const key = handoffReceiptKey('source-a', 'quota-test')
  vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
    throw new Error('quota')
  })
  expect(() => saveHandoffReceipt(key, receipt)).toThrow('Could not save')
  expect(readHandoffReceipt(key)).toEqual(receipt)
  vi.restoreAllMocks()
  saveHandoffReceipt(key, receipt)
  expect(JSON.parse(localStorage.getItem(key)!)).toEqual(receipt)
})
