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
  const malformed = ['{broken', 'null', 'false', '0', '[]', '{}']
  const nonText: (null | boolean | number | object)[] = [null, false, 0, [], {}, { constructor: 'String' }]

  for (const field of ['storedId', 'runtimeId', 'task', 'brief']) {
    malformed.push(...nonText.map(value => JSON.stringify({ ...receipt, [field]: value })))
  }

  malformed.push(
    ...nonText
      .filter(value => value !== null)
      .map(connectionId => JSON.stringify({ ...receipt, owner: { ...receipt.owner, connectionId } }))
  )
  malformed.push(JSON.stringify({ ...receipt, storedId: '' }))

  for (const raw of malformed) {
    localStorage.setItem(key, raw)
    expect(() => readHandoffReceipt(key)).toThrow('could not be read')
  }

  saveHandoffReceipt(key, {
    ...receipt,
    runtimeId: '',
    task: '',
    brief: '',
    owner: { connectionId: null, profile: 'default' }
  })
  expect(readHandoffReceipt(key)).toEqual({
    ...receipt,
    runtimeId: '',
    task: '',
    brief: '',
    owner: { connectionId: null, profile: 'default' }
  })
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
