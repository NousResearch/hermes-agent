import { expect, it } from 'vitest'

import { readKey, writeKey } from '@/lib/storage'

import { handoffReceiptKey, quarantineHandoffReceipt, readHandoffReceipt } from './handoff-receipt'

it('preserves an unreadable receipt and removes it from the retry lookup', () => {
  const key = handoffReceiptKey('receipt-recovery', 'guide')
  const corrupt = '{unreadable receipt'
  writeKey(key, corrupt)

  expect(() => readHandoffReceipt(key)).toThrow()

  quarantineHandoffReceipt(key)

  expect(readKey(`${key}.unreadable`)).toBe(corrupt)
  expect(readHandoffReceipt(key)).toBeNull()
})
