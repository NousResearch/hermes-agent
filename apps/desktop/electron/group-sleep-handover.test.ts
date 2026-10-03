import { expect, test, vi } from 'vitest'

import { handOverGroupsBeforeSleep } from './group-sleep-handover'

function gateway(methods: string[], handover: () => Promise<unknown> = async () => ({ moved: ['room'], skipped: [] })) {
  const calls: string[] = []
  const close = vi.fn()

  return { calls, close, client: { close, async request(method: string, params?: Record<string, unknown>) {
    calls.push(method + (params ? JSON.stringify(params) : ''))

    return method === 'groups.capabilities' ? { methods } : handover()
  } } }
}

test('asks the local gateway to hand its groups over, only when it advertises the call, and always closes', async () => {
  const able = gateway(['groups.discard', 'groups.succession.handover_all'])
  expect(await handOverGroupsBeforeSleep(async () => able.client)).toBe('handed_over')
  expect(able.calls).toEqual(['groups.capabilities', 'groups.succession.handover_all{"reason":"sleep"}'])
  expect(able.close).toHaveBeenCalled()

  const older = gateway(['groups.discard'])
  expect(await handOverGroupsBeforeSleep(async () => older.client)).toBe('skipped')
  expect(older.calls).toEqual(['groups.capabilities'])

  expect(await handOverGroupsBeforeSleep(async () => {throw new Error('no warm backend')})).toBe('skipped')
  const failing = gateway(['groups.succession.handover_all'], async () => {throw new Error('handover refused')})
  expect(await handOverGroupsBeforeSleep(async () => failing.client)).toBe('skipped')
  expect(failing.close).toHaveBeenCalled()
})

test('gives up at the deadline without waiting for a slow gateway', async () => {
  vi.useFakeTimers()

  try {
    const slow = gateway(['groups.succession.handover_all'], () => new Promise(() => undefined))
    const result = handOverGroupsBeforeSleep(async () => slow.client, 3000)
    await vi.advanceTimersByTimeAsync(3000)
    expect(await result).toBe('skipped')
    expect(slow.close).toHaveBeenCalled()
  } finally {vi.useRealTimers()}
})
