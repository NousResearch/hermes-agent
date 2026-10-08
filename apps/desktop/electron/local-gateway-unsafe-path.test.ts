import { expect, test, vi } from 'vitest'

import { redialLocalGateway } from './local-gateway'

test('an unsafe control path surfaces without forgetting and re-ensuring the same owner', async () => {
  const failure = new Error('Unsafe gateway control path')
  const ensure = vi.fn(async () => 'owner')
  const forget = vi.fn()
  const use = vi.fn(async () => { throw failure })

  await expect(redialLocalGateway({ ensure, forget, use })).rejects.toBe(failure)
  expect(ensure).toHaveBeenCalledOnce()
  expect(use).toHaveBeenCalledOnce()
  expect(forget).not.toHaveBeenCalled()
})

test.each(['Gateway ticket control socket missing', 'Invalid gateway control pointer', 'Noncanonical gateway profile'])(
  'a stale %s still refreshes its endpoint once', async message => {
    const ensure = vi.fn(async () => 'owner')
    const forget = vi.fn()
    const use = vi.fn().mockRejectedValueOnce(new Error(message)).mockResolvedValue('fresh ticket')

    await expect(redialLocalGateway({ ensure, forget, use })).resolves.toBe('fresh ticket')
    expect(ensure).toHaveBeenCalledTimes(2)
    expect(forget).toHaveBeenCalledOnce()
  }
)
