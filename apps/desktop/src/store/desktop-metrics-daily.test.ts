import { JsonRpcGatewayError } from '@hermes/shared'
import { expect, it, vi } from 'vitest'

import { sendDesktopDaily } from './desktop-metrics-daily'
import type { SharedMetricsRequester } from './shared-metrics'

const params = {
  day: '2026-11-03',
  actions: [
    { action: 'ia.view.chat', via: 'state' as const, count: 3 },
    { action: 'ia.rail.plugin', via: 'click' as const, count: 2 },
    { action: 'session.new', via: 'shortcut' as const, count: 1 }
  ]
}

it('preserves legacy daily metrics when an older gateway rejects the new observation source', async () => {
  const request = vi
    .fn()
    .mockRejectedValueOnce(new JsonRpcGatewayError('invalid params', { code: 4000 }))
    .mockResolvedValue({ recorded: true })

  await expect(sendDesktopDaily(request as SharedMetricsRequester, params, () => true)).resolves.toEqual({
    recorded: true
  })
  expect(request).toHaveBeenCalledTimes(2)
  expect(request.mock.calls[1][1].actions).toEqual([params.actions[2]])
})
it('does not retry network failures or turn an unknown write outcome into a second report', async () => {
  const request = vi.fn().mockRejectedValue(new Error('offline'))
  await expect(sendDesktopDaily(request as SharedMetricsRequester, params, () => true)).rejects.toThrow('offline')
  expect(request).toHaveBeenCalledOnce()
})

it('does not start a compatibility retry after consent or scope is invalidated', async () => {
  const request = vi.fn().mockRejectedValue(new JsonRpcGatewayError('invalid params', { code: 4000 }))
  await expect(sendDesktopDaily(request as SharedMetricsRequester, params, () => false)).resolves.toBeUndefined()
  expect(request).toHaveBeenCalledOnce()
})
