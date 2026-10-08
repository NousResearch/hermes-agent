import { expect, test, vi } from 'vitest'
import { deviceCommand, revokeRemoteAccess } from './remote-control-actions'

test('enrollment uses one stock plugin command; connection errors are not replayed', async () => {
  const request = vi.fn().mockRejectedValue(new Error('connection closed'))
  await expect(deviceCommand(request, 'live-session', 'target')).rejects.toThrow('connection closed')
  expect(request).toHaveBeenCalledExactlyOnceWith(
    'command.dispatch',
    {
      name: 'desktop-devices',
      arg: 'target',
      session_id: 'live-session'
    },
    120000
  )
})

test('offline revoke clears native credentials before gateway cleanup and stays revoked', async () => {
  const order: string[] = []
  const native = {
    remote: vi.fn(async () => {
      order.push('native')
      return { revoked: true }
    })
  }
  const cleanup = async () => {
    order.push('gateway')
    throw new Error('offline')
  }
  expect(await revokeRemoteAccess(native, '', cleanup)).toEqual({ gatewayConfirmed: false })
  expect(order).toEqual(['native', 'gateway'])
  expect(native.remote).toHaveBeenCalledExactlyOnceWith({ sessionId: '', action: 'revoke', arguments: {} })
})

test('native revoke failure never claims success or substitutes a gateway-only revoke', async () => {
  const cleanup = vi.fn()
  const native = { remote: vi.fn().mockRejectedValue(new Error('window closed')) }
  await expect(revokeRemoteAccess(native, '', cleanup)).rejects.toThrow('window closed')
  expect(cleanup).not.toHaveBeenCalled()
})

test('plugin refusal and missing plugin are displayed without invoking the model', async () => {
  await expect(
    deviceCommand(async () => ({ type: 'plugin', output: '{"error":"refused"}' }), 'live', 'target')
  ).rejects.toThrow('refused')
  await expect(deviceCommand(async () => ({ type: 'skill' }), 'live', 'target')).rejects.toThrow('unavailable')
})
