import { afterEach, expect, test, vi } from 'vitest'

const calls = vi.hoisted(() => [] as any[])
vi.mock('@/hermes', () => ({ setApiRequestConnection: vi.fn(), HermesGateway: class {
  id = calls.length; connectionState = 'closed'; subscriptions = new Set<string>(); states = new Set<(state: string) => void>()
  constructor() { calls.push({ instance: this, methods: [], closed: false }) }
  connect = async () => { this.connectionState = 'open' }
  close = () => { this.connectionState = 'closed'; calls[this.id].closed = true; this.states.forEach(fn => fn('closed')) }
  onEvent = () => () => {}
  onState = (fn: (state: string) => void) => { this.states.add(fn);

 return () => this.states.delete(fn) }
  request = async (method: string, params: any) => {
    calls[this.id].methods.push(method)

    if (method === 'session.resume') { this.subscriptions.add(params.session_id);

 return { session_id: params.session_id } }

    if (['durable.upload', 'durable.submit'].includes(method) && !this.subscriptions.has(params.session_id)) {throw new Error('permission_denied: unsubscribed socket')}

    return { admission_id: 'ack', status: 'queued' }
  }
}}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
import { closeSecondaryGateways, configureGatewayRegistry, leaseGatewayForAgent, setPrimaryGateway, setPrimaryGatewayConnection } from './gateway'

afterEach(() => { closeSecondaryGateways(); calls.length = 0 })

function setup() {
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as any, 'default')
  setPrimaryGatewayConnection({ connectionId: 'foreground', mode: 'remote' })
  window.hermesDesktop = { getConnectionFor: async () => ({ connectionId: 'background', profile: 'default', mode: 'remote', authMode: 'token', token: 'synthetic', wsUrl: 'wss://fixture.invalid/ws', baseUrl: 'https://fixture.invalid' }) } as any
}

test('real registry keeps one secondary socket from resume through all uploads/admission, then releases', async () => {
  setup()
  const lease = await leaseGatewayForAgent('background', 'default')

  try {
    await lease.request('session.resume', { session_id: 'sid' })
    await lease.request('durable.upload', { session_id: 'sid' })
    await lease.request('durable.upload', { session_id: 'sid' })
    expect(await lease.request('durable.submit', { session_id: 'sid' })).toMatchObject({ admission_id: 'ack' })
    expect(calls).toHaveLength(1)
    expect(calls[0].closed).toBe(false)
  } finally { lease.release() }

  expect(calls[0].closed).toBe(true)
})

test('socket replacement invalidates an old lease; reconnect must acquire and resume again', async () => {
  setup()
  const lease = await leaseGatewayForAgent('background', 'default')
  await lease.request('session.resume', { session_id: 'sid' })
  calls[0].instance.close()
  calls[0].instance.connectionState = 'open'
  await expect(lease.request('durable.submit', { session_id: 'sid' })).rejects.toThrow('disconnected')
  lease.release()
  const next = await leaseGatewayForAgent('background', 'default')

  try {
    await expect(next.request('durable.submit', { session_id: 'sid' })).rejects.toThrow('unsubscribed')
    await next.request('session.resume', { session_id: 'sid' })
    expect(await next.request('durable.submit', { session_id: 'sid' })).toMatchObject({ admission_id: 'ack' })
  } finally { next.release() }
})
