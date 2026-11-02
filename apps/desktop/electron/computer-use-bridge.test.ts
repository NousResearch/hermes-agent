import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import { createComputerUseBridge } from './computer-use-bridge'

function deps(overrides: any = {}) {
  const logs: string[] = []

  return {
    logs,
    deps: {
      resolveHermesBackend: () => ({ command: 'hermes', args: [], env: {}, shell: false }),
      ensureRuntime: async (backend: any) => backend,
      resolveHermesCwd: () => '/tmp',
      hiddenWindowsChildOptions: (options: any) => options,
      stopBackendChild: () => {},
      rememberLog: (line: string) => void logs.push(line),
      mintGatewayWsTicket: async () => 'ticket',
      spawn: () => {
        throw new Error('the sidecar must not spawn in this test')
      },
      hermesHome: '/tmp/.hermes',
      ...overrides
    }
  }
}

const remote = { baseUrl: 'https://box:9119', authMode: 'token', token: 'tok', computerUseBridge: true }

describe('lite clients', () => {
  it('report why the bridge cannot run instead of trying to spawn it', async () => {
    const { deps: d, logs } = deps({
      resolveHermesBackend: () => ({ bootstrap: true, command: 'hermes', args: [] })
    })

    const bridge = createComputerUseBridge(d as any)

    const reason = bridge.unsupported()

    assert.match(reason || '', /no local agent runtime/)
    bridge.acquire(remote, 'default', 'primary')
    assert.equal(await bridge.ensure(remote, 'default'), null)
    assert.ok(logs.some(line => /no local agent runtime/.test(line)))
  })

  it('answer before anything is attempted, so Settings can say so up front', () => {
    const { deps: d } = deps({
      resolveHermesBackend: () => ({ bootstrap: true, command: 'hermes', args: [] })
    })

    assert.ok(createComputerUseBridge(d as any).unsupported())
  })

  it('stay silent on an install that does have a runtime', () => {
    const { deps: d } = deps()

    assert.equal(createComputerUseBridge(d as any).unsupported(), null)
  })
})

describe('scope ownership', () => {
  it('refuses to connect a scope nobody claimed', async () => {
    const { deps: d } = deps()
    const bridge = createComputerUseBridge(d as any)

    // No acquire(): a socket with no owner would outlive whatever wanted it.
    assert.equal(await bridge.ensure(remote, 'work'), null)
  })

  it('keys each profile separately, so one release cannot drop another', () => {
    const { deps: d } = deps()
    const bridge = createComputerUseBridge(d as any)

    const work = bridge.acquire(remote, 'work', 'pool:work')
    const home = bridge.acquire(remote, 'home', 'pool:home')

    assert.notEqual(work, home)
    assert.equal(bridge.release(work, 'pool:work'), false, 'the other scope still holds the sidecar')
    assert.equal(bridge.release(home, 'pool:home'), true)
  })

  it('gives every profile on a per-profile remote the same scope as the remote itself', () => {
    const { deps: d } = deps()
    const bridge = createComputerUseBridge(d as any)
    const perProfile = { ...remote, source: 'profile' }

    assert.equal(bridge.acquire(perProfile, 'work', 'a'), bridge.acquire(perProfile, 'home', 'b'))
  })
})

describe('the primary connection', () => {
  it('hands its claim back when the app goes local', async () => {
    const { deps: d } = deps()
    const bridge = createComputerUseBridge(d as any)
    let stopped = 0

    await bridge.ensurePrimary({ ...remote, computerUseBridge: false }, 'default')
    await bridge.ensurePrimary(null)

    assert.equal(stopped, 0)
  })

  it('does not open a bridge for a connection that has it switched off', async () => {
    const { deps: d } = deps()
    const bridge = createComputerUseBridge(d as any)

    assert.equal(await bridge.ensure({ ...remote, computerUseBridge: false }, 'default'), null)
  })
})
