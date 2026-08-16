import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { $pinnedSessionIds, activatePinnedSessionConnection } from './session-pins'

const connection = (overrides: Partial<HermesConnection>): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...overrides }) as HermesConnection

beforeEach(() => {
  window.localStorage.clear()
  activatePinnedSessionConnection(null)
})

afterEach(() => {
  activatePinnedSessionConnection(null)
  window.localStorage.clear()
})

describe('connection-scoped session pins', () => {
  it('keeps local and remote gateway pins isolated across connection switches', () => {
    const local = connection({ mode: 'local', profile: 'default' })
    const remote = connection({ baseUrl: 'https://remote.example.test/', mode: 'remote', profile: 'work' })

    activatePinnedSessionConnection(local)
    $pinnedSessionIds.set(['local-pin'])

    activatePinnedSessionConnection(remote)
    expect($pinnedSessionIds.get()).toEqual([])
    $pinnedSessionIds.set(['remote-pin'])

    activatePinnedSessionConnection(local)
    expect($pinnedSessionIds.get()).toEqual(['local-pin'])

    activatePinnedSessionConnection(remote)
    expect($pinnedSessionIds.get()).toEqual(['remote-pin'])
  })

  it('keeps pins isolated between remote gateway targets', () => {
    const gatewayA = connection({ baseUrl: 'https://gateway-a.example.test', mode: 'remote', remoteKind: 'url' })
    const gatewayB = connection({ baseUrl: 'https://gateway-b.example.test', mode: 'remote', remoteKind: 'url' })

    activatePinnedSessionConnection(gatewayA)
    $pinnedSessionIds.set(['gateway-a-pin'])

    activatePinnedSessionConnection(gatewayB)
    expect($pinnedSessionIds.get()).toEqual([])
    $pinnedSessionIds.set(['gateway-b-pin'])

    activatePinnedSessionConnection(gatewayA)
    expect($pinnedSessionIds.get()).toEqual(['gateway-a-pin'])

    activatePinnedSessionConnection(gatewayB)
    expect($pinnedSessionIds.get()).toEqual(['gateway-b-pin'])
  })

  it('uses stable SSH identity instead of an ephemeral forwarded port', () => {
    activatePinnedSessionConnection(
      connection({
        baseUrl: 'http://127.0.0.1:41001',
        mode: 'remote',
        profile: 'work',
        remoteHost: 'operator@remote-box',
        remoteIdentity: 'operator@remote-box',
        remoteKind: 'ssh'
      })
    )
    $pinnedSessionIds.set(['ssh-pin'])

    activatePinnedSessionConnection(
      connection({
        baseUrl: 'http://127.0.0.1:52002',
        mode: 'remote',
        profile: 'work',
        remoteHost: 'operator@remote-box',
        remoteIdentity: 'operator@remote-box',
        remoteKind: 'ssh'
      })
    )

    expect($pinnedSessionIds.get()).toEqual(['ssh-pin'])
  })
})
