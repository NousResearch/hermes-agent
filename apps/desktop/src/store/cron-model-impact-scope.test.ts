import { describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { getCronModelImpactScope, syncCronModelImpactConnection } from './cron-model-impact-scope'

function connection(baseUrl: string, wsUrl: string, overrides: Partial<HermesConnection> = {}): HermesConnection {
  return {
    baseUrl,
    isFullscreen: false,
    mode: 'remote',
    nativeOverlayWidth: 0,
    token: 'secret-not-part-of-identity',
    wsUrl,
    logs: [],
    windowButtonPosition: null,
    ...overrides
  }
}

describe('cron model impact backend identity', () => {
  it('survives disconnects and reminted websocket tickets but invalidates for another backend', () => {
    syncCronModelImpactConnection(connection('https://one.example', 'wss://one.example?ticket=first'))
    const first = getCronModelImpactScope()

    syncCronModelImpactConnection(null)
    expect(getCronModelImpactScope()).toEqual(first)

    syncCronModelImpactConnection(connection('https://one.example', 'wss://one.example?ticket=second'))
    expect(getCronModelImpactScope()).toEqual(first)

    syncCronModelImpactConnection(connection('https://two.example', 'wss://two.example?ticket=third'))
    expect(getCronModelImpactScope().generation).toBe(first.generation + 1)
  })

  it('treats reminted SSH tunnel ports as the same remote backend', () => {
    const ssh = {
      remoteKind: 'ssh' as const,
      remoteIdentity: 'operator@remote-box',
      remoteHost: 'remote-box'
    }

    syncCronModelImpactConnection(connection('http://127.0.0.1:41001', 'ws://127.0.0.1:41001', ssh))
    const first = getCronModelImpactScope()

    syncCronModelImpactConnection(connection('http://127.0.0.1:52002', 'ws://127.0.0.1:52002', ssh))

    expect(getCronModelImpactScope()).toEqual(first)
  })
})
