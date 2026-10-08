import type { ConnectorsListResult } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $freeTierStatus } from '@/store/free-tier'
import type { FreeTierStatus } from '@/types/hermes'

import { type FactSources, watchFacts } from './facts'
import type { Facts } from './flow'

// Free tier on, no free account: what a signed-in Nous account and a pending free account both answer.
const NO_GUEST: FreeTierStatus = {
  available: false,
  enabled: true,
  has_guest: false,
  label: 'Nous · free tier',
  model: 'free-model',
  notice_pending: false
}

const LIST: ConnectorsListResult = {
  available: true,
  connectors: [
    {
      connected: false,
      connection_status: null,
      connector: 'github',
      enabled: true,
      gateway_disabled_tools: [],
      status_reason: null
    }
  ]
}

function sources(list: ConnectorsListResult): FactSources {
  return {
    listConnectors: async () => list,
    localFit: async () => null,
    request: vi
      .fn()
      .mockImplementation(async (method: string) =>
        method === 'free_tier.status' ? NO_GUEST : Promise.reject(new Error(`${method} is not read here`))
      )
  }
}

async function connectorsAfterOpen(list: ConnectorsListResult): Promise<Facts['connectors'] | undefined> {
  let connectors: Facts['connectors'] | undefined
  const stop = watchFacts(sources(list), facts => (connectors = facts.connectors ?? connectors))

  await new Promise(resolve => setTimeout(resolve, 0))
  stop()

  return connectors
}

afterEach(() => $freeTierStatus.set(null))

describe('watchFacts connector list', () => {
  it('lists connectors for a signed-in account that has no free account', async () => {
    expect(await connectorsAfterOpen(LIST)).toEqual({ rows: [{ id: 'github', label: 'GitHub' }], status: 'ready' })
  })

  it('keeps loading while the free account is still being made', async () => {
    expect(await connectorsAfterOpen({ available: false, connectors: [] })).toEqual({ status: 'loading' })
  })
})
