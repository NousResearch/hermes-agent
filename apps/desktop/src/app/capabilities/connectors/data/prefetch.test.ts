import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import type { ConnectorCardModel, ConnectorState } from '../types'

import { prefetchConnectorTools } from './prefetch'

function hostedCard(slug: string, state: ConnectorState): ConnectorCardModel {
  return {
    inCatalog: true,
    name: slug,
    residency: 'hosted',
    slug,
    state,
    stateWord: 'available',
    ways: { hosted: { accounts: [], connected: state === 'connected', retiredAccounts: [], state }, local: null }
  }
}

function localCard(slug: string): ConnectorCardModel {
  return {
    inCatalog: false,
    name: slug,
    residency: 'local',
    slug,
    state: 'connected',
    stateWord: 'serverOn',
    ways: { hosted: null, local: { state: 'connected' } }
  }
}

describe('prefetchConnectorTools', () => {
  afterEach(() => vi.restoreAllMocks())

  it('reads the tool list only for a card whose hosted account is connected', () => {
    const prefetch = vi.spyOn(queryClient, 'prefetchQuery').mockResolvedValue(undefined)
    const states: ConnectorState[] = ['available', 'broken', 'connected', 'connecting', 'expired', 'off', 'unknown']

    for (const state of states) {
      prefetchConnectorTools('default', hostedCard(`app-${state}`, state))
    }

    prefetchConnectorTools('default', localCard('local-server'))

    expect(prefetch).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(prefetch.mock.calls[0]?.[0].queryKey)).toContain('app-connected')
  })
})
