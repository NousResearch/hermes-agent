import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getApiRequestConnection: vi.fn(() => 'secondary-gateway'),
  hermesApi: vi.fn(async () => ({ ok: true, profile: 'default', stamped: 0 }))
}))

vi.mock('@/api/client', () => api)
vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/connection-registry-state', () => ({
  $connectionsRegistry: { get: () => ({ connections: [{ id: 'local' }, { id: 'secondary-gateway' }] }) },
  hasRegistryTopology: () => true
}))

import { maybeBackfillLegacySessionOwners, resetLegacyOwnerBackfillAttempts } from './legacy-session-owner-backfill'
import { resolveLegacyOwnerBackfillScope } from './session-owner-stamp'

beforeEach(() => {
  vi.clearAllMocks()
  resetLegacyOwnerBackfillAttempts()
})

describe('resolveLegacyOwnerBackfillScope (#94724 single-match owner backfill)', () => {
  it('targets the serving registered connection (the backend that serves a page owns its rows)', () => {
    const scope = resolveLegacyOwnerBackfillScope({
      hasRegistryTopology: true,
      registryConnectionIds: ['local', 'gw-b', 'gw-c'],
      servingConnectionId: 'gw-b'
    })

    expect(scope).toEqual({ connectionId: 'gw-b', profile: null })
  })

  it('targets the primary store when the primary pool is serving', () => {
    // The primary's own per-profile store is a single known owner even with
    // several registered connections — its rows can live nowhere else.
    const scope = resolveLegacyOwnerBackfillScope({
      hasRegistryTopology: true,
      registryConnectionIds: ['gw-b', 'gw-c'],
      servingConnectionId: null
    })

    expect(scope).toEqual({ connectionId: null, profile: null })
  })

  it("treats the explicit 'local' source as the primary store", () => {
    expect(
      resolveLegacyOwnerBackfillScope({
        hasRegistryTopology: true,
        registryConnectionIds: ['gw-b'],
        servingConnectionId: 'local'
      })
    ).toEqual({ connectionId: null, profile: null })
  })

  it('fails closed when the serving source is unknown and several backends could own the store', () => {
    // Multi-candidate: never guess. The rows stay NULL and the read-only
    // stored-transcript path keeps their history reachable.
    expect(
      resolveLegacyOwnerBackfillScope({
        hasRegistryTopology: true,
        registryConnectionIds: ['gw-b', 'gw-c'],
        servingConnectionId: undefined
      })
    ).toBeNull()
  })

  it('resolves the single registered backend when the serving source is unknown but only one candidate exists', () => {
    expect(
      resolveLegacyOwnerBackfillScope({
        hasRegistryTopology: true,
        registryConnectionIds: ['local', 'gw-b'],
        servingConnectionId: undefined
      })
    ).toEqual({ connectionId: 'gw-b', profile: null })
  })

  it('fails closed when the serving connection is not in the registry', () => {
    expect(
      resolveLegacyOwnerBackfillScope({
        hasRegistryTopology: true,
        registryConnectionIds: ['gw-b'],
        servingConnectionId: 'gw-unregistered'
      })
    ).toBeNull()
  })

  it('does nothing without registry topology (legacy single-backend installs are unaffected)', () => {
    expect(
      resolveLegacyOwnerBackfillScope({
        hasRegistryTopology: false,
        registryConnectionIds: [],
        servingConnectionId: null
      })
    ).toBeNull()
  })
})

it('pins an explicit local primary backfill instead of inheriting the ambient remote', () => {
  maybeBackfillLegacySessionOwners('local')

  expect(api.hermesApi).toHaveBeenCalledWith(
    expect.objectContaining({ connectionId: 'local', method: 'POST', path: '/api/sessions/owner-backfill' })
  )
  expect(api.getApiRequestConnection).not.toHaveBeenCalled()
})
