import { type ConnectionState, JSON_RPC_METHOD_NOT_FOUND, JsonRpcGatewayError } from '@hermes/shared'
import { atom } from 'nanostores'
import { describe, expect, it, vi } from 'vitest'

import {
  type CatalogLookupReply,
  type CatalogRequest,
  lookupPluginCatalogEntry,
  sendWhenGatewayOpen
} from './plugin-catalog'

const SHA = 'a'.repeat(40)

const replying =
  (reply: CatalogLookupReply | null | undefined): CatalogRequest =>
  async () =>
    reply

const noFetch = vi.fn<typeof fetch>(async () => {
  throw new Error('the renderer must not fetch the docs host when the backend answers')
})

describe('lookupPluginCatalogEntry', () => {
  it('asks the backend for the entry, not the docs host', async () => {
    const found = { name: 'aihubmix', repo: 'https://github.com/AIhubmix/mono', sha: SHA, subdir: 'aihubmix' }
    const request = vi.fn<CatalogRequest>(async () => ({ status: 'found', entry: found }))

    await expect(lookupPluginCatalogEntry(' aihubmix ', { fetchImpl: noFetch, request })).resolves.toEqual({
      ok: true,
      entry: found
    })
    expect(request.mock.calls).toEqual([['plugins.catalog.lookup', { name: 'aihubmix' }]])
    expect(noFetch).not.toHaveBeenCalled()
  })

  it.each([
    [{ status: 'unknown' }, 'unknown'],
    [{ status: 'invalid_name' }, 'invalid_name'],
    [{ status: 'unavailable' }, 'unavailable'],
    [{ status: 'found', entry: { name: 'x' } }, 'unavailable'],
    [{ status: 'something-new' }, 'unavailable'],
    [null, 'unavailable'],
    [undefined, 'unavailable']
  ] as const)('backend %j → %s, with no fetch', async (reply, error) => {
    await expect(
      lookupPluginCatalogEntry('weather', { fetchImpl: noFetch, request: replying(reply) })
    ).resolves.toEqual({
      ok: false,
      error
    })
  })

  it('refuses a non-catalog name before asking anyone', async () => {
    const request = vi.fn<CatalogRequest>(async () => ({ status: 'found' }))

    await expect(lookupPluginCatalogEntry('../etc', { fetchImpl: noFetch, request })).resolves.toEqual({
      ok: false,
      error: 'invalid_name'
    })
    expect(request).not.toHaveBeenCalled()
  })

  it('falls back to the direct feed only when the backend predates the lookup', async () => {
    const feed = [{ name: 'weather', repo: 'https://github.com/x/weather', sha: SHA, subdir: '' }]
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response(JSON.stringify(feed)))

    const request: CatalogRequest = async () => {
      throw new JsonRpcGatewayError('method not found: plugins.catalog.lookup', { code: JSON_RPC_METHOD_NOT_FOUND })
    }

    await expect(lookupPluginCatalogEntry('weather', { fetchImpl, request })).resolves.toMatchObject({
      ok: true,
      entry: { name: 'weather', repo: 'https://github.com/x/weather' }
    })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('reads an unreachable backend as unavailable, without fetching', async () => {
    const request: CatalogRequest = async () => {
      throw new Error('Hermes gateway unavailable for profile "default"')
    }

    await expect(lookupPluginCatalogEntry('weather', { fetchImpl: noFetch, request })).resolves.toEqual({
      ok: false,
      error: 'unavailable'
    })
    expect(noFetch).not.toHaveBeenCalled()
  })
})

describe('sendWhenGatewayOpen', () => {
  it('holds a lookup that arrives before the backend connects, then sends it once the gateway opens', async () => {
    const state = atom<ConnectionState>('connecting')
    const send = vi.fn(async () => 'sent')
    const pending = sendWhenGatewayOpen(send, { state, timeoutMs: 5_000 })

    await Promise.resolve()
    expect(send).not.toHaveBeenCalled()

    state.set('open')

    await expect(pending).resolves.toBe('sent')
    expect(send).toHaveBeenCalledTimes(1)
  })

  it('rejects when the backend never connects, so a startup link still ends in a visible error', async () => {
    const send = vi.fn(async () => 'sent')

    await expect(
      sendWhenGatewayOpen(send, { state: atom<ConnectionState>('connecting'), timeoutMs: 10 })
    ).rejects.toThrow('Hermes backend did not connect')
    expect(send).not.toHaveBeenCalled()
  })
})
