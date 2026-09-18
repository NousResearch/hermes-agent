import { webcrypto } from 'node:crypto'

import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { observeDownloads } from './canonical-download-test-utils'
import { listCanonicalFiles, saveCanonicalFile } from './canonical-files-client'
import { deferred, FILE_BINDING, fileItem, filePage, parsedFilePage } from './canonical-files-test-fixtures'

// Real Files client -> canonicalGroupRequest -> SDK descriptor overload -> registry.
// Only the native descriptor and socket are synthetic; replies are decoded RPC bodies.
const sockets = vi.hoisted(() => ({ created: vi.fn() }))
vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    constructor() { sockets.created() }
    connectionState = 'closed'
    onEvent = () => () => undefined
    onState = () => () => undefined
    close = () => undefined
    connect = async () => { throw new Error('Unexpected secondary dial') }
  }
}))

const {
  closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForAgent, openGatewayForAgent,
  requestGatewayForAgent, retainGatewayForAgent, setPrimaryGateway, setPrimaryGatewayConnectionId
} = await import('@/store/gateway')

const originalDesktop = window.hermesDesktop
let observed: ReturnType<typeof observeDownloads>

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
  sockets.created.mockClear()
  vi.stubGlobal('crypto', webcrypto)
  observed = observeDownloads()
})
afterEach(() => {
  closeSecondaryGateways()
  setPrimaryGateway(null)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  window.hermesDesktop = originalDesktop
})

it.each(['success', 'failure', 'isolated'] as const)(
  'never forwards private Files selections after a held %s probe loses its owner', async outcome => {
    const held = deferred<unknown>()

    const receipt = {
      ...fileItem(), room_id: FILE_BINDING.roomId, authority: filePage().authority,
      data_base64: 'QQ==',
      sha256: Buffer.from(await webcrypto.subtle.digest('SHA-256', new Uint8Array([65]))).toString('hex')
    }

    const reply = async (method: string) => method === 'groups.attachment.list' ? filePage() : receipt
    const a = { connectionState: 'open', request: vi.fn(reply) }
    const b = { connectionState: 'open', request: vi.fn(reply) }
    const descriptor = vi.fn().mockReturnValue(held.promise)
    window.hermesDesktop = { getConnectionFor: descriptor } as unknown as typeof window.hermesDesktop
    setPrimaryGateway(a as never)
    setPrimaryGatewayConnectionId(FILE_BINDING.connectionId)
    const page = parsedFilePage()
    const oldList = listCanonicalFiles(FILE_BINDING, { query: 'private query' }, page.authority!, undefined, () => true).catch(e => e)
    const oldSave = saveCanonicalFile(FILE_BINDING, page.authority!, page.items[0], undefined, () => true).catch(e => e)
    expect(descriptor).toHaveBeenCalledTimes(2)
    setPrimaryGateway(b as never)
    setPrimaryGatewayConnectionId('gateway-b')

    if (outcome === 'failure') { held.reject(new Error('native probe failed')) } else {
      held.resolve({ sharedRemote: outcome === 'success' })
    }

    const results = await Promise.all([oldList, oldSave])
    expect(b.request).not.toHaveBeenCalled()
    expect(a.request).not.toHaveBeenCalled()
    expect(sockets.created).not.toHaveBeenCalled()
    expect(results.every(result => result instanceof Error)).toBe(true)
    expect(observed.create).not.toHaveBeenCalled()
    expect(observed.click).not.toHaveBeenCalled()

    descriptor.mockResolvedValue({ sharedRemote: true })
    const binding = { ...FILE_BINDING, connectionId: 'gateway-b' }
    const fresh = await listCanonicalFiles(binding, {}, page.authority!, undefined, () => true)
    await saveCanonicalFile(binding, fresh.authority!, fresh.items[0], undefined, () => true)
    expect(b.request).toHaveBeenCalledTimes(2)
    expect(b.request.mock.calls[1][0]).toBe('groups.attachment.download')
    expect(observed.click).toHaveBeenCalledOnce()
  }
)

it.each(['profile', 'connection', 'gateway-aba', 'profile-aba', 'connection-aba', 'abort'] as const)(
  'retires all shared-route probes on %s without a fallback', async change => {
    const held = deferred<unknown>()
    const a = { connectionState: 'open', request: vi.fn(async () => ({})) }
    const descriptor = vi.fn().mockReturnValue(held.promise)
    window.hermesDesktop = { getConnectionFor: descriptor } as unknown as typeof window.hermesDesktop
    setPrimaryGateway(a as never)
    setPrimaryGatewayConnectionId('gateway-a')
    const controller = new AbortController()
    const request = requestGatewayForAgent('gateway-a', 'reviewer', 'groups.attachment.list', { query: 'private' }, undefined, controller.signal).catch(e => e)

    const probes = change === 'abort' ? [] : [
      openGatewayForAgent('gateway-a', 'reviewer').catch(e => e),
      ensureGatewayForAgent('gateway-a', 'reviewer').catch(e => e),
      retainGatewayForAgent('gateway-a', 'reviewer').catch(e => e)
    ]

    if (change === 'abort') { controller.abort() }
    else if (change === 'gateway-aba') {
      setPrimaryGateway({ connectionState: 'open' } as never)
      setPrimaryGateway(a as never)
      setPrimaryGatewayConnectionId('gateway-a')
    } else if (change.startsWith('profile')) {
      setPrimaryGateway(a as never, 'other')

      if (change.endsWith('aba')) { setPrimaryGateway(a as never, 'default') }
    } else {
      setPrimaryGatewayConnectionId('gateway-b')

      if (change.endsWith('aba')) { setPrimaryGatewayConnectionId('gateway-a') }
    }

    held.resolve({ sharedRemote: true })
    const results = await Promise.all([request, ...probes])
    expect(a.request).not.toHaveBeenCalled()
    expect(sockets.created).not.toHaveBeenCalled()
    expect(results.every(result => result instanceof Error)).toBe(true)
  }
)
