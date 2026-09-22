import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Attachment, GroupMember } from './types'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

interface TestGateway {
  close: ReturnType<typeof vi.fn>
  connect: ReturnType<typeof vi.fn>
  connectionState: string
  onEvent: ReturnType<typeof vi.fn>
  onState: ReturnType<typeof vi.fn>
  request: ReturnType<typeof vi.fn>
}

const gateways = vi.hoisted(() => ({
  instances: [] as TestGateway[],
  request: vi.fn()
}))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  HermesGateway: class implements TestGateway {
    connectionState = 'closed'
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    onEvent = vi.fn(() => () => undefined)
    onState = vi.fn(() => () => undefined)
    request = vi.fn((method: string, params: Record<string, unknown>) => gateways.request(this, method, params))

    constructor() {
      gateways.instances.push(this)
    }
  }
}))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} = await import('@/store/gateway')

const { $groupChats } = await import('./group-chat')
const { downloadGroupChatAttachment } = await import('./group-attachment-download')

const content = 'Y2xhc3NpYy1jdXJyZW50LWJ5dGVz'
const contentBytes = new TextEncoder().encode('classic-current-bytes')
const digest = '30c313332d0220a37fef2d7ab61b099304914737bc0c3c853c59bf1b20249d4b'
const route = { connectionId: 'source-a', mode: 'remote' as const, profile: 'default', targetProfile: 'default' }

const source: GroupMember = {
  connectionId: route.connectionId,
  connectionKind: route.mode,
  name: route.profile,
  remoteSource: true,
  route,
  sourceScoped: true,
  targetProfile: route.targetProfile
}

const message = { at: 1, from: { kind: 'member' as const, name: 'default' }, images: [], text: '' }
const originalDesktop = window.hermesDesktop
const originalCreateObjectURL = URL.createObjectURL
const originalRevokeObjectURL = URL.revokeObjectURL

function retainedAttachment(): Attachment {
  return {
    kind: 'file',
    mime: 'application/octet-stream',
    name: 'proof.bin',
    size: contentBytes.length,
    classicExport: {
      artifactId: 'rart_0123456789abcdef0123456789abcdef',
      exportId: `ce_${'1'.repeat(64)}`,
      generation: 7,
      group: 'classic-room',
      installation: 'install:root',
      recipients: [{ installation: 'install:root', profile: 'default' }],
      session: 'original-session',
      sha256: digest,
      source
    }
  }
}

function exportResponse() {
  return {
    content_base64: content,
    export_id: `ce_${'1'.repeat(64)}`,
    generation: 7,
    group_id: 'classic-room',
    item: {
      artifact_id: 'rart_0123456789abcdef0123456789abcdef',
      kind: 'file',
      mime: 'application/octet-stream',
      name: 'proof.bin',
      sha256: digest,
      size: contentBytes.length
    }
  }
}

function primaryGateway() {
  return {
    connectionState: 'open',
    request: vi.fn(async () => ({}))
  }
}

function invalidate(kind: 'aba' | 'edit' | 'remove') {
  disposeSecondariesForConnection('source-a', kind === 'remove' ? {} : { redial: true })

  if (kind === 'aba') {
    disposeSecondariesForConnection('source-a', { redial: true })
  }
}

function observeSave() {
  const blobs: Blob[] = []

  const createObjectURL = vi.fn((blob: Blob) => {
    blobs.push(blob)

    return `blob:classic-${blobs.length}`
  })

  const revokeObjectURL = vi.fn()
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

  return { blobs, click, createObjectURL }
}

async function expectFreshSave(saved: ReturnType<typeof observeSave>) {
  gateways.request.mockImplementation(async (_gateway, method: string, params: Record<string, unknown>) => {
    if (method === 'session.resume') {
      expect(params).toEqual({ session_id: 'original-session', profile: 'default', omit_messages: true })

      return { session_id: 'fresh-runtime' }
    }

    if (method === 'session.export.read') {
      expect(params).toMatchObject({ session_id: 'fresh-runtime', generation: 7 })

      return exportResponse()
    }

    throw new Error(`Unexpected RPC ${method}`)
  })

  await downloadGroupChatAttachment('Classic', message, retainedAttachment())

  expect(saved.createObjectURL).toHaveBeenCalledOnce()
  expect(saved.click).toHaveBeenCalledOnce()
  expect(Array.from(new Uint8Array(await saved.blobs[0].arrayBuffer()))).toEqual(Array.from(contentBytes))
}

beforeEach(() => {
  gateways.instances.length = 0
  gateways.request.mockReset()
  configureGatewayRegistry({ onEvent: vi.fn() })
  closeSecondaryGateways()
  setPrimaryGateway(primaryGateway() as never, 'default')
  setPrimaryGatewayConnectionId('foreground')
  window.hermesDesktop = {
    getConnectionFor: vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile })),
    getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
      ok: true,
      wsUrl: `wss://${connectionId}.invalid/${profile}`
    }))
  } as unknown as typeof window.hermesDesktop
  $groupChats.set({
    Classic: {
      continuityMode: 'desktop',
      log: [],
      members: [source],
      roomId: 'classic-room',
      watermarks: {}
    }
  })
})

afterEach(() => {
  closeSecondaryGateways()
  setPrimaryGateway(null)
  window.hermesDesktop = originalDesktop
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: originalCreateObjectURL })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: originalRevokeObjectURL })
  vi.restoreAllMocks()
})

describe('classic retained-file source route lifetime', () => {
  it.each(['edit', 'aba'] as const)(
    'blocks read selectors when the source has a material %s between resume and read',
    async change => {
      const saved = observeSave()
      let sessionReads = 0
      gateways.request.mockImplementation(async (_gateway, method: string) => {
        if (method === 'session.resume') {
          return Object.defineProperty({}, 'session_id', {
            get: () => {
              sessionReads += 1

              if (sessionReads === 1) {
                invalidate(change)
              }

              return 'original-runtime'
            }
          })
        }

        if (method === 'session.export.read') {
          return exportResponse()
        }

        throw new Error(`Unexpected RPC ${method}`)
      })

      await expect(downloadGroupChatAttachment('Classic', message, retainedAttachment())).rejects.toThrow(
        'route lease expired'
      )
      expect(gateways.request.mock.calls.filter(call => call[1] === 'session.export.read')).toHaveLength(0)
      expect(saved.createObjectURL).not.toHaveBeenCalled()

      gateways.request.mockReset()
      await expectFreshSave(saved)
    }
  )

  it.each(['read', 'digest', 'remove'] as const)(
    'rejects stale bytes after source invalidation at the %s boundary and keeps a fresh same-ID operation usable',
    async boundary => {
      const saved = observeSave()
      const heldRead = deferred<ReturnType<typeof exportResponse>>()
      const heldDigest = deferred<ArrayBuffer>()
      const actualDigest = await crypto.subtle.digest('SHA-256', contentBytes)
      const digestSpy = boundary === 'digest' ? vi.spyOn(crypto.subtle, 'digest').mockReturnValueOnce(heldDigest.promise) : null

      gateways.request.mockImplementation(async (_gateway, method: string) => {
        if (method === 'session.resume') {
          return { session_id: 'original-runtime' }
        }

        if (method === 'session.export.read') {
          return boundary === 'digest' ? exportResponse() : heldRead.promise
        }

        throw new Error(`Unexpected RPC ${method}`)
      })

      const pending = downloadGroupChatAttachment('Classic', message, retainedAttachment())

      if (boundary === 'digest') {
        await vi.waitFor(() => expect(digestSpy).toHaveBeenCalledOnce())
        invalidate('edit')
        heldDigest.resolve(actualDigest)
      } else {
        await vi.waitFor(() =>
          expect(gateways.request.mock.calls.filter(call => call[1] === 'session.export.read')).toHaveLength(1)
        )
        invalidate(boundary === 'remove' ? 'remove' : 'aba')
        heldRead.resolve(exportResponse())
      }

      await expect(pending).rejects.toThrow('route lease expired')
      expect(saved.createObjectURL).not.toHaveBeenCalled()

      digestSpy?.mockRestore()
      gateways.request.mockReset()
      await expectFreshSave(saved)
    }
  )

  it('rejects a same-ID ABA during synchronous pre-Save validation and keeps a fresh operation usable', async () => {
    const saved = observeSave()
    const originalAtob = globalThis.atob
    let decodes = 0

    const atobSpy = vi.spyOn(globalThis, 'atob').mockImplementation(value => {
      decodes += 1

      if (decodes === 2) {
        invalidate('aba')
      }

      return originalAtob(value)
    })

    gateways.request.mockImplementation(async (_gateway, method: string) => {
      if (method === 'session.resume') {
        return { session_id: 'original-runtime' }
      }

      if (method === 'session.export.read') {
        return exportResponse()
      }

      throw new Error(`Unexpected RPC ${method}`)
    })

    await expect(downloadGroupChatAttachment('Classic', message, retainedAttachment())).rejects.toThrow(
      'route lease expired'
    )
    expect(saved.createObjectURL).not.toHaveBeenCalled()

    atobSpy.mockRestore()
    gateways.request.mockReset()
    await expectFreshSave(saved)
  })
})
