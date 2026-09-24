import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { Attachment, GroupMember } from './types'

const calls = vi.hoisted(() => ({
  acquire: vi.fn(),
  assertCurrent: vi.fn(),
  release: vi.fn(),
  request: vi.fn()
}))

const route = { connectionId: 'root', mode: 'remote' as const, profile: 'default', targetProfile: 'default' }

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({ acquireProfileRoute: calls.acquire })
})

vi.mock('./routing', () => ({
  botConnectionRoute: () => route,
  requestForBot: vi.fn()
}))

const source: GroupMember = { name: 'default' }
const digest = '30c313332d0220a37fef2d7ab61b099304914737bc0c3c853c59bf1b20249d4b'
const content = 'Y2xhc3NpYy1jdXJyZW50LWJ5dGVz'

function retainedAttachment(): Attachment {
  return {
    kind: 'file',
    mime: 'application/octet-stream',
    name: 'proof.bin',
    size: 21,
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

beforeEach(async () => {
  vi.clearAllMocks()
  calls.acquire.mockResolvedValue({
    assertCurrent: calls.assertCurrent,
    generation: 1,
    release: calls.release,
    request: calls.request,
    route
  })
  const { $groupChats } = await import('./group-chat')
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

describe('current-runtime retained classic reader', () => {
  it('resumes the original producer and sends the exact generation before accepting verified bytes', async () => {
    calls.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === 'session.resume') {
        expect(params).toEqual({ session_id: 'original-session', profile: 'default', omit_messages: true })

        return { session_id: 'original-runtime' }
      }

      if (method === 'session.export.read') {
        expect(params).toEqual({
          session_id: 'original-runtime',
          installation: 'install:root',
          group_id: 'classic-room',
          export_id: `ce_${'1'.repeat(64)}`,
          artifact_id: 'rart_0123456789abcdef0123456789abcdef',
          generation: 7
        })

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
            size: 21
          }
        }
      }

      throw new Error(`unexpected RPC ${method}`)
    })

    const { readClassicAttachment } = await import('./classic-output')
    const result = await readClassicAttachment('Classic', retainedAttachment())

    expect(result.data).toBe(`data:application/octet-stream;base64,${content}`)
    expect(calls.request.mock.calls.map(call => call[0])).toEqual(['session.resume', 'session.export.read'])
    expect(calls.acquire).toHaveBeenCalledExactlyOnceWith(route)
    expect(calls.release).toHaveBeenCalledOnce()
  })

  it('fails closed when the original session is gone and never creates a replacement', async () => {
    calls.request.mockRejectedValueOnce(Object.assign(new Error('gone'), { code: 4007 }))
    const { readClassicAttachment } = await import('./classic-output')

    await expect(readClassicAttachment('Classic', retainedAttachment())).rejects.toThrow(
      'original producer session is unavailable'
    )
    expect(calls.request.mock.calls.map(call => call[0])).toEqual(['session.resume'])
    expect(calls.release).toHaveBeenCalledOnce()
  })

  it('rejects a response from another generation before exposing bytes', async () => {
    calls.request.mockResolvedValueOnce({ session_id: 'original-runtime' }).mockResolvedValueOnce({
      content_base64: content,
      export_id: `ce_${'1'.repeat(64)}`,
      generation: 8,
      group_id: 'classic-room',
      item: {
        artifact_id: 'rart_0123456789abcdef0123456789abcdef',
        kind: 'file',
        mime: 'application/octet-stream',
        name: 'proof.bin',
        sha256: digest,
        size: 21
      }
    })

    const { readClassicAttachment } = await import('./classic-output')
    await expect(readClassicAttachment('Classic', retainedAttachment())).rejects.toThrow('verification failed')
    expect(calls.release).toHaveBeenCalledOnce()
  })
})
