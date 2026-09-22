import type * as HermesSdk from '@hermes/plugin-sdk'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Attachment, GroupMember } from './types'

const calls = vi.hoisted(() => ({
  acquire: vi.fn(),
  assertCurrent: vi.fn(),
  release: vi.fn(),
  request: vi.fn()
}))

const route = { connectionId: 'root', mode: 'remote' as const, profile: 'default', targetProfile: 'default' }

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, acquireProfileRoute: calls.acquire } }
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
  calls.request.mockResolvedValueOnce({ session_id: 'original-runtime' }).mockResolvedValueOnce({
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
  })
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('retained classic byte delivery', () => {
  it('reads verified original-generation bytes and hands them to the current Save sink', async () => {
    vi.useFakeTimers()
    let saved: Blob | undefined

    const createObjectURL = vi.fn((blob: Blob) => {
      saved = blob

      return 'blob:classic-current-proof'
    })

    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    const { downloadGroupChatAttachment } = await import('./group-attachment-download')
    await downloadGroupChatAttachment(
      'Classic',
      { at: 1, from: { kind: 'member', name: 'default' }, images: [], text: '' },
      retainedAttachment()
    )

    expect(calls.request.mock.calls.map(call => call[0])).toEqual(['session.resume', 'session.export.read'])
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(saved?.size).toBe(21)
    expect(saved?.type).toBe('application/octet-stream')
    expect(click).toHaveBeenCalledTimes(1)
    expect(calls.release).toHaveBeenCalledOnce()
    vi.advanceTimersByTime(30_000)
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:classic-current-proof')
  })
})
