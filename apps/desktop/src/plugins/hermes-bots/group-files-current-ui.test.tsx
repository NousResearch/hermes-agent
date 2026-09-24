import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Attachment, GroupMember } from './types'

const mocks = vi.hoisted(() => ({
  acquire: vi.fn(),
  assertCurrent: vi.fn(),
  notify: vi.fn(),
  release: vi.fn(),
  request: vi.fn()
}))

const route = { connectionId: 'root', mode: 'remote' as const, profile: 'default', targetProfile: 'default' }

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, acquireProfileRoute: mocks.acquire, notify: mocks.notify }
  }
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

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(async () => {
  vi.clearAllMocks()
  mocks.acquire.mockResolvedValue({
    assertCurrent: mocks.assertCurrent,
    generation: 1,
    release: mocks.release,
    request: mocks.request,
    route
  })
  const { $groupChats } = await import('./group-chat')
  $groupChats.set({
    Classic: {
      continuityMode: 'desktop',
      log: [
        {
          at: 1_700_000_000_000,
          from: { kind: 'member', name: 'default' },
          id: 'classic-message',
          images: [retainedAttachment()],
          text: 'Retained file'
        }
      ],
      members: [source],
      roomId: 'classic-room',
      watermarks: {}
    }
  })
  mocks.request.mockResolvedValueOnce({ session_id: 'original-runtime' }).mockResolvedValueOnce({
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
  cleanup()
  vi.restoreAllMocks()
})

describe('classic Files in the current Desktop composition', () => {
  it('opens from the legacy header, lists the retained ref, reads its exact generation and reaches Save bytes', async () => {
    let saved: Blob | undefined

    const createObjectURL = vi.fn((blob: Blob) => {
      saved = blob

      return 'blob:classic-files-ui'
    })

    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    const { SharedFilesControl } = await import('./group-files-view')
    const { $groupChats } = await import('./group-chat')
    render(<SharedFilesControl group="Classic" room={$groupChats.get().Classic} />)

    fireEvent.click(screen.getByRole('button', { name: 'Files' }))
    await screen.findByText('proof.bin')
    expect(screen.getByText('Files available on this Desktop.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download proof.bin' }))

    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1))
    expect(mocks.request.mock.calls.map(call => call[0])).toEqual(['session.resume', 'session.export.read'])
    expect(mocks.request.mock.calls[1]?.[1]).toMatchObject({ generation: 7, session_id: 'original-runtime' })
    expect(saved?.size).toBe(21)
    expect(saved?.type).toBe('application/octet-stream')
    expect(mocks.notify).not.toHaveBeenCalled()
    expect(mocks.release).toHaveBeenCalledOnce()
  })
})
