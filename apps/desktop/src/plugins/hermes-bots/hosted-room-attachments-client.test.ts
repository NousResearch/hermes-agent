import { describe, expect, it, vi } from 'vitest'

import { readHostedMessageAttachment, stageHostedMessageAttachments } from './hosted-room-attachments-client'
import type { Attachment, ProfileRoute } from './types'

const ROUTE = {
  connectionId: 'gateway-a',
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
} as ProfileRoute

const image = (data: string, name = 'proof.png'): Attachment => ({
  data: `data:image/png;base64,${data}`,
  kind: 'image',
  name
})

describe('hosted Group Chat attachment client', () => {
  it('rejects count and aggregate limits before the first upload', async () => {
    const request = vi.fn()

    await expect(
      stageHostedMessageAttachments(
        request,
        ROUTE,
        'room-1',
        Array.from({ length: 9 }, (_, index) => image('YQ==', `proof-${index}.png`))
      )
    ).rejects.toThrow('at most 8')
    expect(request).not.toHaveBeenCalled()

    const thirteenMb = 'A'.repeat(Math.ceil(13_000_000 / 3) * 4)

    await expect(
      stageHostedMessageAttachments(request, ROUTE, 'room-1', [
        image(thirteenMb, 'first.png'),
        image(thirteenMb, 'second.png')
      ])
    ).rejects.toThrow('25MB')
    expect(request).not.toHaveBeenCalled()
  })

  it('rejects bytes that do not match the committed receipt', async () => {
    const request = vi.fn(async () => ({
      attachment: {
        attachment_id: 'att_0123456789abcdef0123456789abcdef',
        mime: 'image/png',
        name: 'proof.png',
        size: 2
      },
      content_base64: 'YQ=='
    }))

    await expect(
      readHostedMessageAttachment(request as never, ROUTE, 'room-1', 'event-1', {
        attachmentId: 'att_0123456789abcdef0123456789abcdef',
        kind: 'image',
        mime: 'image/png',
        name: 'proof.png',
        size: 1
      })
    ).rejects.toThrow('invalid attachment')
  })

  it('rejects a response whose MIME type differs from the committed manifest', async () => {
    const request = vi.fn(async () => ({
      attachment: {
        attachment_id: 'att_0123456789abcdef0123456789abcdef',
        mime: 'text/html',
        name: 'proof.png',
        size: 1
      },
      content_base64: 'YQ=='
    }))

    await expect(
      readHostedMessageAttachment(request as never, ROUTE, 'room-1', 'event-1', {
        attachmentId: 'att_0123456789abcdef0123456789abcdef',
        kind: 'image',
        mime: 'image/png',
        name: 'proof.png',
        size: 1
      })
    ).rejects.toThrow('invalid attachment')
  })

  it('fails a partial upload without producing a manifest', async () => {
    const attachments = [image('YQ==', 'first.png'), image('Yg==', 'second.png')]

    const request = vi
      .fn()
      .mockResolvedValueOnce({
        attachment: {
          attachment_id: 'att_0123456789abcdef0123456789abcdef',
          kind: 'image',
          mime: 'image/png',
          name: 'first.png',
          size: 1
        }
      })
      .mockRejectedValueOnce(new Error('connection lost'))

    await expect(stageHostedMessageAttachments(request, ROUTE, 'room-1', attachments)).rejects.toThrow(
      'connection lost'
    )
    expect(request).toHaveBeenCalledTimes(2)
    const firstAttemptIds = request.mock.calls.map(call => call[2].upload_id)

    request.mockReset()
    request
      .mockResolvedValueOnce({
        attachment: {
          attachment_id: 'att_0123456789abcdef0123456789abcdef',
          kind: 'image',
          mime: 'image/png',
          name: 'first.png',
          size: 1
        }
      })
      .mockResolvedValueOnce({
        attachment: {
          attachment_id: 'att_fedcba9876543210fedcba9876543210',
          kind: 'image',
          mime: 'image/png',
          name: 'second.png',
          size: 1
        }
      })

    await expect(stageHostedMessageAttachments(request, ROUTE, 'room-1', attachments)).resolves.toHaveLength(2)
    expect(request.mock.calls.map(call => call[2].upload_id)).toEqual(firstAttemptIds)
  })

  it('stops oversized reads before constructing a data URL', async () => {
    const request = vi.fn(async () => ({
      attachment: {
        attachment_id: 'att_0123456789abcdef0123456789abcdef',
        mime: 'image/png',
        name: 'proof.png',
        size: 15_000_001
      },
      content_base64: 'A'.repeat(Math.ceil(15_000_001 / 3) * 4)
    }))

    await expect(
      readHostedMessageAttachment(request as never, ROUTE, 'room-1', 'event-1', {
        attachmentId: 'att_0123456789abcdef0123456789abcdef',
        kind: 'image',
        mime: 'image/png',
        name: 'proof.png',
        size: 15_000_001
      })
    ).rejects.toThrow('invalid attachment')
  })
})
