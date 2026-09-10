import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { $gateway } from '@/store/gateway'
import { handleDesktopBridgeEvent } from './desktop-bridge'

vi.mock('@/app/chat/right-rail/preview-reader')
vi.mock('@/store/gateway')

const mockRequest = vi.fn()

describe('handleDesktopBridgeEvent - preview.read.request', () => {
  beforeEach(() => {
    mockRequest.mockReset()
    vi.mocked($gateway.get).mockReturnValue({ request: mockRequest } as any)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('sends preview.read.respond with text when readActivePreview resolves', async () => {
    vi.mocked(readActivePreview).mockResolvedValue({ text: 'hello', title: 'Page', url: 'https://x.com' })
    handleDesktopBridgeEvent({
      event: { type: 'preview.read.request' },
      payload: { request_id: 'req-1' },
      isActiveEvent: true,
    } as any)
    await Promise.resolve()
    expect(mockRequest).toHaveBeenCalledWith('preview.read.respond', {
      request_id: 'req-1',
      text: expect.any(String),
    })
  })

  it('sends empty preview.read.respond when readActivePreview rejects', async () => {
    vi.mocked(readActivePreview).mockRejectedValue(new Error('pane unavailable'))
    handleDesktopBridgeEvent({
      event: { type: 'preview.read.request' },
      payload: { request_id: 'req-2' },
      isActiveEvent: true,
    } as any)
    await Promise.resolve()
    await Promise.resolve()
    expect(mockRequest).toHaveBeenCalledWith('preview.read.respond', {
      request_id: 'req-2',
      text: '',
    })
  })
})
