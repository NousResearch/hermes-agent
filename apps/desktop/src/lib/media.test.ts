import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { downloadGatewayMediaFile } from './media'

const api = vi.fn(async ({ path }: { path: string }) => {
  if (path.startsWith('/api/fs/read-data-url?')) {
    return { dataUrl: 'data:application/pdf;base64,cGRm' }
  }

  throw new Error(`unexpected path ${path}`)
})

describe('downloadGatewayMediaFile', () => {
  let clickSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    $connection.set({ mode: 'remote' } as never)
    vi.stubGlobal('window', { hermesDesktop: { api }, setTimeout: vi.fn() })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob(['pdf'], { type: 'application/pdf' }) }))
    )
    URL.createObjectURL = vi.fn(() => 'blob:media-test')
    URL.revokeObjectURL = vi.fn()
    clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    clickSpy.mockRestore()
    $connection.set(null)
  })

  it('fetches the file over /api/fs/read-data-url and triggers a download', async () => {
    await downloadGatewayMediaFile('/home/user/.hermes/cache/report.pdf')

    expect(api).toHaveBeenCalledWith({
      path: `/api/fs/read-data-url?path=${encodeURIComponent('/home/user/.hermes/cache/report.pdf')}`
    })
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('strips a file:// prefix before asking the gateway', async () => {
    await downloadGatewayMediaFile('file:///home/user/.hermes/cache/report.pdf')

    expect(api).toHaveBeenCalledWith({
      path: `/api/fs/read-data-url?path=${encodeURIComponent('/home/user/.hermes/cache/report.pdf')}`
    })
  })

  it('rejects when the gateway refuses the read', async () => {
    api.mockRejectedValueOnce(new Error('413 File too large'))

    await expect(downloadGatewayMediaFile('/home/user/huge.pdf')).rejects.toThrow('413')
    expect(clickSpy).not.toHaveBeenCalled()
  })

  it('rejects when the gateway returns no data', async () => {
    api.mockResolvedValueOnce({ dataUrl: '' })

    await expect(downloadGatewayMediaFile('/home/user/empty.pdf')).rejects.toThrow('no file data')
    expect(clickSpy).not.toHaveBeenCalled()
  })
})
