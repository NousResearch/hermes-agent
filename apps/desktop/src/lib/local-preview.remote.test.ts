import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { normalizeOrLocalPreviewTarget } from './local-preview'

describe('normalizeOrLocalPreviewTarget remote mode', () => {
  afterEach(() => {
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  it('keeps the gateway path but uses the remote bytes-to-temp URL for file opens', async () => {
    const api = vi.fn(async () => ({
      binary: false,
      byteSize: 12,
      language: 'html',
      mimeType: 'text/html',
      path: '/tmp/report.html',
      text: '<h1>ok</h1>'
    }))

    $connection.set({ mode: 'remote', profile: 'mbp' } as never)
    vi.stubGlobal('window', { hermesDesktop: { api } })

    const target = await normalizeOrLocalPreviewTarget('/tmp/report.html')

    expect(target).toMatchObject({
      kind: 'file',
      path: '/tmp/report.html',
      previewKind: 'html',
      url: 'hermes-gateway-file://open?path=%2Ftmp%2Freport.html&profile=mbp'
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-text?path=%2Ftmp%2Freport.html',
      profile: 'mbp'
    })
  })
})
