import { describe, expect, it, vi } from 'vitest'

import { createBrowserThemesBridge } from './browser-themes'

describe('createBrowserThemesBridge', () => {
  it('proxies search queries with query parameters to the gateway API', async () => {
    const mockApi = vi.fn().mockResolvedValue([
      { displayName: 'Night Owl', extensionId: 'sdras.night-owl', installs: 100 }
    ])

    const bridge = createBrowserThemesBridge(mockApi)
    const results = await bridge.themes.searchMarketplace('night')

    expect(mockApi).toHaveBeenCalledWith({
      path: '/api/themes/marketplace/search?q=night'
    })
    expect(results).toHaveLength(1)
    expect(results[0].extensionId).toBe('sdras.night-owl')
  })

  it('proxies empty search queries cleanly', async () => {
    const mockApi = vi.fn().mockResolvedValue([])

    const bridge = createBrowserThemesBridge(mockApi)
    await bridge.themes.searchMarketplace('')

    expect(mockApi).toHaveBeenCalledWith({
      path: '/api/themes/marketplace/search'
    })
  })

  it('proxies theme package downloads and returns contributed theme contents', async () => {
    const mockApi = vi.fn().mockResolvedValue({
      displayName: 'Tokyo Night',
      extensionId: 'enkia.tokyo-night',
      themes: [{ label: 'Tokyo Night', contents: '{"colors": {}}' }]
    })

    const bridge = createBrowserThemesBridge(mockApi)
    const result = await bridge.themes.fetchMarketplace('enkia.tokyo-night')

    expect(mockApi).toHaveBeenCalledWith({
      path: '/api/themes/marketplace/fetch?id=enkia.tokyo-night'
    })
    expect(result.extensionId).toBe('enkia.tokyo-night')
    expect(result.themes).toHaveLength(1)
  })
})
