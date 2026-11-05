import type { DesktopMarketplaceSearchItem, DesktopMarketplaceThemeResult, HermesApiRequest } from '@/global'

export interface BrowserThemesBridge {
  themes: {
    fetchMarketplace: (id: string) => Promise<DesktopMarketplaceThemeResult>
    searchMarketplace: (query: string) => Promise<DesktopMarketplaceSearchItem[]>
  }
}

/**
 * Connect the webapp renderer to the gateway's Marketplace theme search and download
 * proxy, providing feature parity with Electron's native main-process fetcher.
 */
export function createBrowserThemesBridge(api: <T>(request: HermesApiRequest) => Promise<T>): BrowserThemesBridge {
  return {
    themes: {
      searchMarketplace: async (query: string): Promise<DesktopMarketplaceSearchItem[]> => {
        const trimmed = (query || '').trim()
        const q = trimmed ? `?q=${encodeURIComponent(trimmed)}` : ''

        return api<DesktopMarketplaceSearchItem[]>({
          path: `/api/themes/marketplace/search${q}`
        })
      },
      fetchMarketplace: async (id: string): Promise<DesktopMarketplaceThemeResult> => {
        return api<DesktopMarketplaceThemeResult>({
          path: `/api/themes/marketplace/fetch?id=${encodeURIComponent(id.trim())}`
        })
      }
    }
  }
}
