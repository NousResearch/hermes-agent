import type { HermesApiRequest } from '@/global'
import { translateNow } from '@/i18n'
import { requireBrowserConnection } from '@/lib/browser-connection'
import { fetchBrowserImage } from '@/lib/browser-image-download'
import { type BrowserBootstrap, fileEndpointUrl } from '@/lib/browser-transport'
import { clickDownloadLink, downloadBlob, downloadFilename } from '@/lib/download'
import { notifyError } from '@/store/notifications'

function queryPath(route: string, values: Record<string, boolean | null | string | undefined>) {
  const query = new URLSearchParams()

  for (const [key, value] of Object.entries(values)) {
    if (value !== null && value !== undefined) {query.set(key, String(value))}
  }

  return `${route}?${query.toString()}`
}

interface BrowserFilesOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  bootstrap: BrowserBootstrap
  currentProfile: () => string | null
}

/** Server files reach the page through /api/fs reads and ticketed download/stream URLs. */
export function createBrowserFilesBridge({
  api,
  bootstrap,
  currentProfile
}: BrowserFilesOptions): Pick<
  Window['hermesDesktop'],
  | 'getGatewayFileStreamUrl'
  | 'readDir'
  | 'readFileDataUrl'
  | 'readFileDataUrlForAttach'
  | 'readFileText'
  | 'saveGatewayFile'
  | 'saveImageFromUrl'
> {
  const fsGet = <T>(route: string, path: string) =>
    api<T>({ path: queryPath(`/api/fs/${route}`, { path }), profile: currentProfile() })

  const readDataUrl = async (path: string) => (await fsGet<{ dataUrl: string }>('read-data-url', path)).dataUrl

  const downloadUrl = async (url: string, filename = '') => {
    const target = new URL(url, window.location.href)

    // Cross-origin anchors ignore download and navigate the current tab.
    // Fetch without Hermes credentials; CORS denial must not fall back to navigation.
    if (target.origin !== window.location.origin && !['blob:', 'data:'].includes(target.protocol)) {
      const image = await fetchBrowserImage(target)

      // A blob URL carries no name of its own; unnamed, the file saves as its UUID.
      downloadBlob(image, filename || downloadFilename(target.href, image.type))
    } else {
      // An empty name lets the server's Content-Disposition name gateway files.
      clickDownloadLink(target.href, filename)
    }

    return true
  }

  return {
    getGatewayFileStreamUrl: async payload => {
      requireBrowserConnection(payload.connectionId)

      return (await fileEndpointUrl(bootstrap, 'stream', {
        path: payload.path,
        profile: payload.profile ?? currentProfile()
      })).href
    },
    readDir: (path: string) =>
      fsGet<Awaited<ReturnType<Window['hermesDesktop']['readDir']>>>('list', path),
    readFileDataUrl: readDataUrl,
    readFileDataUrlForAttach: readDataUrl,
    readFileText: (path: string) =>
      fsGet<Awaited<ReturnType<Window['hermesDesktop']['readFileText']>>>('read-text', path),
    saveGatewayFile: async payload => {
      requireBrowserConnection(payload.connectionId)
      const query = { path: payload.path, profile: payload.profile ?? currentProfile(), session_id: payload.sessionId }

      // Validate before handing the transfer to the browser so missing or
      // denied files surface in the chat without navigating away. HEAD keeps
      // large downloads out of renderer memory.
      await api({
        method: 'HEAD',
        path: queryPath('/api/files/download', { path: query.path, session_id: query.session_id }),
        profile: query.profile
      })

      return {
        saved: await downloadUrl((await fileEndpointUrl(bootstrap, 'download', query)).href, payload.suggestedName)
      }
    },
    saveImageFromUrl: async (url: string) => {
      try {
        return await downloadUrl(url)
      } catch (error) {
        // Context-menu consumers fire and forget; surface failures here.
        notifyError(error, translateNow('fileMenu.downloadFailed'))

        return false
      }
    }
  }
}
