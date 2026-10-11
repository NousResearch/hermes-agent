import { type RefObject, useEffect } from 'react'

import { openExternalLink } from '@/lib/external-link'
import { CATALOG_ORIGIN, PLUGIN_CATALOG_NAME_RE } from '@/lib/plugin-catalog'
import { requestPluginCatalogInstallFromDeepLink } from '@/store/plugin-catalog-install'

/** A Hub message belongs to this iframe, never merely to a window on the same origin. */
export function useHubLinks(frame: RefObject<HTMLIFrameElement | null>, profile: null | string): void {
  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (
        !frame.current?.contentWindow ||
        event.source !== frame.current.contentWindow ||
        event.origin !== CATALOG_ORIGIN
      ) {
        return
      }

      if (event.data?.type === 'hermes-hub-links-ready') {
        frame.current.contentWindow.postMessage({ type: 'hermes-hub-links-enable' }, CATALOG_ORIGIN)

        return
      }

      if (event.data?.type !== 'hermes-hub-open-link' || typeof event.data.url !== 'string') {
        return
      }

      let url: URL

      try {
        url = new URL(event.data.url)
      } catch {
        return
      }

      if (url.protocol === 'https:' || url.protocol === 'http:') {
        openExternalLink(url.href)
      } else if (url.protocol === 'hermes:' && url.hostname === 'plugin' && url.pathname === '/install') {
        const name = url.searchParams.get('catalog') ?? ''

        if (PLUGIN_CATALOG_NAME_RE.test(name)) {
          void requestPluginCatalogInstallFromDeepLink(name, undefined, profile)
        }
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [frame, profile])
}
