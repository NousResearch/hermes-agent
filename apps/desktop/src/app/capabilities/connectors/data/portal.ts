// Where the portal's connectors admin lives.
//
// The desktop already resolves ONE portal origin — the default, or the one the
// install was pointed at — and hands it to the renderer as
// `cloud.status().portalBaseUrl`. Every connectors link on this page asks that
// one resolver, so a staging or self-hosted portal is where the person is sent
// rather than an origin they have no session on. The billing constant stays as
// the last rung, for a bridge that cannot answer.

import { useEffect, useState } from 'react'

import { FALLBACK_PORTAL_URL } from '../../../settings/billing/use-billing-state'

const CONNECTORS_PATH = '/connectors'

/** Shown until the bridge answers, and kept when it cannot. */
export const FALLBACK_CONNECTORS_ADMIN_URL = `${FALLBACK_PORTAL_URL}${CONNECTORS_PATH}`

const adminUrl = (base: string): string => {
  const origin = base.replace(/\/+$/, '')

  return origin ? `${origin}${CONNECTORS_PATH}` : FALLBACK_CONNECTORS_ADMIN_URL
}

/** The connectors admin URL for this install. One resolver, so the kebab item
 *  and the dialog's organisation link can never point at different portals. */
export function useConnectorsAdminUrl(): string {
  const [url, setUrl] = useState(FALLBACK_CONNECTORS_ADMIN_URL)

  useEffect(() => {
    let cancelled = false

    void window.hermesDesktop?.cloud
      ?.status()
      .then(status => {
        if (!cancelled) {
          setUrl(adminUrl(status.portalBaseUrl))
        }
      })
      // The fallback is already on screen, and a portal link is not worth a
      // notice: the person came here for their tools.
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [])

  return url
}
