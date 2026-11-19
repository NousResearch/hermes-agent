import { endpointUrl } from './browser-transport'

/** End this gateway's cookie session without navigating away from the loaded app. */
export async function logoutBrowserSession(basePath: string, fetcher: typeof fetch = fetch): Promise<void> {
  const response = await fetcher(endpointUrl('/auth/logout', basePath), {
    credentials: 'same-origin',
    method: 'POST',
    signal: AbortSignal.timeout(30_000)
  })

  if (!response.ok && response.status !== 401) {
    throw new Error(`Sign out failed (${response.status})`)
  }
}
