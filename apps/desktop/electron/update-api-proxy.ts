import { HttpsProxyAgent } from 'https-proxy-agent'
import { getProxyForUrl } from 'proxy-from-env'

/** Node's https.get ignores proxy env vars; resolve them only for update API requests. */
export function updateCheckAgent(url: string): HttpsProxyAgent<string> | undefined {
  const proxyUrl = getProxyForUrl(url)

  return proxyUrl ? new HttpsProxyAgent(proxyUrl) : undefined
}
