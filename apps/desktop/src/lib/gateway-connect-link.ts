export interface GatewayConnectRequest {
  url: string
  name: string
}

export function parseGatewayConnectRequest(params: Record<string, string>): GatewayConnectRequest | null {
  if (Object.keys(params).some(key => key !== 'url' && key !== 'name')) {
    return null
  }

  try {
    const url = new URL(params.url || '')

    if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) {
      return null
    }

    const name = (params.name || url.hostname).trim()

    if (!name || name.length > 128 || [...name].some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
      return null
    }

    url.pathname = url.pathname.replace(/\/+$/, '') || '/'

    return { url: url.toString().replace(/\/$/, ''), name }
  } catch {
    return null
  }
}
