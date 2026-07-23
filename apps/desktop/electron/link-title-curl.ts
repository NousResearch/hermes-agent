const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308])

export interface LinkTitleCurlHop {
  body: string
  location: null | string
  statusCode: number
}

export interface LinkTitleCurlDependencies {
  admitUrl(value: string): null | string
  maxRedirects: number
  now?: () => number
  readTitle(body: string): string
  request(url: string, timeoutMs: number): Promise<LinkTitleCurlHop>
  timeoutMs: number
}

export interface LinkTitleCurlRequestOptions {
  connectTimeoutSeconds: number
  headerPath: string
  maxBytes: number
  proxyUrl: string
  timeoutSeconds: number
  userAgent: string
}

function curlSocksProxyUrl(proxyUrl: string): string {
  return proxyUrl.replace(/^socks5:/, 'socks5h:')
}

export function linkTitleCurlRequestArgs(url: string, options: LinkTitleCurlRequestOptions): string[] {
  return [
    '--disable',
    '--no-location',
    '--proxy',
    curlSocksProxyUrl(options.proxyUrl),
    '--noproxy',
    '',
    '--silent',
    '--show-error',
    '--max-time',
    String(options.timeoutSeconds),
    '--connect-timeout',
    String(options.connectTimeoutSeconds),
    '--user-agent',
    options.userAgent,
    '--header',
    'Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5',
    '--header',
    'Accept-Language: en-US,en;q=0.7',
    '--header',
    'Accept-Encoding: identity',
    '--proto',
    '=http,https',
    '--max-filesize',
    String(options.maxBytes),
    '--dump-header',
    options.headerPath,
    '--raw',
    url
  ]
}

export function parseLinkTitleCurlHeaders(value: string): Pick<LinkTitleCurlHop, 'location' | 'statusCode'> {
  let location: null | string = null
  let statusCode = 0

  for (const block of value.split(/\r?\n\r?\n/)) {
    const lines = block.split(/\r?\n/)
    const status = /^HTTP\/\S+\s+(\d{3})\b/i.exec(lines[0] ?? '')

    if (!status) {
      continue
    }

    statusCode = Number.parseInt(status[1], 10)
    location = null

    for (const line of lines.slice(1)) {
      const match = /^location\s*:\s*(.*)$/i.exec(line)

      if (match) {
        location = match[1].trim() || null
      }
    }
  }

  return { location, statusCode }
}

export function createLinkTitleCurlFetcher(deps: LinkTitleCurlDependencies): (rawUrl: string) => Promise<string> {
  return async rawUrl => {
    const now = deps.now ?? Date.now
    const deadline = now() + deps.timeoutMs
    let redirects = 0
    let url = deps.admitUrl(rawUrl)

    if (!url) {
      return ''
    }

    while (true) {
      const remainingMs = deadline - now()

      if (remainingMs <= 0) {
        return ''
      }

      let response: LinkTitleCurlHop

      try {
        response = await deps.request(url, remainingMs)
      } catch {
        return ''
      }

      if (!REDIRECT_STATUS_CODES.has(response.statusCode) || !response.location) {
        return deps.readTitle(response.body)
      }

      if (redirects >= deps.maxRedirects) {
        return ''
      }

      let resolved: string

      try {
        resolved = new URL(response.location, url).href
      } catch {
        return ''
      }

      const nextUrl = deps.admitUrl(resolved)

      if (!nextUrl) {
        return ''
      }

      url = nextUrl
      redirects += 1
    }
  }
}
