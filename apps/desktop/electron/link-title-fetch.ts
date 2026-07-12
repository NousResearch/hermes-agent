import { lookup as dnsLookup } from 'node:dns/promises'
import { isIP } from 'node:net'

type LookupAddress = { address: string; family: number }

type CurlChild = {
  on(event: 'close' | 'error', listener: (...args: any[]) => void): void
  stdout?: { on(event: 'data', listener: (chunk: Buffer | string) => void): void }
}

type CurlSpawner = (args: string[]) => CurlChild
type Lookup = (hostname: string) => Promise<LookupAddress[]>

export const TITLE_BYTE_BUDGET = 96 * 1024
export const TITLE_MAX_REDIRECTS = 3

const TITLE_TIMEOUT_SECONDS = 5
const TITLE_CONNECT_TIMEOUT_SECONDS = 4
const IPV4_MAPPED_IPV6_PREFIX = '::ffff:'

export type PublicLinkTitleTarget = {
  curlResolveArgs: string[]
  url: URL
}

export type LinkTitleFetchOptions = {
  lookup?: Lookup
  maxRedirects?: number
  spawnCurl: CurlSpawner
  userAgent: string
}

export function createBoundedLinkTitleQueue(
  resolveTitle: (rawUrl: string) => Promise<string>,
  options: { maxConcurrent: number; maxQueued: number }
): (rawUrl: string) => Promise<string> {
  const queue: { resolve: (title: string) => void; url: string }[] = []
  let inFlight = 0

  const dequeue = () => {
    while (inFlight < options.maxConcurrent && queue.length) {
      const item = queue.shift()

      if (!item) {
        return
      }

      inFlight += 1
      resolveTitle(item.url)
        .catch(() => '')
        .then(title => item.resolve(title))
        .finally(() => {
          inFlight -= 1
          dequeue()
        })
    }
  }

  return rawUrl =>
    new Promise(resolve => {
      if (queue.length >= options.maxQueued) {
        resolve('')

        return
      }

      queue.push({ resolve, url: rawUrl })
      dequeue()
    })
}

function normalizedHost(value: string): string {
  return value.replace(/^\[/, '').replace(/\]$/, '').replace(/\.$/, '').toLowerCase()
}

function parseIpv4(value: string): number[] | null {
  const parts = value.split('.')

  if (parts.length !== 4) {
    return null
  }

  const octets = parts.map(part => Number(part))

  return octets.every(octet => Number.isInteger(octet) && octet >= 0 && octet <= 255) ? octets : null
}

function isPublicIpv4(value: string): boolean {
  const octets = parseIpv4(value)

  if (!octets) {
    return false
  }

  const [a, b, c] = octets

  return !(
    a === 0 ||
    a === 10 ||
    a === 127 ||
    a >= 224 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 0 && c === 0) ||
    (a === 192 && b === 0 && c === 2) ||
    (a === 192 && b === 88 && c === 99) ||
    (a === 192 && b === 168) ||
    (a === 198 && (b === 18 || b === 19)) ||
    (a === 198 && b === 51 && c === 100) ||
    (a === 203 && b === 0 && c === 113)
  )
}

function isPublicIpv6(value: string): boolean {
  const normalized = normalizedHost(value).split('%', 1)[0] ?? ''

  if (!normalized || normalized === '::' || normalized === '::1') {
    return false
  }

  if (normalized.startsWith(IPV4_MAPPED_IPV6_PREFIX)) {
    return isPublicIpv4(normalized.slice(IPV4_MAPPED_IPV6_PREFIX.length))
  }

  return !(
    normalized.startsWith('fc') ||
    normalized.startsWith('fd') ||
    /^fe[89ab]/.test(normalized) ||
    normalized.startsWith('ff') ||
    normalized.startsWith('2001:db8:')
  )
}

function isPublicAddress(value: string): boolean {
  const address = normalizedHost(value).split('%', 1)[0] ?? ''
  const family = isIP(address)

  if (family === 4) {
    return isPublicIpv4(address)
  }

  if (family === 6) {
    return isPublicIpv6(address)
  }

  return false
}

function isBlockedHostname(hostname: string): boolean {
  return (
    hostname === 'localhost' ||
    hostname === 'localhost.localdomain' ||
    hostname === 'metadata.google.internal' ||
    hostname === 'metadata.goog' ||
    hostname.endsWith('.corp') ||
    hostname.endsWith('.home') ||
    hostname.endsWith('.internal') ||
    hostname.endsWith('.lan') ||
    hostname.endsWith('.local') ||
    hostname.endsWith('.localdomain') ||
    !hostname.includes('.')
  )
}

function parsePublicUrl(rawUrl: string): URL | null {
  try {
    const url = new URL(String(rawUrl || '').trim())
    const host = normalizedHost(url.hostname)

    if (
      !/^https?:$/.test(url.protocol) ||
      !host ||
      url.username ||
      url.password ||
      isBlockedHostname(host) ||
      (isIP(host) !== 0 && !isPublicAddress(host))
    ) {
      return null
    }

    return url
  } catch {
    return null
  }
}

export function linkTitleTargetIsPublic(rawUrl: string): boolean {
  return parsePublicUrl(rawUrl) !== null
}

function curlResolveValue(hostname: string, port: string, address: string): string {
  const host = hostname.includes(':') ? `[${hostname}]` : hostname
  const pinnedAddress = address.includes(':') ? `[${address}]` : address

  return `${host}:${port}:${pinnedAddress}`
}

export async function resolvePublicLinkTitleTarget(
  rawUrl: string,
  options: { lookup?: Lookup } = {}
): Promise<PublicLinkTitleTarget> {
  const url = parsePublicUrl(rawUrl)

  if (!url) {
    throw new Error('Link title target is not public')
  }

  const hostname = normalizedHost(url.hostname)
  const lookup = options.lookup ?? (async host => dnsLookup(host, { all: true, verbatim: true }))
  const addresses = isIP(hostname) ? [{ address: hostname, family: isIP(hostname) }] : await lookup(hostname)

  if (!addresses.length || addresses.some(result => !isPublicAddress(result.address))) {
    throw new Error('Link title target resolved to a non-public address')
  }

  const port = url.port || (url.protocol === 'https:' ? '443' : '80')
  const curlResolveArgs = addresses.flatMap(result => ['--resolve', curlResolveValue(hostname, port, result.address)])

  return { curlResolveArgs, url }
}

function parseHeaders(rawHeaders: string): { location: null | string; status: number } {
  const lines = rawHeaders.split(/\r?\n/)
  const status = Number(/^HTTP\/\S+\s+(\d{3})/.exec(lines[0] ?? '')?.[1] ?? 0)
  const location = lines.slice(1).find(line => /^location\s*:/i.test(line))

  return { location: location ? location.slice(location.indexOf(':') + 1).trim() : null, status }
}

function parseCurlOutput(output: Buffer): { body: Buffer; location: null | string; status: number } {
  let offset = 0
  let parsed = { location: null as null | string, status: 0 }

  while (output.subarray(offset).toString('latin1', 0, 5) === 'HTTP/') {
    const lf = output.indexOf('\n\n', offset, 'latin1')
    const crlf = output.indexOf('\r\n\r\n', offset, 'latin1')
    const headerEnd = crlf >= 0 && (lf < 0 || crlf < lf) ? crlf + 4 : lf >= 0 ? lf + 2 : -1

    if (headerEnd < 0) {
      return { body: Buffer.alloc(0), location: null, status: 0 }
    }

    parsed = parseHeaders(output.subarray(offset, headerEnd).toString('latin1'))
    offset = headerEnd
  }

  return { body: output.subarray(offset), ...parsed }
}

function invokeCurl(args: string[], spawnCurl: CurlSpawner): Promise<Buffer> {
  return new Promise(resolve => {
    let child: CurlChild

    try {
      child = spawnCurl(args)
    } catch {
      resolve(Buffer.alloc(0))

      return
    }

    const chunks: Buffer[] = []
    let bytes = 0

    child.stdout?.on('data', chunk => {
      if (bytes >= TITLE_BYTE_BUDGET) {
        return
      }
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      const remaining = TITLE_BYTE_BUDGET - bytes
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer

      chunks.push(next)
      bytes += next.length
    })
    child.on('error', () => resolve(Buffer.alloc(0)))
    child.on('close', () => resolve(Buffer.concat(chunks)))
  })
}

function curlArgsForTarget(target: PublicLinkTitleTarget, userAgent: string): string[] {
  return [
    '--silent',
    '--show-error',
    '--noproxy',
    '*',
    '--max-time',
    String(TITLE_TIMEOUT_SECONDS),
    '--connect-timeout',
    String(TITLE_CONNECT_TIMEOUT_SECONDS),
    '--max-filesize',
    String(TITLE_BYTE_BUDGET),
    '--user-agent',
    userAgent,
    '--header',
    'Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5',
    '--header',
    'Accept-Language: en-US,en;q=0.7',
    '--header',
    'Accept-Encoding: identity',
    '--raw',
    '--dump-header',
    '-',
    '--output',
    '-',
    ...target.curlResolveArgs,
    target.url.toString()
  ]
}

export async function fetchPinnedLinkTitle(rawUrl: string, options: LinkTitleFetchOptions): Promise<string> {
  let currentUrl = String(rawUrl || '').trim()
  const maxRedirects = options.maxRedirects ?? TITLE_MAX_REDIRECTS

  for (let redirectCount = 0; redirectCount <= maxRedirects; redirectCount += 1) {
    let target: PublicLinkTitleTarget

    try {
      target = await resolvePublicLinkTitleTarget(currentUrl, { lookup: options.lookup })
    } catch {
      return ''
    }

    const response = parseCurlOutput(await invokeCurl(curlArgsForTarget(target, options.userAgent), options.spawnCurl))

    if (response.status >= 300 && response.status < 400 && response.location) {
      try {
        currentUrl = new URL(response.location, target.url).toString()
      } catch {
        return ''
      }
      continue
    }

    return response.status >= 200 && response.status < 300 ? response.body.toString('utf8') : ''
  }

  return ''
}
