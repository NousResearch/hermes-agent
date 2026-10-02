// Destination admission for the link-title pipeline (#126885).
//
// The renderer's title prefetch fires for every http(s) link a conversation or
// the Artifacts page renders — including links injected by tool payloads — so
// the fetch is attacker-influenced, not user-chosen. `isFetchableHttpUrl`
// (electron/link-title-url.ts) only gates the scheme; this module refuses the
// *destinations* a title GET must never touch, mirroring the classes Hermes'
// own `tools/url_safety.py` blocks for agent fetches: private, loopback,
// link-local, CGNAT, cloud-metadata and reserved ranges, plus token-bearing
// URLs whose GET would consume a one-time link.
//
// The classification functions are pure so the synchronous
// `webRequest.onBeforeRequest` hook can use them; they see through `127.1`,
// `0x7f.1` and integer IPv4 spellings because curl's inet_aton does.
// `isSafeTitleFetchTarget` adds the DNS-resolution layer and is the check the
// curl tier and the hidden window's `loadURL()` run before any network I/O,
// and again on every redirect hop the curl tier follows manually.
// `resolveTitleFetchTarget` also returns the vetted DNS answers so the curl
// tier can pin the connection to them (no second, rebindable lookup), and
// `isSafeLinkMetadataTarget` is the same policy for Chromium-stack fetches
// (favicons, the hidden title window), which never use the env proxy.

import { lookup } from 'node:dns/promises'

/** Curl's inet_aton accepts 1–4 parts; leading parts are bytes, the last part
 * fills the remaining width (so `127.1` is 127.0.0.1), each decimal, 0x-hex
 * or leading-0 octal. */
function parseLooseIpv4(host: string): [number, number, number, number] | null {
  const parts = host.split('.')

  if (!parts.length || parts.length > 4) {
    return null
  }

  const values: number[] = []

  for (const part of parts) {
    if (/^0[xX][0-9a-fA-F]{1,8}$/.test(part)) {
      values.push(parseInt(part.slice(2), 16))
    } else if (/^0[0-7]{1,11}$/.test(part) || /^\d{1,10}$/.test(part)) {
      values.push(parseInt(part, part.startsWith('0') && part.length > 1 ? 8 : 10))
    } else {
      return null
    }
  }

  const head = values.slice(0, -1)
  const tail = values[values.length - 1]

  if (
    head.some(value => value > 0xff) ||
    tail >= 2 ** (32 - 8 * head.length) ||
    values.some(value => !Number.isSafeInteger(value))
  ) {
    return null
  }

  let value = head.reduce((acc, part) => ((acc << 8) | part) >>> 0, 0)

  value = ((value << (32 - 8 * head.length)) | tail) >>> 0

  return [value >>> 24, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff]
}

/** 128-bit BigInt for an IPv6 literal (handles `::`, IPv4 tails, zone IDs), else null. */
function parseIpv6(host: string): bigint | null {
  const raw = host.split('%')[0] // strip the zone ID (fe80::1%eth0)

  if (!raw.includes(':')) {
    return null
  }

  const tailV4 = raw.match(/^(.*:)(\d+\.\d+\.\d+\.\d+)$/)
  // The capture keeps the ':' before the IPv4 tail; drop it so the remaining
  // groups parse cleanly (`::ffff:` + `127.0.0.1`).
  const head = tailV4 ? tailV4[1].replace(/:$/, '') : raw
  const v4 = tailV4 ? parseLooseIpv4(tailV4[2]) : null

  if (tailV4 && !v4) {
    return null
  }

  const sections = head.split('::')

  if (sections.length > 2) {
    return null
  }

  const parseGroups = (text: string): number[] | null => {
    if (!text.length) {
      return []
    }

    const groups = text.split(':').map(group => (/^[0-9a-fA-F]{1,4}$/.test(group) ? parseInt(group, 16) : null))

    return groups.includes(null) ? null : (groups as number[])
  }

  const left = parseGroups(sections[0])
  const right = sections.length === 2 ? parseGroups(sections[1]) : null

  if (left === null || right === null) {
    return null
  }

  const v4Groups = v4 ? [((v4[0] << 8) | v4[1]) & 0xffff, ((v4[2] << 8) | v4[3]) & 0xffff] : []
  const fill = 8 - left.length - (right?.length ?? 0) - v4Groups.length

  if (sections.length === 2 ? fill < 0 : left.length !== 8) {
    return null
  }

  const groups = [...left, ...new Array(Math.max(fill, 0)).fill(0), ...(right ?? []), ...v4Groups]

  return groups.reduce((acc, group) => (acc << 16n) | BigInt(group), 0n)
}

const IPV4_BLOCKED_RANGES: readonly (readonly [number, number, string])[] = [
  [0x00000000, 0x00000000, 'unspecified'], // 0.0.0.0
  [0x0a000000, 0x0affffff, 'private'], // 10.0.0.0/8
  [0x64400000, 0x647fffff, 'cgnat'], // 100.64.0.0/10 — Tailscale/WireGuard/cloud-internal
  [0x7f000000, 0x7fffffff, 'loopback'], // 127.0.0.0/8
  [0xa9fe0000, 0xa9feffff, 'link-local'], // 169.254.0.0/16
  [0xac100000, 0xac1fffff, 'private'], // 172.16.0.0/12
  [0xc0000000, 0xc00000ff, 'reserved'], // 192.0.0.0/24 IETF protocol assignments
  [0xc0000200, 0xc00002ff, 'reserved'], // 192.0.2.0/24 TEST-NET-1
  [0xc0a80000, 0xc0a8ffff, 'private'], // 192.168.0.0/16
  [0xc6120000, 0xc613ffff, 'benchmark'], // 198.18.0.0/15
  [0xc6336400, 0xc63364ff, 'reserved'], // 198.51.100.0/24 TEST-NET-2
  [0xcb007100, 0xcb0071ff, 'reserved'], // 203.0.113.0/24 TEST-NET-3
  [0xe0000000, 0xefffffff, 'multicast'], // 224.0.0.0/4
  [0xf0000000, 0xffffffff, 'reserved'] // 240.0.0.0/4
]

// Cloud metadata endpoints (the #1 SSRF target), always blocked.
const METADATA_V4 = new Set(['169.254.169.254', '169.254.169.253', '169.254.170.2', '100.100.100.200'])

const IPV6_BLOCKED_RANGES: readonly (readonly [bigint, bigint, string])[] = [
  // Most specific first: AWS's IPv6 metadata address sits inside fc00::/7.
  [0xfd000ec2000000000000000000000254n, 0xfd000ec2000000000000000000000254n, 'metadata'], // AWS metadata (IPv6)
  [0x00000000000000000000000000000000n, 0x00000000000000000000000000000000n, 'unspecified'], // ::
  [0x00000000000000000000000000000001n, 0x00000000000000000000000000000001n, 'loopback'], // ::1
  [0xfe800000000000000000000000000000n, 0xfebfffffffffffffffffffffffffffffn, 'link-local'], // fe80::/10
  [0xfec00000000000000000000000000000n, 0xfeffffffffffffffffffffffffffffffn, 'private'], // fec0::/10 site-local (deprecated)
  [0xfc000000000000000000000000000000n, 0xfdffffffffffffffffffffffffffffffn, 'private'], // fc00::/7 ULA
  [0x20010db8000000000000000000000000n, 0x20010db8ffffffffffffffffffffffffn, 'reserved'], // 2001:db8::/32 doc
  [0xff000000000000000000000000000000n, 0xffffffffffffffffffffffffffffffffn, 'multicast'] // ff00::/8
]

function ipv4BlockReason(value: number): string | null {
  const text = `${(value >>> 24) & 0xff}.${(value >>> 16) & 0xff}.${(value >>> 8) & 0xff}.${value & 0xff}`

  if (METADATA_V4.has(text)) {
    return 'metadata'
  }

  return IPV4_BLOCKED_RANGES.find(([start, end]) => value >= start && value <= end)?.[2] ?? null
}

function ipv6BlockReason(value: bigint): string | null {
  // ::ffff:a.b.c.d (mapped) and ::ffff:0:a.b.c.d (translated) are IPv6
  // wrappers resolvers hand out for IPv4 names; both must classify as their
  // embedded address (tools/url_safety.py::_embedded_ipv4).
  if (value >> 32n === 0xffffn || value >> 48n === 0xffffn) {
    return ipv4BlockReason(Number(value & 0xffff_ffffn))
  }

  return IPV6_BLOCKED_RANGES.find(([start, end]) => value >= start && value <= end)?.[2] ?? null
}

/** Hostnames whose only useful answer is a cloud metadata service. */
const BLOCKED_TITLE_HOSTNAMES = new Set(['metadata.google.internal', 'metadata.goog'])

/**
 * Synchronous hostname classification — literal addresses and always-blocked
 * names. Returns the block reason, or null when the host is a public literal
 * or an unresolved name (DNS is `isSafeTitleFetchTarget`'s job).
 */
export function literalTitleBlockReason(hostname: string): string | null {
  const host = String(hostname || '')
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, '')
    .replace(/\.+$/, '')

  if (!host) {
    return 'empty-host'
  }

  if (BLOCKED_TITLE_HOSTNAMES.has(host)) {
    return 'metadata-host'
  }

  // mDNS names resolve into link-local space, and some resolvers answer them
  // from cache; refuse the suffix instead of racing it.
  if (host.endsWith('.local')) {
    return 'mdns'
  }

  // `.internal` is reserved for private networks (ICANN, 2024); cloud VPC DNS
  // (`*.ec2.internal`, `*.c.<project>.internal`) lives there.
  if (host.endsWith('.internal')) {
    return 'internal-host'
  }

  // `localhost` and its subdomains always mean loopback, so they classify
  // without DNS — that keeps the synchronous onBeforeRequest hook able to
  // cancel redirect hops to them, and a configured proxy from re-opening them.
  if (host === 'localhost' || host.endsWith('.localhost')) {
    return 'loopback-host'
  }

  const v4 = parseLooseIpv4(host)

  if (v4) {
    return ipv4BlockReason(((v4[0] << 24) | (v4[1] << 16) | (v4[2] << 8) | v4[3]) >>> 0)
  }

  const v6 = parseIpv6(host)

  if (v6 !== null) {
    return ipv6BlockReason(v6)
  }

  // A single-label name (`router`, `intranet`) only resolves through a LAN
  // search domain or a proxy's own network; no public site answers to one.
  return host.includes('.') ? null : 'single-label'
}

// The same narrow credential-param list as tools/url_safety.py
// (_SENSITIVE_QUERY_PARAM_NAMES): unambiguous bearers only — `code`, `key` and
// friends double as ordinary page facets, so they stay prefetchable.
// One deliberate widening over url_safety.py's exact names: any `*_token`
// parameter is refused here too (confirmation_token, reset_password_token,
// magic_token — the one-time links from #126885). A prefetch is passive, so
// consuming such a link breaks the user's later click; url_safety.py gates an
// explicit agent fetch, where the narrower list keeps ordinary pages reachable.
const SENSITIVE_TITLE_QUERY_PARAMS = new Set([
  'access_token',
  'api_key',
  'apikey',
  'auth_token',
  'authorization',
  'awsaccesskeyid',
  'client_secret',
  'credential',
  'credentials',
  'jwt',
  'password',
  'passwd',
  'secret',
  'session_id',
  'signature',
  'token',
  'x_amz_security_token',
  'x_amz_signature',
  'x-amz-security-token',
  'x-amz-signature'
])

/** First credential-named query parameter in the URL (with a value), else null. */
export function sensitiveTitleQueryParam(rawUrl: string): string | null {
  if (!rawUrl || !rawUrl.includes('?')) {
    return null
  }

  try {
    for (const [key, value] of new URL(rawUrl).searchParams) {
      const name = key.toLowerCase()

      if (value && (SENSITIVE_TITLE_QUERY_PARAMS.has(name) || name.endsWith('_token'))) {
        return key
      }
    }
  } catch {
    return null
  }

  return null
}

export type TitleHostLookup = (host: string) => Promise<{ address: string; family: number }[]>

/**
 * The env proxy the curl tier hands `url` to, or '' for a direct dial. The
 * curl tier passes the answer to curl explicitly (`--proxy`/`--noproxy`), so
 * curl's own env rules (lowercase-only `http_proxy`, `NO_PROXY`) can no longer
 * disagree with the guard about who dials the name. Either way the name's DNS
 * answers are vetted first; only the pin (`--resolve`) is direct-dial only.
 */
export function titleProxyFor(url: URL, env: NodeJS.ProcessEnv = process.env): string {
  const names =
    url.protocol === 'https:'
      ? ['https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
      : ['http_proxy', 'HTTP_PROXY', 'all_proxy', 'ALL_PROXY']

  for (const name of names) {
    const value = env[name]?.trim()

    if (value) {
      return value
    }
  }

  return ''
}

async function defaultLookup(host: string): Promise<{ address: string; family: number }[]> {
  return lookup(host, { all: true })
}

/** A title/metadata destination that passed admission. */
export interface TitleFetchTarget {
  /** Every DNS answer for `hostname`, all vetted (also for a proxied dial, which cannot pin them); [] for a literal IP. */
  addresses: string[]
  hostname: string
  port: string
  /** The env proxy that owns the dial ('' = direct). */
  proxy: string
}

const DEFAULT_DNS_TIMEOUT_MS = 1_500

async function withinDeadline<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('DNS lookup timed out')), timeoutMs)
      })
    ])
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}

/**
 * Admission plus the vetted answers. Null means refused (fail closed): bad
 * scheme, credential-bearing query, blocked literal/special name, a lookup
 * that fails or times out, or any answer in a blocked range. `honorProxyEnv`
 * is for the curl tier only; Chromium-stack callers pass false because
 * Chromium does not route through these env vars, so the name is dialed here.
 */
export async function resolveTitleFetchTarget(
  rawUrl: string,
  options: { dnsTimeoutMs?: number; honorProxyEnv?: boolean; lookup?: TitleHostLookup } = {}
): Promise<TitleFetchTarget | null> {
  let url: URL

  try {
    url = new URL(rawUrl)
  } catch {
    return null
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return null
  }

  if (sensitiveTitleQueryParam(rawUrl) || literalTitleBlockReason(url.hostname)) {
    return null
  }

  const hostname = url.hostname.replace(/^\[|\]$/g, '')
  const port = url.port || (url.protocol === 'https:' ? '443' : '80')
  const proxy = options.honorProxyEnv === false ? '' : titleProxyFor(url)

  // A public literal cannot rebind; there is nothing to resolve or pin.
  if (parseLooseIpv4(hostname) || parseIpv6(hostname) !== null) {
    return { addresses: [], hostname, port, proxy }
  }

  // A name is resolved and vetted here even when a proxy will dial it. The
  // proxy resolves it again and could still reach a private address on its own
  // network, but `evil.example.com -> 169.254.169.254` must not be admitted just
  // because this host did not look: "admitted" means "looked, and every answer
  // is public". A name this host cannot resolve is refused like a direct one.

  let answers: { address: string; family: number }[]

  try {
    answers = await withinDeadline(
      Promise.resolve((options.lookup ?? defaultLookup)(hostname)),
      Math.max(1, options.dnsTimeoutMs ?? DEFAULT_DNS_TIMEOUT_MS)
    )
  } catch {
    return null
  }

  if (!answers.length || answers.some(entry => literalTitleBlockReason(entry.address))) {
    return null
  }

  return { addresses: answers.map(entry => entry.address), hostname, port, proxy }
}

/**
 * Full admission for one title-fetch target: scheme, credential-bearing query,
 * literal/special hostname, and every DNS answer (with or without a proxy). False means "no title", fail closed: a name
 * that fails to resolve, or resolves into any blocked range, never reaches
 * curl or the hidden title window.
 */
export async function isSafeTitleFetchTarget(
  rawUrl: string,
  lookupHost: TitleHostLookup = defaultLookup
): Promise<boolean> {
  return (await resolveTitleFetchTarget(rawUrl, { lookup: lookupHost })) !== null
}

export type LinkMetadataPolicy = (rawUrl: string) => Promise<boolean>

/** The same admission for Chromium-stack fetches: always resolved and vetted here. */
export function createLinkMetadataPolicy(
  options: { dnsTimeoutMs?: number; lookup?: TitleHostLookup } = {}
): LinkMetadataPolicy {
  return async rawUrl => (await resolveTitleFetchTarget(rawUrl, { ...options, honorProxyEnv: false })) !== null
}

export const isSafeLinkMetadataTarget: LinkMetadataPolicy = createLinkMetadataPolicy()
