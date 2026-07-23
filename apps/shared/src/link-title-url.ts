const LOCAL_ONLY_HOST_SUFFIXES = ['home.arpa', 'internal', 'lan', 'local', 'localdomain', 'localhost']
const PUBLIC_IPV4_EXCEPTIONS = new Set([0xc0000009, 0xc000000a])

const NON_PUBLIC_IPV4_RANGES: ReadonlyArray<readonly [network: number, prefixLength: number]> = [
  [0x00000000, 8],
  [0x0a000000, 8],
  [0x64400000, 10],
  [0x7f000000, 8],
  [0xa9fe0000, 16],
  [0xac100000, 12],
  [0xc0000000, 24],
  [0xc0000200, 24],
  [0xc0586300, 24],
  [0xc0a80000, 16],
  [0xc6120000, 15],
  [0xc6336400, 24],
  [0xcb007100, 24],
  [0xe0000000, 4],
  [0xf0000000, 4]
]

const PUBLIC_IETF_IPV6_RANGES: ReadonlyArray<readonly [network: readonly number[], prefixLength: number]> = [
  [[0x2001, 0x0001, 0, 0, 0, 0, 0, 1], 128],
  [[0x2001, 0x0001, 0, 0, 0, 0, 0, 2], 128],
  [[0x2001, 0x0001, 0, 0, 0, 0, 0, 3], 128],
  [[0x2001, 0x0003], 32],
  [[0x2001, 0x0004, 0x0112], 48],
  [[0x2001, 0x0020], 28],
  [[0x2001, 0x0030], 28]
]

const NON_PUBLIC_GLOBAL_UNICAST_IPV6_RANGES: ReadonlyArray<
  readonly [network: readonly number[], prefixLength: number]
> = [
  [[0x2001, 0x0db8], 32],
  [[0x2002], 16],
  [[0x3fff, 0], 20]
]

function canonicalIpv4Value(hostname: string): null | number {
  const octets = hostname.split('.')

  if (octets.length !== 4 || octets.some(octet => !/^\d+$/.test(octet))) {
    return null
  }

  const values = octets.map(Number)

  if (values.some(value => value < 0 || value > 255)) {
    return null
  }

  return (((values[0] * 256 + values[1]) * 256 + values[2]) * 256 + values[3]) >>> 0
}

function isInIpv4Range(value: number, network: number, prefixLength: number): boolean {
  const mask = prefixLength === 0 ? 0 : (0xffffffff << (32 - prefixLength)) >>> 0

  return (value & mask) >>> 0 === (network & mask) >>> 0
}

function isNonPublicIpv4(value: number): boolean {
  if (PUBLIC_IPV4_EXCEPTIONS.has(value)) {
    return false
  }

  return NON_PUBLIC_IPV4_RANGES.some(([network, prefixLength]) => isInIpv4Range(value, network, prefixLength))
}

function parseIpv6Section(section: string): null | number[] {
  if (!section) {
    return []
  }

  const words = section.split(':')

  if (words.some(word => !/^[0-9a-f]{1,4}$/i.test(word))) {
    return null
  }

  return words.map(word => Number.parseInt(word, 16))
}

function canonicalIpv6Words(hostname: string): null | number[] {
  if (!hostname.startsWith('[') || !hostname.endsWith(']')) {
    return null
  }

  const address = hostname.slice(1, -1)
  const halves = address.split('::')

  if (halves.length > 2) {
    return null
  }

  const left = parseIpv6Section(halves[0])
  const right = parseIpv6Section(halves[1] ?? '')

  if (!left || !right) {
    return null
  }

  const omitted = 8 - left.length - right.length

  if ((halves.length === 1 && omitted !== 0) || (halves.length === 2 && omitted < 1)) {
    return null
  }

  return [...left, ...Array.from({ length: omitted }, () => 0), ...right]
}

function isInIpv6Range(words: number[], network: readonly number[], prefixLength: number): boolean {
  const wholeWords = Math.floor(prefixLength / 16)

  for (let index = 0; index < wholeWords; index += 1) {
    if (words[index] !== (network[index] ?? 0)) {
      return false
    }
  }

  const remainingBits = prefixLength % 16

  if (!remainingBits) {
    return true
  }

  const mask = (0xffff << (16 - remainingBits)) & 0xffff

  return (words[wholeWords] & mask) === ((network[wholeWords] ?? 0) & mask)
}

function embeddedIpv4(words: number[]): null | number {
  const isMapped = words.slice(0, 5).every(word => word === 0) && words[5] === 0xffff
  const isNat64Wkp =
    words[0] === 0x0064 && words[1] === 0xff9b && words.slice(2, 6).every(word => word === 0)

  if (!isMapped && !isNat64Wkp) {
    return null
  }

  return ((words[6] << 16) | words[7]) >>> 0
}

function isPublicIpv6(words: number[]): boolean {
  const ipv4 = embeddedIpv4(words)

  if (ipv4 !== null) {
    return !isNonPublicIpv4(ipv4)
  }

  if (!isInIpv6Range(words, [0x2000], 3)) {
    return false
  }

  if (isInIpv6Range(words, [0x2001], 23)) {
    return PUBLIC_IETF_IPV6_RANGES.some(([network, prefixLength]) => isInIpv6Range(words, network, prefixLength))
  }

  return !NON_PUBLIC_GLOBAL_UNICAST_IPV6_RANGES.some(([network, prefixLength]) =>
    isInIpv6Range(words, network, prefixLength)
  )
}

function isLocalOnlyHostname(hostname: string): boolean {
  if (!hostname.includes('.')) {
    return true
  }

  return LOCAL_ONLY_HOST_SUFFIXES.some(suffix => hostname === suffix || hostname.endsWith(`.${suffix}`))
}

function unbracketAddress(value: string): null | string {
  const startsWithBracket = value.startsWith('[')
  const endsWithBracket = value.endsWith(']')

  if (startsWithBracket !== endsWithBracket) {
    return null
  }

  return startsWithBracket ? value.slice(1, -1) : value
}

export function isPublicLinkTitleAddress(value: string): boolean {
  const address = unbracketAddress(value)

  if (address === null) {
    return false
  }

  const ipv4 = canonicalIpv4Value(address)

  if (ipv4 !== null) {
    return !isNonPublicIpv4(ipv4)
  }

  const ipv6 = canonicalIpv6Words(`[${address}]`)

  return ipv6 !== null && isPublicIpv6(ipv6)
}

export function admitLinkTitleUrl(value: string): null | string {
  let url: URL

  try {
    url = new URL(value)
  } catch {
    return null
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return null
  }

  const hostname = url.hostname.toLowerCase().replace(/\.$/, '')
  const ipv4 = canonicalIpv4Value(hostname)
  const ipv6 = canonicalIpv6Words(hostname)

  if ((ipv4 !== null || ipv6) && !isPublicLinkTitleAddress(hostname)) {
    return null
  }

  if (ipv4 === null && !ipv6 && isLocalOnlyHostname(hostname)) {
    return null
  }

  return url.href
}

export function isLinkTitleFetchableUrl(value: string): boolean {
  return admitLinkTitleUrl(value) !== null
}
