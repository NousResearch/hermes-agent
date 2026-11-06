import { getPlayerService, type PlayerServiceDefinition, type PlayerSource } from '../../src/lib/player-services'

export type FixtureOrigin = { hostname: '127.0.0.1'; port: string }

function parsed(raw: string): URL | null {
  try {
    return new URL(raw)
  } catch {
    return null
  }
}

export function isAllowedPlayerNavigation(
  raw: string,
  fixture?: FixtureOrigin,
  source: PlayerSource = 'youtube'
): boolean {
  const url = parsed(raw)

  if (!url) {
    return false
  }

  if (url.protocol === 'https:') {
    return isAllowedPlayerServiceHost(url.hostname, source) || isAllowedPlayerAuthHost(url.hostname, source)
  }

  return Boolean(fixture && url.protocol === 'http:' && url.hostname === fixture.hostname && url.port === fixture.port)
}

export function isAllowedPlayerFrame(raw: string, fixture?: FixtureOrigin, source: PlayerSource = 'youtube'): boolean {
  const url = parsed(raw)

  if (!url) {
    return false
  }

  if (url.protocol === 'https:') {
    return isAllowedHttpsFrame(url, source)
  }

  if (url.protocol === 'about:') {
    return url.href === 'about:blank'
  }

  if (url.protocol === 'blob:') {
    return isAllowedBlobFrame(url, fixture, source)
  }

  return isAllowedFixtureFrame(url, fixture)
}

function isAllowedHttpsFrame(url: URL, source: PlayerSource): boolean {
  const service: PlayerServiceDefinition = getPlayerService(source)

  const embeddedAuth =
    !url.port &&
    !url.username &&
    !url.password &&
    service.authFrameRules?.some(rule => url.hostname === rule.hostname && url.pathname.startsWith(rule.pathnamePrefix))

  return (
    source === 'youtube' ||
    source === 'tv' ||
    isAllowedPlayerServiceHost(url.hostname, source) ||
    isAllowedPlayerAuthHost(url.hostname, source) ||
    Boolean(embeddedAuth)
  )
}

function isAllowedBlobFrame(url: URL, fixture: FixtureOrigin | undefined, source: PlayerSource): boolean {
  const origin = parsed(url.origin)

  if (!origin) {
    return false
  }

  if (origin.protocol === 'https:') {
    return (
      source === 'youtube' ||
      source === 'tv' ||
      isAllowedPlayerServiceHost(origin.hostname, source) ||
      isAllowedPlayerAuthHost(origin.hostname, source)
    )
  }

  return Boolean(
    fixture && origin.protocol === 'http:' && origin.hostname === fixture.hostname && origin.port === fixture.port
  )
}

function isAllowedFixtureFrame(url: URL, fixture?: FixtureOrigin): boolean {
  return Boolean(fixture && url.protocol === 'http:' && url.hostname === fixture.hostname && url.port === fixture.port)
}

export function isAllowedPlayerPopup(raw: string, source: PlayerSource = 'youtube'): boolean {
  const url = parsed(raw)

  return Boolean(
    url &&
    url.protocol === 'https:' &&
    (isAllowedPlayerServiceHost(url.hostname, source) || isAllowedPlayerAuthHost(url.hostname, source))
  )
}

export function safePlayerOrigin(
  raw: string,
  fixture?: FixtureOrigin,
  source: PlayerSource = 'youtube'
): string | null {
  const url = parsed(raw)

  if (!url) {
    return null
  }

  if (url.protocol === 'https:' && isAllowedPlayerServiceHost(url.hostname, source)) {
    return url.origin
  }

  if (fixture && url.protocol === 'http:' && url.hostname === fixture.hostname && url.port === fixture.port) {
    return `http://${fixture.hostname}:${fixture.port}`
  }

  return null
}

export function isAllowedPlayerServiceHost(hostname: string, source: PlayerSource): boolean {
  return (getPlayerService(source).serviceHosts as readonly string[]).includes(hostname)
}

export function isAllowedPlayerAuthHost(hostname: string, source: PlayerSource): boolean {
  return (getPlayerService(source).authHosts as readonly string[]).includes(hostname)
}

export function isAllowedPlayerMediaKeyHost(hostname: string, source: PlayerSource): boolean {
  return (getPlayerService(source).mediaKeyHosts as readonly string[]).includes(hostname)
}

export function clampPlayerBounds(
  bounds: { x: number; y: number; width: number; height: number },
  contentSize: { width: number; height: number }
): { x: number; y: number; width: number; height: number } {
  const rawX = Math.round(bounds.x)
  const rawY = Math.round(bounds.y)
  const rawRight = Math.round(bounds.x + bounds.width)
  const rawBottom = Math.round(bounds.y + bounds.height)
  const x = Math.max(0, Math.min(contentSize.width, rawX))
  const y = Math.max(0, Math.min(contentSize.height, rawY))
  const right = Math.max(x, Math.min(contentSize.width, rawRight))
  const bottom = Math.max(y, Math.min(contentSize.height, rawBottom))

  return { x, y, width: right - x, height: bottom - y }
}
