export interface HermesDeepLinkPayload {
  kind: 'blueprint' | 'profile'
  name: string
  params: Record<string, string>
}

const HERMES_PROTOCOL = 'hermes:'
const PROFILE_LINK_RE = /^hermes:\/\/profile\/([^/?#]+)\?([^#]*)$/i
const PROFILE_NAME_RE = /^(?:default|[a-z0-9][a-z0-9_-]{0,63})$/

export function extractHermesDeepLink(argv: unknown): null | string {
  if (!Array.isArray(argv)) {
    return null
  }

  return argv.find(value => typeof value === 'string' && value.startsWith('hermes://')) ?? null
}

/**
 * Parse the public hermes:// surface at the native protocol boundary.
 *
 * Profile links are deliberately narrower than blueprint links: they carry
 * only an installed profile identifier plus the explicit new-chat intent.
 * They cannot transport prompt text or another action for the renderer to run.
 */
export function parseHermesDeepLink(raw: unknown): HermesDeepLinkPayload | null {
  if (typeof raw !== 'string') {
    return null
  }

  let parsed: URL

  try {
    parsed = new URL(raw)
  } catch {
    return null
  }

  if (parsed.protocol !== HERMES_PROTOCOL) {
    return null
  }

  const kind = parsed.hostname.toLowerCase()

  if (kind === 'profile') {
    const match = PROFILE_LINK_RE.exec(raw)

    if (!match) {
      return null
    }

    let name: string

    try {
      name = decodeURIComponent(match[1])
    } catch {
      return null
    }

    const keys = [...parsed.searchParams.keys()]

    if (
      !PROFILE_NAME_RE.test(name) ||
      keys.length !== 1 ||
      keys[0] !== 'new' ||
      parsed.searchParams.getAll('new').length !== 1 ||
      parsed.searchParams.get('new') !== '1'
    ) {
      return null
    }

    return { kind: 'profile', name, params: { new: '1' } }
  }

  if (kind !== 'blueprint') {
    return null
  }

  let name: string

  try {
    name = decodeURIComponent((parsed.pathname || '').replace(/^\//, ''))
  } catch {
    return null
  }

  const params: Record<string, string> = {}
  parsed.searchParams.forEach((value, key) => {
    params[key] = value
  })

  return { kind: 'blueprint', name, params }
}
