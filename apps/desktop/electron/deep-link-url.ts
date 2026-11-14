/**
 * Parse an inbound `hermes://<kind>/<name>?<params>` deep link into the
 * payload the renderer routes on. The link comes from outside the app (a web
 * page, another program, argv), so a malformed one is refused with a reason
 * instead of throwing out of the open-url / second-instance handlers. The
 * reason never quotes the link: its path and query can carry tokens.
 */

export interface DeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

/** The payload, or null after telling `reject` why the link was refused. */
export function parseDeepLink(
  url: string,
  schemes: readonly string[],
  reject: (reason: string) => void
): DeepLinkPayload | null {
  let parsed: URL

  try {
    parsed = new URL(url)
  } catch {
    reject('ignoring malformed url')

    return null
  }

  const scheme = parsed.protocol.replace(/:$/, '')

  if (!schemes.includes(scheme)) {
    reject(`ignoring scheme ${scheme} (expected ${schemes.join(' or ')})`)

    return null
  }

  // hermes://blueprint/<key>?slot=val  -> host="blueprint", path="/<key>"
  const kind = parsed.hostname || ''
  let name: string

  try {
    name = decodeURIComponent((parsed.pathname || '').replace(/^\//, ''))
  } catch {
    // A stray or truncated percent escape (`%E0%A4%A`): URIError.
    reject(`ignoring ${kind} link with a malformed path`)

    return null
  }

  const params: Record<string, string> = {}
  parsed.searchParams.forEach((value, key) => {
    params[key] = value
  })

  return { kind, name, params }
}
