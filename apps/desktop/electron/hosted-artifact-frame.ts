/**
 * Which origins may host a server-hosted artifact frame.
 *
 * A hosted artifact is served from its own origin, one DNS label per artifact
 * under a dedicated frame domain (`https://<artifact id>.<frame domain>/`), so
 * two artifacts can never read each other's DOM or storage. The desktop frames
 * that origin with `allow-same-origin`, which is only safe because the origin
 * is a throwaway subdomain owned by the artifact and nothing else. Every other
 * URL a tool result names is shown as a link, never framed.
 *
 * Shared by the renderer (decides whether to frame) and Electron (refuses any
 * subframe navigation that would put a non-artifact document on an artifact
 * origin, or move an artifact frame anywhere else). Pure, no Electron import.
 */

/**
 * The production frame domain. EMPTY means "never frame, always link": until
 * a frame domain is provisioned, hosted artifacts render as a link card to the
 * viewer page in packaged builds. Set this (and only this) to the bare domain,
 * without scheme or leading dot, once it exists.
 */
export const HOSTED_ARTIFACT_FRAME_DOMAIN = ''

/** The local frame domain the artifacts server uses in development. Honoured
 *  only by development builds, over plain http on any port. */
export const HOSTED_ARTIFACT_DEV_FRAME_DOMAIN = 'artifacts.localhost'

/** An artifact id: 26 lowercase Crockford base32 characters (no i, l, o, u). */
const ARTIFACT_ID_RE = /^[0-9a-hjkmnp-tv-z]{26}$/

export interface HostedArtifactFramePolicy {
  /** Development build: also accept `http://<id>.artifacts.localhost:<port>`. */
  dev: boolean
  /** Production frame domain override for tests; defaults to the constant. */
  frameDomain?: string
}

/** The policy of this build: the development frame domain only in unpackaged runs. */
export function hostedArtifactFramePolicy(isPackaged: boolean): HostedArtifactFramePolicy {
  return { dev: !isPackaged }
}

export function isHostedArtifactId(value: unknown): value is string {
  return typeof value === 'string' && ARTIFACT_ID_RE.test(value)
}

function parse(url: string): URL | null {
  try {
    return new URL(url)
  } catch {
    return null
  }
}

/** The artifact id a URL's host names under an allowed frame domain, or null.
 *  Exactly one label above the frame domain, and it must be an artifact id. */
function artifactLabel(parsed: URL, policy: HostedArtifactFramePolicy): string | null {
  if (parsed.username || parsed.password) {
    return null
  }

  const host = parsed.hostname.toLowerCase()
  const domain = (policy.frameDomain ?? HOSTED_ARTIFACT_FRAME_DOMAIN).trim().toLowerCase()

  const under = (suffix: string): string | null => {
    if (!suffix || !host.endsWith(`.${suffix}`)) {
      return null
    }

    const label = host.slice(0, -(suffix.length + 1))

    return isHostedArtifactId(label) ? label : null
  }

  if (parsed.protocol === 'https:' && parsed.port === '') {
    const label = under(domain)

    if (label) {
      return label
    }
  }

  if (policy.dev && parsed.protocol === 'http:') {
    return under(HOSTED_ARTIFACT_DEV_FRAME_DOMAIN)
  }

  return null
}

/**
 * The origin to frame for a hosted artifact's `content_url`, or null when it
 * must not be framed. When `artifactId` is given, the host label must be that
 * id: a result for artifact A can never frame artifact B's origin.
 */
export function hostedArtifactFrameOrigin(
  contentUrl: unknown,
  policy: HostedArtifactFramePolicy,
  artifactId?: string
): string | null {
  if (typeof contentUrl !== 'string') {
    return null
  }

  const parsed = parse(contentUrl.trim())

  if (!parsed) {
    return null
  }

  const label = artifactLabel(parsed, policy)

  if (!label || (artifactId !== undefined && label !== artifactId.toLowerCase())) {
    return null
  }

  return parsed.origin
}

/** True when a URL sits anywhere under a frame domain this build knows of,
 *  valid artifact label or not. Used to fence the whole namespace off. */
function underAnyFrameDomain(parsed: URL, policy: HostedArtifactFramePolicy): boolean {
  const host = parsed.hostname.toLowerCase()

  const domains = [
    (policy.frameDomain ?? HOSTED_ARTIFACT_FRAME_DOMAIN).trim().toLowerCase(),
    HOSTED_ARTIFACT_DEV_FRAME_DOMAIN
  ]

  return domains.some(domain => domain !== '' && (host === domain || host.endsWith(`.${domain}`)))
}

/**
 * May a subframe navigate from `fromOrigin` to `toUrl`? Only decides for the
 * artifact namespace; every other navigation is `null` (not ours, existing
 * rules apply).
 *
 * - An artifact frame stays on its own origin (the ticket redirect to `/` and
 *   in-artifact links are same-origin); anything else is denied.
 * - A frame may enter an artifact origin only when this build allows framing
 *   that origin (so packaged builds never load the development domain, and an
 *   unprovisioned production domain frames nothing).
 */
export function hostedArtifactFrameNavigation(
  fromOrigin: string | null,
  toUrl: string,
  policy: HostedArtifactFramePolicy
): 'allow' | 'deny' | null {
  const target = parse(toUrl)
  const fromParsed = fromOrigin ? parse(fromOrigin) : null
  const fromArtifact = fromParsed !== null && underAnyFrameDomain(fromParsed, policy)
  const toArtifact = target !== null && underAnyFrameDomain(target, policy)

  if (!fromArtifact && !toArtifact) {
    return null
  }

  if (!target) {
    return 'deny'
  }

  const allowedTarget = hostedArtifactFrameOrigin(toUrl, policy)

  if (fromArtifact && fromParsed!.origin !== target.origin) {
    return 'deny'
  }

  return allowedTarget === target.origin ? 'allow' : 'deny'
}

/** True when a permission request comes from anywhere in the artifact
 *  namespace. Artifact frames are granted no permissions at all. */
export function isHostedArtifactRequestingUrl(
  url: string | null | undefined,
  policy: HostedArtifactFramePolicy
): boolean {
  const parsed = url ? parse(url) : null

  return parsed !== null && underAnyFrameDomain(parsed, policy)
}
