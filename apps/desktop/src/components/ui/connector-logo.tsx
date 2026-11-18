import { brandFor } from '@/lib/mcp-brands'
import { cn } from '@/lib/utils'

import { AvatarChip, monogramFor } from './avatar-chip'
import { Favicon } from './favicon'

/** The least a mark needs: something to name it, and somewhere a logo might
 *  live. Structural on purpose — a caller's richer connector type satisfies
 *  this without the visual layer knowing that type exists. */
export interface ConnectorLogoSubject {
  docs?: string
  homepage?: string
  /** The vendor's own mark, served as a public SVG; connector rows always carry one. */
  iconUrl?: string
  name: string
  title?: string
  url?: null | string
}

/** Hosts whose favicon would name the wrong thing: a bridge published on
 *  GitHub is not GitHub, and a package page is not the product. */
const NOT_A_LOGO =
  /(^|\.)(github\.com|githubusercontent\.com|gitlab\.com|bitbucket\.org|npmjs\.com|pypi\.org|readthedocs\.io)$/

const isLoopback = (host: string) =>
  host === 'localhost' || host === '::1' || /^127\./.test(host) || /^(10|192\.168)\./.test(host)

/**
 * Where to read this subject's mark from.
 *
 * The product's own site first (the only source that is certainly the right
 * logo), then the endpoint it talks to — `mcp.linear.app` is Linear — then
 * vendor docs, which for most entries is `docs.stripe.com` and friends.
 * Nothing for a private or loopback host: there is no logo out there to find,
 * and asking would announce an internal hostname.
 */
export function connectorLogoSource(subject: ConnectorLogoSubject): string {
  for (const candidate of [subject.homepage, subject.url, subject.docs]) {
    if (!candidate) {
      continue
    }

    try {
      const { hostname, origin } = new URL(candidate)

      if (!NOT_A_LOGO.test(hostname) && !isLoopback(hostname)) {
        // The origin, never the path: the icon lives on the site, and this
        // way a not-yet-connected endpoint is never fetched just to draw a
        // logo.
        return origin
      }
    } catch {
      // Not a URL (a bare repo path, a note) — nothing to read a mark from.
    }
  }

  return ''
}

/**
 * A connector's mark, resolved as far as it goes.
 *
 * Curated brand glyph → the vendor's icon → the product's own favicon → the
 * monogram every other unknown name in the app falls back to. The middle rungs
 * are what keep the long tail from all looking alike: a curated icon set is a
 * couple dozen names and a public registry is thousands, so something we ship
 * no icon for still arrives wearing its own logo.
 *
 * The vendor icon is a plain image: its host sends no CORS header, so nothing
 * here may read it, and an unknown slug answers a grey placeholder with a 200,
 * so a load event never tells us whether a logo exists.
 */
export function ConnectorLogo({ className, connector }: { className?: string; connector: ConnectorLogoSubject }) {
  const label = connector.title || connector.name
  const brand = brandFor(connector.name)
  const icon = brand ? '' : (connector.iconUrl ?? '')
  const site = brand || icon ? '' : connectorLogoSource(connector)

  return (
    <AvatarChip
      brand={brand}
      className={cn((icon || site) && 'overflow-hidden', className)}
      name={label}
      title={connector.title}
    >
      {icon ? (
        <img alt="" aria-hidden className="size-full object-contain" src={icon} />
      ) : site ? (
        <Favicon fallback={monogramFor(label)} url={site} />
      ) : undefined}
    </AvatarChip>
  )
}
