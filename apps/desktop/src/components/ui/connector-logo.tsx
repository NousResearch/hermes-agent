import { brandFor } from '@/lib/mcp-brands'
import { type Connector, connectorLogoSource } from '@/lib/mcp-connectors'
import { cn } from '@/lib/utils'

import { AvatarChip, monogramFor } from './avatar-chip'
import { Favicon } from './favicon'

/**
 * A connector's mark, resolved as far as it goes.
 *
 * Curated brand glyph → the product's own favicon → the monogram every other
 * unknown name in the app falls back to. The middle rung is what keeps the
 * long tail from all looking alike: the catalog is a couple dozen names and
 * the registry is thousands, so a connector we ship no icon for still arrives
 * wearing its own logo. Which site to read that favicon from is
 * `connectorLogoSource`'s call, not this component's.
 */
export function ConnectorLogo({
  className,
  connector
}: {
  className?: string
  // Name and title are the identity; the URLs are what a favicon lookup needs
  // and a settled transcript row no longer has.
  connector: Partial<Pick<Connector, 'docs' | 'homepage' | 'url'>> & Pick<Connector, 'name' | 'title'>
}) {
  const brand = brandFor(connector.name)
  const site = brand ? '' : connectorLogoSource(connector)

  return (
    <AvatarChip
      brand={brand}
      className={cn(site && 'overflow-hidden', className)}
      name={connector.title || connector.name}
      title={connector.title}
    >
      {site ? <Favicon fallback={monogramFor(connector.title || connector.name)} url={site} /> : undefined}
    </AvatarChip>
  )
}
