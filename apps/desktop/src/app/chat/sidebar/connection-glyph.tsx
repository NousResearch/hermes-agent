import type { DesktopRegistryConnection } from '@/global'
import { Cloud, Monitor, Network, Terminal } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { FleetGatewayCondition } from './fleet-rail'
import { SIDEBAR_ROW_LEAD } from './row-geometry'

// One glyph per connection kind — device, cloud, network, terminal — shared by
// the statusbar switcher, its menu, the fleet profile rail and the Bots rail so
// a gateway looks the same wherever it is named. Dependency-free on purpose
// (icons, a type and class strings) so light components can use it without
// pulling in stores.
export function ConnectionGlyph({
  className,
  connection
}: {
  className?: string
  connection: Pick<DesktopRegistryConnection, 'kind'>
}) {
  const Icon =
    connection.kind === 'local'
      ? Monitor
      : connection.kind === 'cloud'
        ? Cloud
        : connection.kind === 'ssh'
          ? Terminal
          : Network

  return (
    <span
      aria-hidden="true"
      className={cn(SIDEBAR_ROW_LEAD, 'text-(--ui-text-quaternary)', className)}
      data-connection-kind={connection.kind}
      data-slot="connection-glyph"
    >
      <Icon className="size-3" />
    </span>
  )
}

// The status dot beside that glyph, shared by every surface that lists a
// gateway so one colour always means one thing: amber = the roster dialed it
// and got nothing back; muted = it was deliberately not dialed (ssh before
// first use, the local runtime under a remote primary) and a click starts it.
// Painting the second amber reports a healthy machine as broken.
export function GatewayConditionDot({
  className,
  condition
}: {
  className?: string
  condition: FleetGatewayCondition
}) {
  if (condition === 'ready') {
    return null
  }

  return (
    <span
      aria-hidden="true"
      className={cn(
        'size-1.5 shrink-0 rounded-full',
        condition === 'down' ? 'bg-amber-500' : 'bg-(--ui-text-tertiary)',
        className
      )}
      data-condition={condition}
      data-slot="gateway-condition-dot"
    />
  )
}
