import type { DesktopRegistryConnection } from '@/global'
import { Cloud, Monitor, Network, Terminal } from '@/lib/icons'

// One glyph per connection kind — device, cloud, network, terminal — shared by
// the statusbar switcher, its menu, and the fleet profile rail so a gateway
// looks the same wherever it is named. Dependency-free on purpose (icons and
// a type only) so light components can use it without pulling in stores.
export function ConnectionGlyph({ connection }: { connection: Pick<DesktopRegistryConnection, 'kind'> }) {
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
      className="grid size-3.5 shrink-0 place-items-center text-(--ui-text-quaternary)"
      data-connection-kind={connection.kind}
      data-slot="connection-glyph"
    >
      <Icon className="size-3" />
    </span>
  )
}
