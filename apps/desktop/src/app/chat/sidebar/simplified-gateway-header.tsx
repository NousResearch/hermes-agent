import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { cn } from '@/lib/utils'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $gatewayState } from '@/store/session'

export function SimplifiedGatewayHeader({
  connectionId,
  fallbackLabel,
  selected,
  open,
  onToggle
}: {
  connectionId: string | null
  fallbackLabel: string
  selected: boolean
  open: boolean
  onToggle?: () => void
}) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const roster = useStore($fleetRoster)
  const gatewayState = useStore($gatewayState)
  const source = roster?.sources.find(source => source.connectionId === connectionId)
  const registered = registry?.connections.find(connection => connection.id === connectionId)

  const address =
    registered?.url && URL.canParse(registered.url)
      ? new URL(registered.url).host
      : registered?.host || (selected && isBrowserHostedDesktop() ? window.location.host : '')

  const label = registered?.label || (address === fallbackLabel ? t.settings.gateway.webappHostTitle : fallbackLabel)

  const state = selected ? gatewayState : source ? (source.reachable ? 'open' : 'closed') : 'idle'
  const connected = state === 'open'

  const status = {
    open: t.shell.gatewayMenu.connected,
    connecting: t.shell.gatewayMenu.connecting,
    idle: t.common.loading,
    closed: t.shell.gatewayMenu.offline,
    error: t.shell.gatewayMenu.offline
  }[state]

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={label}
      className={cn(
        'flex min-h-8.5 items-center gap-2 px-2 py-1.5 text-xs',
        selected && 'bg-(--chrome-action-hover)',
        selected && open ? 'mb-1 rounded-t-md' : 'rounded-md'
      )}
      role="group"
    >
      {onToggle && (
        <Button aria-expanded={open} aria-label={label} onClick={onToggle} size="icon-xs" variant="grip">
          <Codicon name={open ? 'chevron-down' : 'chevron-right'} size={12} />
        </Button>
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium text-(--ui-text-primary)" title={label}>
          {label}
        </span>
        {address && address !== label && (
          <span className="block truncate text-[0.6875rem] text-(--ui-text-tertiary)" title={address}>
            {address}
          </span>
        )}
      </span>
      <span
        className={cn(
          'flex shrink-0 items-center gap-1 text-[0.6875rem]',
          connected ? 'text-green-600 dark:text-green-400' : 'text-(--ui-text-tertiary)'
        )}
      >
        <span aria-hidden className="size-1.5 rounded-full bg-current" />
        {status}
      </span>
    </div>
  )
}
