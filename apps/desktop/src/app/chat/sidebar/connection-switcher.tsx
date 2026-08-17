import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Cloud, Loader2, Monitor, Network, Terminal } from '@/lib/icons'
import {
  $activeConnectionId,
  $connectionsRegistry,
  $pendingConnectionId,
  refreshConnectionsRegistry,
  selectConnection
} from '@/store/connections'
import { notifyError } from '@/store/notifications'

export function ConnectionSwitcher({ onConnect }: { onConnect: () => void }) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const pendingConnectionId = useStore($pendingConnectionId)

  useEffect(() => {
    void refreshConnectionsRegistry().catch(() => undefined)

    // Registry events are local IPC notifications, not remote polling. They
    // keep a second Settings window or a removal/edit reflected here.
    const off = window.hermesDesktop?.connections?.onChanged?.(() => {
      void refreshConnectionsRegistry().catch(() => undefined)
    })

    return off
  }, [])

  const connections = registry?.connections ?? []

  if (connections.length <= 1) {
    return null
  }

  const activeConnection = connections.find(connection => connection.id === activeConnectionId)

  const choose = (connectionId: string) => {
    triggerHaptic('selection')
    const connection = connections.find(candidate => candidate.id === connectionId)

    void selectConnection(connectionId).catch(error =>
      notifyError(error, t.profiles.switchConnectionFailed(connection?.label ?? connectionId))
    )
  }

  return (
    <div
      aria-busy={pendingConnectionId !== null}
      aria-label={t.settings.connections.title}
      className="min-w-20 max-w-[46%] shrink overflow-hidden"
      data-slot="connection-switcher"
      role="group"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={
              activeConnection
                ? `${t.settings.connections.title}: ${activeConnection.label}`
                : t.settings.connections.title
            }
            className="w-full min-w-0 justify-between overflow-hidden px-1 text-(--ui-text-secondary) data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground"
            size="xs"
            type="button"
            variant="ghost"
          >
            <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
              {pendingConnectionId && <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />}
              {activeConnection ? (
                <ConnectionLabel connection={activeConnection} />
              ) : (
                <span className="truncate">{t.settings.connections.title}</span>
              )}
            </span>
            <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-52 max-w-72" collisionPadding={8} side="top">
          <DropdownMenuItem onSelect={onConnect}>
            <span className="flex min-w-0 items-center gap-1.5 text-(--ui-text-secondary)">
              <Codicon aria-hidden="true" name="plug" size="0.875rem" />
              <span className="truncate">{t.profiles.connectGateway}</span>
            </span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuRadioGroup onValueChange={choose} value={activeConnectionId ?? ''}>
            {connections.map(connection => (
              <DropdownMenuRadioItem className="min-w-0" key={connection.id} value={connection.id}>
                <ConnectionLabel connection={connection} />
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

function ConnectionGlyph({ connection }: { connection: DesktopRegistryConnection }) {
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

function ConnectionLabel({ connection }: { connection: DesktopRegistryConnection }) {
  return (
    <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
      <ConnectionGlyph connection={connection} />
      <span className="truncate">{connection.label}</span>
    </span>
  )
}
