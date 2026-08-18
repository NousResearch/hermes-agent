import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow,
  DropdownMenuSearch,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionMatchesQuery,
  connectionTooltip,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { Cloud, Loader2, Monitor, Network, Terminal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopBoot } from '@/store/boot'
import {
  $activeConnectionId,
  $connectionsRegistry,
  $pendingConnectionId,
  initializeConnectionsRegistry,
  refreshConnectionsRegistry,
  selectConnection
} from '@/store/connections'
import { notifyError } from '@/store/notifications'

export function ConnectionSwitcher({ onConnect }: { onConnect: () => void }) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const boot = useStore($desktopBoot)
  const pendingConnectionId = useStore($pendingConnectionId)
  const [searchQuery, setSearchQuery] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const connectionListRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void refreshConnectionsRegistry().catch(() => undefined)

    // Registry events are local IPC notifications, not remote polling. They
    // keep a second Settings window or a removal/edit reflected here.
    const off = window.hermesDesktop?.connections?.onChanged?.(() => {
      void refreshConnectionsRegistry().catch(() => undefined)
    })

    return off
  }, [])

  useEffect(() => {
    // The primary boot owns its initial config/session fetches. Restoring a
    // different source before those settle lets a late primary response repaint
    // the sidebar under the new source label. Switch only after boot completes,
    // then the normal source reset/refetch remains the final writer.
    if (!boot.running) {
      void initializeConnectionsRegistry().catch(() => undefined)
    }
  }, [boot.running])

  const connections = useMemo(() => sortConnectionsForDisplay(registry?.connections ?? []), [registry?.connections])

  const activeConnection = connections.find(connection => connection.id === activeConnectionId)
  const searchable = connections.length >= CONNECTION_SEARCH_THRESHOLD

  const kindLabels: Record<DesktopRegistryConnection['kind'], string> = {
    cloud: t.settings.connections.kindCloud,
    local: t.settings.connections.kindLocal,
    remote: t.settings.connections.kindRemote,
    ssh: t.settings.connections.kindSsh
  }

  const displayedConnections = searchable
    ? connections.filter(connection => connectionMatchesQuery(connection, searchQuery, [kindLabels[connection.kind]]))
    : connections

  useEffect(() => {
    if (!menuOpen || !searchable || searchQuery) {
      return
    }

    connectionListRef.current?.querySelector('[aria-checked="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [activeConnectionId, menuOpen, searchQuery, searchable])

  useEffect(() => {
    if (!menuOpen) {
      return
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      setMenuOpen(false)
      setSearchQuery('')
    }

    window.addEventListener('keydown', closeOnEscape, { capture: true })

    return () => window.removeEventListener('keydown', closeOnEscape, { capture: true })
  }, [menuOpen])

  if (connections.length <= 1) {
    return null
  }

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
      <DropdownMenu
        onOpenChange={open => {
          setMenuOpen(open)

          if (!open) {
            setSearchQuery('')
          }
        }}
        open={menuOpen}
      >
        <DropdownMenuTrigger asChild>
          <ConnectionSwitcherTrigger
            activeConnection={activeConnection}
            pending={pendingConnectionId !== null}
            title={t.settings.connections.title}
          />
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className={cn('min-w-52 max-w-72', searchable && 'w-72 overflow-hidden p-0')}
          collisionPadding={8}
          onKeyDownCapture={event => {
            if (searchable && (event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === 'f') {
              event.preventDefault()
              event.stopPropagation()
              searchInputRef.current?.focus()
              searchInputRef.current?.select()
            }
          }}
          side="top"
        >
          {searchable && (
            <>
              <DropdownMenuSearch
                className="[font-family:inherit] text-xs font-normal leading-4"
                onKeyDown={event => {
                  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
                    return
                  }

                  const results = connectionListRef.current?.querySelectorAll<HTMLElement>(
                    '[role="menuitemradio"]:not([data-disabled])'
                  )

                  const target = event.key === 'ArrowDown' ? results?.item(0) : results?.item((results?.length ?? 1) - 1)

                  if (target) {
                    event.preventDefault()
                    event.stopPropagation()
                    target.focus()
                  }
                }}
                onValueChange={setSearchQuery}
                placeholder={t.settings.connections.searchPlaceholder}
                ref={searchInputRef}
                value={searchQuery}
              />
              <DropdownMenuSeparator className="m-0" />
            </>
          )}
          <DropdownMenuRadioGroup
            className={
              searchable
                ? 'dt-portal-scrollbar h-48 max-h-[calc(var(--radix-dropdown-menu-content-available-height)-4.5rem)] overflow-y-auto p-1'
                : undefined
            }
            onValueChange={choose}
            ref={connectionListRef}
            value={activeConnectionId ?? ''}
          >
            {displayedConnections.length === 0 ? (
              <div
                className="flex h-full items-center justify-center px-4 text-center text-xs text-(--ui-text-tertiary)"
                role="status"
              >
                {t.settings.connections.noSearchResults}
              </div>
            ) : (
              displayedConnections.map(connection => (
                <DropdownMenuRadioItem
                  className={cn('min-w-0', searchable && dropdownMenuRow)}
                  key={connection.id}
                  value={connection.id}
                >
                  <ConnectionLabel connection={connection} />
                </DropdownMenuRadioItem>
              ))
            )}
          </DropdownMenuRadioGroup>
          <DropdownMenuSeparator className={searchable ? 'm-0' : undefined} />
          <DropdownMenuItem className={searchable ? dropdownMenuRow : undefined} onSelect={onConnect}>
            <ManageGatewaysLabel label={t.profiles.connectGateway} />
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

interface ConnectionMenuProps {
  activeConnection?: DesktopRegistryConnection
  pending: boolean
  title: string
}

function ConnectionSwitcherTrigger({
  activeConnection,
  pending,
  title,
  ...triggerProps
}: ConnectionMenuProps & React.ComponentProps<'button'>) {
  return (
    <Button
      {...triggerProps}
      aria-label={activeConnection ? `${title}: ${activeConnection.label}` : title}
      className={cn(
        'w-full min-w-0 justify-between overflow-hidden px-1 text-(--ui-text-secondary) data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground',
        triggerProps.className
      )}
      size="xs"
      type="button"
      variant="ghost"
    >
      <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
        {pending && <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />}
        {activeConnection ? (
          <ConnectionLabel connection={activeConnection} />
        ) : (
          <span className="truncate">{title}</span>
        )}
      </span>
      <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
    </Button>
  )
}

function ManageGatewaysLabel({ label }: { label: string }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5 text-(--ui-text-secondary)">
      <Codicon aria-hidden="true" name="settings-gear" size="0.875rem" />
      <span className="truncate">{label}</span>
    </span>
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
    <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden" title={connectionTooltip(connection)}>
      <ConnectionGlyph connection={connection} />
      <span className="truncate">{connection.label}</span>
    </span>
  )
}
