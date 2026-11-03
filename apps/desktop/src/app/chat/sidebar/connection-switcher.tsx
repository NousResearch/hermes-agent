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
import type { DesktopProfileRoute, DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionMatchesQuery,
  connectionTooltip,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { AppWindow, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $activeConnectionId, $connectionsRegistry, $pendingConnectionId, selectConnection } from '@/store/connections'
import { $defaultProfileRoute } from '@/store/default-profile'
import { closeFindBar } from '@/store/find-in-page'
import { notifyError } from '@/store/notifications'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'

import { ConnectionGlyph } from './connection-glyph'
import { DEFAULT_PROFILE } from './fleet-rail'
import { useLocalDeviceSwitch } from './local-device-switch'

/**
 * A switcher row names a CONNECTION; a new window needs a profile as well.
 * Prefer the route this app already defaults to when it lives on the target
 * connection, then the SSH entry's pinned profile, then the gateway's home
 * profile. It deliberately never falls back to this window's own route —
 * staying off the source window is the whole point of opening a peer.
 */
function windowProfileFor(connection: DesktopRegistryConnection, defaultRoute: DesktopProfileRoute | null): string {
  const pinned = defaultRoute?.connectionId === connection.id ? defaultRoute.profile.trim() : ''

  if (pinned) {
    return pinned
  }

  return connection.remoteProfile?.trim() || DEFAULT_PROFILE
}

export function ConnectionSwitcher({ compact = false, onConnect }: { compact?: boolean; onConnect: () => void }) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const pendingConnectionId = useStore($pendingConnectionId)
  const [searchQuery, setSearchQuery] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const connectionListRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const { dialog: localDeviceDialog, request: requestLocalDevice } = useLocalDeviceSwitch()
  const defaultProfileRoute = useStore($defaultProfileRoute)
  // Only Electron can open a peer window; the web dashboard renders the same
  // switcher and must not advertise a gesture it cannot serve.
  const windowHint = canOpenNewWindow()

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

  const openInNewWindowFor = (connection: DesktopRegistryConnection) => {
    triggerHaptic('selection')
    void openNewWindow({ connectionId: connection.id, profile: windowProfileFor(connection, defaultProfileRoute) })
  }

  const choose = (connectionId: string) => {
    triggerHaptic('selection')
    const connection = connections.find(candidate => candidate.id === connectionId)

    void (async () => {
      if (connection?.kind === 'local' && connectionId !== activeConnectionId) {
        const accepted = await requestLocalDevice({ connectionId, label: connection.label })

        if (!accepted) {
          return
        }
      }

      await selectConnection(connectionId).catch(error =>
        notifyError(error, t.profiles.switchConnectionFailed(connection?.label ?? connectionId))
      )
    })()
  }

  return (
    <div
      aria-busy={pendingConnectionId !== null}
      aria-label={t.settings.connections.title}
      className={cn('min-w-20 shrink overflow-hidden', compact ? 'h-full max-w-40' : 'max-w-72')}
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
            compact={compact}
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
              // The app-level keybind sees the chord at window capture before
              // this portal and may open Find in page. This menu owns the chord
              // while it is open, so close that surface before focusing here.
              closeFindBar()
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

                  const target =
                    event.key === 'ArrowDown' ? results?.item(0) : results?.item((results?.length ?? 1) - 1)

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
                  className={cn('group/item min-w-0', searchable && dropdownMenuRow)}
                  key={connection.id}
                  // Shift opens a peer window instead of re-homing this one. It
                  // cannot ride the radio group's onValueChange: re-picking the
                  // already-active row fires no change, and that is exactly the
                  // row a second window is most often wanted on.
                  onKeyDown={event => {
                    if (!windowHint || !event.shiftKey || (event.key !== 'Enter' && event.key !== ' ')) {
                      return
                    }

                    event.preventDefault()
                    event.stopPropagation()
                    openInNewWindowFor(connection)
                  }}
                  onPointerDown={event => {
                    if (!windowHint || !event.shiftKey) {
                      return
                    }

                    event.preventDefault()
                    event.stopPropagation()
                    openInNewWindowFor(connection)
                  }}
                  value={connection.id}
                >
                  <ConnectionLabel connection={connection} />
                  {windowHint && (
                    <span
                      aria-hidden="true"
                      className="ml-auto shrink-0 pl-1 opacity-0 transition-opacity group-focus-within/item:opacity-60 group-hover/item:opacity-60"
                    >
                      <AppWindow className="size-3.5" />
                    </span>
                  )}
                </DropdownMenuRadioItem>
              ))
            )}
          </DropdownMenuRadioGroup>
          <DropdownMenuSeparator className={searchable ? 'm-0' : undefined} />
          {windowHint && (
            <div className="px-2 pb-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
              {t.profiles.openInNewWindowHint}
            </div>
          )}
          <DropdownMenuItem className={searchable ? dropdownMenuRow : undefined} onSelect={onConnect}>
            <ManageGatewaysLabel label={t.profiles.connectGateway} />
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {localDeviceDialog}
    </div>
  )
}

interface ConnectionMenuProps {
  activeConnection?: DesktopRegistryConnection
  compact: boolean
  pending: boolean
  title: string
}

function ConnectionSwitcherTrigger({
  activeConnection,
  compact,
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
        compact && 'h-full min-h-0 rounded-none px-1.5 text-[0.6875rem] font-normal',
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

function ConnectionLabel({ connection }: { connection: DesktopRegistryConnection }) {
  return (
    <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden" title={connectionTooltip(connection)}>
      <ConnectionGlyph connection={connection} />
      <span className="truncate">{connection.label}</span>
    </span>
  )
}
