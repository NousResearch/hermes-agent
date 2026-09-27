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
import { SidebarMenuButton } from '@/components/ui/sidebar'
import { OverflowTip, Tip } from '@/components/ui/tooltip'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionDisplayName,
  connectionEndpoint,
  connectionMatchesQuery,
  connectionTooltip,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $activeConnectionId, $connectionsRegistry, $pendingConnectionId, selectConnection } from '@/store/connections'
import { closeFindBar } from '@/store/find-in-page'
import { $machine, loadMachineProfile } from '@/store/machine'
import { notifyError } from '@/store/notifications'

import { ConnectionGlyph } from './connection-glyph'
import { useLocalDeviceSwitch } from './local-device-switch'

export function ConnectionSwitcher({ compact = false, onConnect }: { compact?: boolean; onConnect: () => void }) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const machine = useStore($machine)

  useEffect(() => {
    void loadMachineProfile()
  }, [])
  const activeConnectionId = useStore($activeConnectionId)
  const pendingConnectionId = useStore($pendingConnectionId)
  const [searchQuery, setSearchQuery] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const connectionListRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const { dialog: localDeviceDialog, request: requestLocalDevice } = useLocalDeviceSwitch()

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

    void (async () => {
      if (connection?.kind === 'local' && connectionId !== activeConnectionId) {
        const accepted = await requestLocalDevice({
          connectionId,
          label: connection.label,
          replaceCenter: true
        })

        if (!accepted) {
          return
        }
      }

      await selectConnection(connectionId).catch(error =>
        notifyError(error, t.profiles.switchConnectionFailed(connection?.label ?? connectionId))
      )
    })()
  }

  // The compact trigger may shrink, but never below its glyph, "(this device)" suffix and chevron (any locale).
  const activeHasSuffix =
    activeConnection !== undefined &&
    connectionDisplayName(activeConnection, machine?.hostname, t.settings.connections.thisDeviceSuffix).suffix !== ''

  return (
    <div
      aria-busy={pendingConnectionId !== null}
      aria-label={t.settings.connections.title}
      className={cn('min-w-20 shrink', compact ? cn('h-full max-w-52', activeHasSuffix && 'min-w-min') : 'w-full')}
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
            hostname={machine?.hostname}
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
          side={compact ? 'top' : 'bottom'}
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
                  className={cn('min-w-0', searchable && dropdownMenuRow)}
                  key={connection.id}
                  value={connection.id}
                >
                  <ConnectionLabel connection={connection} hostname={machine?.hostname} />
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
      {localDeviceDialog}
    </div>
  )
}

interface ConnectionMenuProps {
  activeConnection?: DesktopRegistryConnection
  compact: boolean
  hostname?: string
  pending: boolean
  title: string
}

function ConnectionSwitcherTrigger({
  activeConnection,
  compact,
  hostname,
  pending,
  title,
  ...triggerProps
}: ConnectionMenuProps & React.ComponentProps<'button'>) {
  const { t } = useI18n()

  const display = activeConnection
    ? connectionDisplayName(activeConnection, hostname, t.settings.connections.thisDeviceSuffix)
    : { name: title, suffix: '' }

  const fullName = [display.name, display.suffix].filter(Boolean).join(' ')

  const sharedProps = {
    ...triggerProps,
    'aria-label': activeConnection ? `${title}: ${fullName}` : title,
    className: cn(
      'min-w-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground',
      triggerProps.className
    )
  }

  const content = (
    <>
      {!compact && activeConnection && (
        <ConnectionGlyph className="size-4 [&>svg]:size-4" connection={activeConnection} />
      )}
      <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
        {pending && <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />}
        {activeConnection ? (
          <ConnectionLabel
            connection={activeConnection}
            containName={compact}
            hostname={hostname}
            showGlyph={compact}
          />
        ) : (
          <span className="min-w-0 truncate">{title}</span>
        )}
      </span>
      <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
    </>
  )

  return compact ? (
    <Button
      {...sharedProps}
      className={cn(
        'h-full min-h-0 w-full min-w-0 justify-between rounded-none px-1.5 text-[0.6875rem] font-normal text-(--ui-text-secondary)',
        sharedProps.className
      )}
      size="xs"
      type="button"
      variant="ghost"
    >
      {content}
    </Button>
  ) : (
    <SidebarMenuButton {...sharedProps} size="nav" type="button" variant="nav">
      {content}
    </SidebarMenuButton>
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

function ConnectionLabel({
  connection,
  containName = false,
  hostname,
  showGlyph = true
}: {
  connection: DesktopRegistryConnection
  /**
   * Status-bar sizing: a minmax(0, max-content) name track adds nothing to min-content (a min-w-min parent floors at
   * glyph + suffix + chevron) but its full width to max-content, so the name still fills the room up to the cap.
   */
  containName?: boolean
  hostname?: string
  showGlyph?: boolean
}) {
  const { t } = useI18n()
  const { name, suffix } = connectionDisplayName(connection, hostname, t.settings.connections.thisDeviceSuffix)
  const fullName = [name, suffix].filter(Boolean).join(' ')
  // Endpoint details teach even when the name fits; local names need a tip only on overflow.
  const NameTip = connectionEndpoint(connection) ? Tip : OverflowTip

  return (
    <span
      className={cn(
        'min-w-0 items-center gap-1 overflow-hidden',
        containName
          ? cn(
              'grid grid-flow-col',
              suffix ? 'grid-cols-[auto_minmax(0,max-content)_auto]' : 'grid-cols-[auto_minmax(0,max-content)]'
            )
          : 'flex flex-1'
      )}
    >
      {showGlyph && <ConnectionGlyph connection={connection} />}
      <NameTip label={connectionTooltip(connection, fullName)} placement="row">
        <span className="min-w-0 truncate" data-connection-name="">
          {name}
        </span>
      </NameTip>
      {suffix && (
        <>
          {' '}
          <span className="shrink-0 whitespace-nowrap text-(--ui-text-tertiary)" data-connection-suffix="">
            {suffix}
          </span>
        </>
      )}
    </span>
  )
}
