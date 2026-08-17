import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { profileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import {
  $activeConnectionId,
  $connectionsRegistry,
  $pendingConnectionId,
  refreshConnectionsRegistry,
  selectConnection
} from '@/store/connections'
import { notifyError } from '@/store/notifications'

// Direct glyphs are faster for the common local + a few remote machines. At
// larger fleet sizes, one stable-width select prevents sources from crowding
// profiles and the create/manage actions out of the sidebar foot.
const CONNECTION_DROPDOWN_THRESHOLD = 5

export function ConnectionSwitcher() {
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

  const choose = (connectionId: string) => {
    triggerHaptic('selection')
    const connection = connections.find(candidate => candidate.id === connectionId)

    void selectConnection(connectionId).catch(error =>
      notifyError(error, t.profiles.switchConnectionFailed(connection?.label ?? connectionId))
    )
  }

  if (connections.length > CONNECTION_DROPDOWN_THRESHOLD) {
    return (
      <div
        aria-busy={pendingConnectionId !== null}
        aria-label={t.settings.connections.title}
        className="min-w-0 shrink"
        role="group"
      >
        <Select onValueChange={choose} value={activeConnectionId ?? ''}>
          <SelectTrigger aria-label={t.settings.connections.title} className="w-28 max-w-full" size="xs">
            <span className="flex min-w-0 items-center gap-1.5">
              {pendingConnectionId && <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />}
              <SelectValue placeholder={t.settings.connections.title} />
            </span>
          </SelectTrigger>
          <SelectContent collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }} side="top">
            {connections.map(connection => (
              <SelectItem key={connection.id} value={connection.id}>
                <ConnectionLabel connection={connection} />
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }

  return (
    <div
      aria-label={t.settings.connections.title}
      className="flex shrink-0 items-center gap-1"
      data-slot="connection-switcher"
      role="group"
    >
      {connections.map(connection => (
        <ConnectionButton
          active={connection.id === activeConnectionId}
          connection={connection}
          key={connection.id}
          onSelect={() => choose(connection.id)}
          pending={connection.id === pendingConnectionId}
        />
      ))}
    </div>
  )
}

function ConnectionButton({
  active,
  connection,
  onSelect,
  pending
}: {
  active: boolean
  connection: DesktopRegistryConnection
  onSelect: () => void
  pending: boolean
}) {
  const { t } = useI18n()
  const label = t.profiles.switchToConnection(connection.label)

  return (
    <Tip label={label}>
      <Button
        aria-busy={pending}
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground'
        )}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        {pending ? (
          <Loader2 aria-hidden="true" className="size-3.5 animate-spin" />
        ) : (
          <ConnectionGlyph connection={connection} />
        )}
      </Button>
    </Tip>
  )
}

function ConnectionGlyph({ connection }: { connection: DesktopRegistryConnection }) {
  const local = connection.kind === 'local'

  return (
    <ProfileGlyph
      aria-hidden="true"
      color={local ? null : profileColor(`connection:${connection.id}`)}
      isDefault={local}
      name={connection.label}
    />
  )
}

function ConnectionLabel({ connection }: { connection: DesktopRegistryConnection }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <ConnectionGlyph connection={connection} />
      <span className="truncate">{connection.label}</span>
    </span>
  )
}
