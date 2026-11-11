import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { EmptyState } from '@/components/ui/empty-state'
import type { DesktopConnectionKind, DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Cloud, Globe, Loader2, Monitor, Pencil, RefreshCw, Terminal, Trash2 } from '@/lib/icons'
import { usePortalShell } from '@/store/shell-variant'

import { sameBackendPeerLabel } from './connections-registry'
import { ListRow, Pill } from './primitives'
import { useSettingsPagesCopy } from './settings-pages-copy'

const KIND_ICONS: Record<DesktopConnectionKind, typeof Globe> = {
  cloud: Cloud,
  local: Monitor,
  remote: Globe,
  ssh: Terminal
}

export type ConnectionKindMeta = Record<DesktopConnectionKind, { label: string; desc: string }>

/** An empty saved-connections list: a quiet line in the redesign's card, classic's empty state. */
export function ConnectionsEmpty({ title }: { title: string }) {
  return usePortalShell() ? <p className="global-settings-empty-row">{title}</p> : <EmptyState title={title} />
}

interface ConnectionActions {
  busy: boolean
  conn: DesktopRegistryConnection
  isPrimary: boolean
  testing: boolean
  onEdit: () => void
  onMakePrimary: () => void
  onRemove: () => void
  onTest: () => void
}

interface ConnectionRowProps extends ConnectionActions {
  isCurrent: boolean
  kindMeta: ConnectionKindMeta
  /** Every saved connection, to name another entry that reaches the same backend. */
  peers: DesktopRegistryConnection[]
}

function connectionDescription(conn: DesktopRegistryConnection, kindMeta: ConnectionKindMeta): string {
  const kind = kindMeta[conn.kind].label

  if (conn.kind === 'ssh') {
    return `${kind} · ${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}`
  }

  return conn.url ? `${kind} · ${conn.url}` : kindMeta[conn.kind].desc
}

/** One saved connection: kind icon, name and state pills, endpoint, and its actions. */
export function ConnectionRow({ isCurrent, kindMeta, peers, ...actions }: ConnectionRowProps) {
  const { t } = useI18n()
  const s = t.settings.connections
  const redesign = usePortalShell()
  const { conn, isPrimary } = actions
  const Icon = KIND_ICONS[conn.kind]
  // Display-only: this connection is a second address for a backend already registered under
  // another entry (same install_id).
  const sameBackendPeer = sameBackendPeerLabel(conn, peers)
  const description = connectionDescription(conn, kindMeta)

  return (
    <ListRow
      action={redesign ? <ConnectionManageMenu {...actions} /> : <ClassicConnectionActions {...actions} />}
      description={sameBackendPeer ? `${description} · ${s.sameBackendHint(sameBackendPeer)}` : description}
      title={
        <span className="flex min-w-0 flex-wrap items-center gap-2">
          <Icon className="size-4 shrink-0 text-muted-foreground" />
          <span className="max-w-full truncate">{conn.label}</span>
          {isCurrent && <Pill tone="primary">{s.currentPill}</Pill>}
          {isPrimary && <Pill>{s.primaryPill}</Pill>}
          {conn.kind === 'local' && <Pill>{s.managedPill}</Pill>}
        </span>
      }
    />
  )
}

function ClassicConnectionActions({
  busy,
  conn,
  isPrimary,
  testing,
  onEdit,
  onMakePrimary,
  onRemove,
  onTest
}: ConnectionActions) {
  const { t } = useI18n()
  const s = t.settings.connections

  return (
    <div className="flex items-center gap-2">
      <Button
        disabled={testing}
        onClick={() => {
          triggerHaptic('selection')
          onTest()
        }}
        size="sm"
        variant="outline"
      >
        {testing ? <Loader2 className="animate-spin" /> : s.testConnection}
      </Button>
      {!isPrimary && (
        <Button disabled={busy} onClick={onMakePrimary} size="sm" variant="outline">
          {s.makePrimary}
        </Button>
      )}
      {conn.kind !== 'local' && (
        <>
          <Button aria-label={s.editConnection} onClick={onEdit} size="icon-sm" variant="ghost">
            <Pencil className="size-3.5" />
          </Button>
          <Button aria-label={s.removeConnection} disabled={busy} onClick={onRemove} size="icon-sm" variant="ghost">
            {busy ? <Loader2 className="animate-spin" /> : <Trash2 />}
          </Button>
        </>
      )}
    </div>
  )
}

/** The redesign's single "Manage" control per saved connection (856:5142); it carries every action
 *  classic lays out as separate buttons. The app-managed local entry cannot be edited or removed. */
function ConnectionManageMenu({
  busy,
  conn,
  isPrimary,
  testing,
  onEdit,
  onMakePrimary,
  onRemove,
  onTest
}: ConnectionActions) {
  const { t } = useI18n()
  const s = t.settings.connections
  const pages = useSettingsPagesCopy()
  const editable = conn.kind !== 'local'

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={`${pages.gateway.manage}: ${conn.label}`} disabled={busy} size="sm" variant="outline">
          {busy || testing ? <Loader2 className="animate-spin" /> : null}
          {pages.gateway.manage}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          disabled={testing}
          onSelect={() => {
            triggerHaptic('selection')
            onTest()
          }}
        >
          <RefreshCw />
          {s.testConnection}
        </DropdownMenuItem>
        {!isPrimary && (
          <DropdownMenuItem onSelect={onMakePrimary}>
            <Check />
            {s.makePrimary}
          </DropdownMenuItem>
        )}
        {editable && (
          <>
            <DropdownMenuItem onSelect={onEdit}>
              <Pencil />
              {s.editConnection}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={onRemove} variant="destructive">
              <Trash2 />
              {s.removeConnection}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
