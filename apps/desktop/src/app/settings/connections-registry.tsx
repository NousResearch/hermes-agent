import { useStore } from '@nanostores/react'
import { useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import type { DesktopConnectionKind, DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionMatchesQuery,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { Cloud, Globe, Loader2, Monitor, Pencil, Plus, RefreshCw, SearchIcon, Terminal, Trash2 } from '@/lib/icons'
import { $activeConnectionId, setConnectionsRegistry } from '@/store/connections'
import { refreshFleetRoster } from '@/store/fleet-roster'
import { notify, notifyError } from '@/store/notifications'

import { type ConnectionDraft, connectionDraft, ConnectionEditor, type KindMeta } from './connection-editor'
import { EmptyState, ListRow, Pill, SectionHeading, SettingsBreadcrumbContext, ToggleRow } from './primitives'

const KIND_ICONS: Record<DesktopConnectionKind, typeof Globe> = {
  cloud: Cloud,
  local: Monitor,
  remote: Globe,
  ssh: Terminal
}

/**
 * Display-only same-backend hint: when two registered connections report the
 * same /api/status install_id, they are one physical backend registered under
 * two addresses (hostname + Tailscale IP). Returns the label of the FIRST
 * earlier sibling sharing this connection's id, or null. Never blocks or
 * deletes — dual routes are legitimate failover config; the roster already
 * collapses the duplicate bot rows.
 */
export function sameBackendPeerLabel(
  conn: Pick<DesktopRegistryConnection, 'id' | 'installId'>,
  connections: Pick<DesktopRegistryConnection, 'id' | 'installId' | 'label'>[]
): null | string {
  if (!conn.installId) {
    return null
  }

  for (const other of connections) {
    if (other.id === conn.id) {
      // Only rows AFTER the first occurrence carry the hint, so exactly one
      // of a same-backend pair is annotated (the later-listed one).
      return null
    }

    if (other.installId === conn.installId) {
      return other.label
    }
  }

  return null
}

function scrollableAncestor(element: HTMLElement): HTMLElement | null {
  let parent = element.parentElement

  while (parent) {
    if (/(auto|scroll)/.test(window.getComputedStyle(parent).overflowY)) {
      return parent
    }

    parent = parent.parentElement
  }

  return null
}

/** The host lets this page add, edit and remove connections (the hosted dashboard's list is the account's). */
const registryEditable = (bridge: Window['hermesDesktop']['connections']) => !bridge?.managedByHost

/** Update all fans out over more than one connection, where the bridge can run it. */
const canUpdateAll = (bridge: Window['hermesDesktop']['connections'], registry: DesktopConnectionsRegistry | null) =>
  Boolean(bridge?.updateAll) && (registry?.connections.length ?? 0) > 1

/** A row offers edit and remove: any but this device's own, unless the host manages the list. */
const rowEditable = (conn: DesktopRegistryConnection, editable: boolean) => editable && conn.kind !== 'local'

/** Below the list: Add connection (absent where the host manages the list) and Update all. */
function RegistryFooterActions({
  canAdd,
  canUpdateAll,
  copy,
  onAdd,
  onUpdateAll,
  updatingAll
}: {
  canAdd: boolean
  canUpdateAll: boolean
  copy: { addConnection: string; updateAll: string; updateAllRunning: string }
  onAdd: () => void
  onUpdateAll: () => void
  updatingAll: boolean
}) {
  return (
    <div className="mt-4 flex items-center gap-2">
      {canAdd && (
        <Button
          onClick={() => {
            triggerHaptic('selection')
            onAdd()
          }}
          size="sm"
          variant="outline"
        >
          <Plus className="size-3.5" /> {copy.addConnection}
        </Button>
      )}
      {canUpdateAll && (
        <Button
          disabled={updatingAll}
          onClick={() => {
            triggerHaptic('selection')
            onUpdateAll()
          }}
          size="sm"
          variant="outline"
        >
          {updatingAll ? <Loader2 className="animate-spin" /> : <RefreshCw />}{' '}
          {updatingAll ? copy.updateAllRunning : copy.updateAll}
        </Button>
      )}
    </div>
  )
}

/**
 * One save at a time across drafts. The editor remounts for each draft, so a lock kept in it is
 * reset by opening another; each save answers with the whole list, so a second one started while
 * the first runs could answer first and then lose its list to the older answer.
 */
function useSaveLock(): { runSave: (work: () => Promise<void>) => Promise<void>; saving: boolean } {
  const [saving, setSaving] = useState(false)
  // Read in the click's own tick: state alone lets two clicks before a render both start.
  const held = useRef(false)

  const runSave = useCallback(async (work: () => Promise<void>) => {
    if (held.current) {
      return
    }

    held.current = true
    setSaving(true)

    try {
      await work()
    } finally {
      held.current = false
      setSaving(false)
    }
  }, [])

  return { runSave, saving }
}

/**
 * Another window, the other host (native Desktop or a Webapp tab: they share one list) or a
 * registry repair changed the list: show it without a spinner or closing an open editor.
 */
function useRegistryChanges(
  bridge: Window['hermesDesktop']['connections'],
  publish: (next: DesktopConnectionsRegistry) => void
): void {
  useEffect(
    () =>
      bridge?.onChanged?.(() => {
        void bridge
          .list()
          .then(publish)
          .catch(() => undefined)
      }),
    [bridge, publish]
  )
}

/**
 * The connections registry section of Settings → Gateways: manage the named
 * agent sources (local runtime + any number of remote gateways / Hermes Cloud
 * instances / SSH hosts). Storage-level management — the active/primary
 * switchover UX is the connection-mode controls above this section.
 */
export function ConnectionsRegistrySection() {
  const hasBreadcrumb = useContext(SettingsBreadcrumbContext)
  const { t } = useI18n()
  const s = t.settings.connections
  const activeConnectionId = useStore($activeConnectionId)
  const [registry, setRegistry] = useState<DesktopConnectionsRegistry | null>(null)
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<null | string>(null)
  const [testingId, setTestingId] = useState<null | string>(null)
  const [removeTarget, setRemoveTarget] = useState<DesktopRegistryConnection | null>(null)
  const [launchModeBusy, setLaunchModeBusy] = useState(false)
  const [updatingAll, setUpdatingAll] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const searchInputRef = useRef<HTMLInputElement>(null)
  const pendingSearchTopRef = useRef<null | number>(null)

  const bridge = window.hermesDesktop?.connections
  // The hosted dashboard keeps this list for the account: rows are tested and ordered, not edited.
  const editable = registryEditable(bridge)

  const publishRegistry = useCallback((next: DesktopConnectionsRegistry) => {
    setRegistry(next)
    setConnectionsRegistry(next)
  }, [])

  const load = useCallback(async () => {
    if (!bridge) {
      setLoading(false)

      return
    }

    setLoading(true)

    try {
      publishRegistry(await bridge.list())
    } catch (err) {
      notifyError(err, s.loadFailed)
    } finally {
      setLoading(false)
    }
  }, [bridge, publishRegistry, s.loadFailed])

  useEffect(() => {
    void load()
  }, [load])
  useRegistryChanges(bridge, publishRegistry)

  // The open add/edit form; `key` remounts it, so every open starts on its own draft.
  const [draft, setDraft] = useState<(ConnectionDraft & { key: number }) | null>(null)
  const draftKey = useRef(0)
  const { runSave, saving } = useSaveLock()

  const openEditor = (conn?: DesktopRegistryConnection): void => {
    draftKey.current += 1
    setDraft({ ...connectionDraft(conn), key: draftKey.current })
  }

  const remove = useCallback(async () => {
    if (!bridge || !removeTarget) {
      return
    }

    setBusyId(removeTarget.id)

    try {
      const result = await bridge.remove(removeTarget.id)
      publishRegistry(result.registry)
    } catch (err) {
      notifyError(err, s.removeFailed)
    } finally {
      setBusyId(null)
      setRemoveTarget(null)
    }
  }, [bridge, publishRegistry, removeTarget, s.removeFailed])

  const makePrimary = useCallback(
    async (id: string) => {
      if (!bridge) {
        return
      }

      setBusyId(id)

      try {
        const result = await bridge.setPrimary(id)
        publishRegistry(result.registry)
      } catch (err) {
        notifyError(err, s.saveFailed)
      } finally {
        setBusyId(null)
      }
    },
    [bridge, publishRegistry, s.saveFailed]
  )

  const setLaunchMode = useCallback(
    async (mode: 'last-used' | 'primary') => {
      if (!bridge?.setLaunchMode) {
        return
      }

      setLaunchModeBusy(true)

      try {
        const result = await bridge.setLaunchMode(mode)
        publishRegistry(result.registry)
      } catch (err) {
        notifyError(err, s.saveFailed)
      } finally {
        setLaunchModeBusy(false)
      }
    },
    [bridge, publishRegistry, s.saveFailed]
  )

  const test = useCallback(
    async (conn: DesktopRegistryConnection) => {
      if (!bridge) {
        return
      }

      setTestingId(conn.id)

      try {
        const result = await bridge.test(conn.id)
        const reachable = result.ok === true || result.reachable === true

        if (reachable) {
          notify({ title: conn.label, message: s.testOk })
          // A successful Test may have warmed a cold OAuth session that the
          // roster missed; explicit recovery should bypass its cache window.
          void refreshFleetRoster({ force: true })
        } else {
          notifyError(new Error(result.error || conn.label), s.testFailed)
        }
      } catch (err) {
        notifyError(err, s.testFailed)
      } finally {
        setTestingId(null)
      }
    },
    [bridge, s.testFailed, s.testOk]
  )

  // Fan out `hermes update` to every eligible source; per-connection results
  // land as individual toasts so one dead box doesn't hide the others.
  const updateAll = useCallback(async () => {
    if (!bridge?.updateAll) {
      return
    }

    setUpdatingAll(true)

    try {
      const { results } = await bridge.updateAll()

      for (const row of results) {
        if (row.ok) {
          notify({ title: row.label, message: row.detail || s.updateAllDone })
        } else if (row.skipped && row.reason === 'cloud-managed') {
          notify({ title: row.label, message: s.updateSkippedCloud })
        } else if (row.skipped && row.reason === 'darwin-drain-unsupported' && row.detail) {
          // A deliberate per-row skip (e.g. a macOS SSH remote whose running
          // serve Desktop cannot safely stop) — informational, not a failure.
          notify({ title: row.label, message: row.detail })
        } else {
          notifyError(new Error(row.error || row.detail || row.reason || row.label), s.updateAllFailed)
        }
      }
    } catch (err) {
      notifyError(err, s.updateAllFailed)
    } finally {
      setUpdatingAll(false)
    }
  }, [bridge, s.updateAllDone, s.updateAllFailed, s.updateSkippedCloud])

  const kindMeta: KindMeta = {
    cloud: { desc: s.kindCloudDesc, label: s.kindCloud },
    local: { desc: s.kindLocalDesc, label: s.kindLocal },
    remote: { desc: s.kindRemoteDesc, label: s.kindRemote },
    ssh: { desc: s.kindSshDesc, label: s.kindSsh }
  }

  const sortedConnections = useMemo(
    () => sortConnectionsForDisplay(registry?.connections ?? []),
    [registry?.connections]
  )

  const showSearch = sortedConnections.length >= CONNECTION_SEARCH_THRESHOLD
  const effectiveSearchQuery = showSearch ? searchQuery : ''

  const displayedConnections = sortedConnections.filter(connection =>
    connectionMatchesQuery(connection, effectiveSearchQuery, [kindMeta[connection.kind].label])
  )

  useLayoutEffect(() => {
    const previousTop = pendingSearchTopRef.current
    const input = searchInputRef.current

    pendingSearchTopRef.current = null

    if (previousTop == null || !input) {
      return
    }

    const scroller = scrollableAncestor(input)

    if (!scroller) {
      return
    }

    const delta = input.getBoundingClientRect().top - previousTop

    if (Math.abs(delta) > 0.5) {
      scroller.scrollTop += delta
    }
  }, [displayedConnections.length, effectiveSearchQuery])

  const updateSearchQuery = (nextQuery: string) => {
    pendingSearchTopRef.current = searchInputRef.current?.getBoundingClientRect().top ?? null
    setSearchQuery(nextQuery)
  }

  if (!bridge) {
    return null
  }

  return (
    <div className={hasBreadcrumb ? undefined : 'mt-8 border-t border-border/60 pt-6'}>
      <SectionHeading icon={Globe} page title={s.title} />
      <p
        className="mb-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)"
        hidden={!editable}
      >
        {s.intro}
      </p>
      {/* Source selection lives in Sessions. Primary is the registry fallback,
          not an immediate workspace switch. */}
      <p className="mb-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.stagedNote}
      </p>

      {!loading && showSearch && (
        <Input
          aria-label={s.searchPlaceholder}
          containerClassName="mt-3 mb-0 w-full max-w-sm"
          onChange={event => updateSearchQuery(event.target.value)}
          placeholder={s.searchPlaceholder}
          prefix={<SearchIcon className="size-3.5" />}
          ref={searchInputRef}
          size="sm"
          type="search"
          value={searchQuery}
        />
      )}

      {loading ? (
        <Loader className="mx-auto my-2 size-6 text-(--ui-text-tertiary)" label={t.common.loading} type="rose-curve" />
      ) : !registry || registry.connections.length === 0 ? (
        <EmptyState title={s.empty} />
      ) : displayedConnections.length === 0 ? (
        <EmptyState title={s.noSearchResults} />
      ) : (
        displayedConnections.map(conn => {
          const Icon = KIND_ICONS[conn.kind]
          const isCurrent = activeConnectionId === conn.id
          const isPrimary = registry.primary === conn.id
          const busy = busyId === conn.id
          // Display-only: this connection is a second address for a backend
          // already registered under another entry (same install_id).
          const sameBackendPeer = sameBackendPeerLabel(conn, sortedConnections)

          const baseDescription =
            conn.kind === 'ssh'
              ? `${kindMeta[conn.kind].label} · ${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}`
              : conn.url
                ? `${kindMeta[conn.kind].label} · ${conn.url}`
                : kindMeta[conn.kind].desc

          return (
            <ListRow
              action={
                <div className="flex items-center gap-2">
                  <Button
                    disabled={testingId === conn.id}
                    onClick={() => {
                      triggerHaptic('selection')
                      void test(conn)
                    }}
                    size="sm"
                    variant="outline"
                  >
                    {testingId === conn.id ? <Loader2 className="animate-spin" /> : s.testConnection}
                  </Button>
                  {!isPrimary && (
                    <Button disabled={busy} onClick={() => void makePrimary(conn.id)} size="sm" variant="outline">
                      {s.makePrimary}
                    </Button>
                  )}
                  {rowEditable(conn, editable) && (
                    <>
                      <Button
                        aria-label={s.editConnection}
                        onClick={() => openEditor(conn)}
                        size="icon-sm"
                        variant="ghost"
                      >
                        <Pencil className="size-3.5" />
                      </Button>
                      <Button
                        aria-label={s.removeConnection}
                        disabled={busy}
                        onClick={() => setRemoveTarget(conn)}
                        size="icon-sm"
                        variant="ghost"
                      >
                        {busy ? <Loader2 className="animate-spin" /> : <Trash2 />}
                      </Button>
                    </>
                  )}
                </div>
              }
              description={
                sameBackendPeer ? `${baseDescription} · ${s.sameBackendHint(sameBackendPeer)}` : baseDescription
              }
              key={conn.id}
              title={
                <span className="flex items-center gap-2">
                  <Icon className="size-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{conn.label}</span>
                  {isCurrent && <Pill tone="primary">{s.currentPill}</Pill>}
                  {isPrimary && <Pill>{s.primaryPill}</Pill>}
                  {conn.kind === 'local' && <Pill>{s.managedPill}</Pill>}
                </span>
              }
            />
          )
        })
      )}

      {draft && editable ? (
        <ConnectionEditor
          bridge={bridge}
          draft={draft}
          key={draft.key}
          kindMeta={kindMeta}
          onClose={() => setDraft(null)}
          onSaved={next => {
            publishRegistry(next)
            // Only the draft that started the save: another row's Edit may have opened one since.
            const saved = draft.key
            setDraft(current => (current?.key === saved ? null : current))
          }}
          registry={registry}
          runSave={runSave}
          saving={saving}
        />
      ) : (
        <RegistryFooterActions
          canAdd={editable}
          canUpdateAll={canUpdateAll(bridge, registry)}
          copy={s}
          onAdd={() => openEditor()}
          onUpdateAll={() => void updateAll()}
          updatingAll={updatingAll}
        />
      )}

      {/* Not gated on connections.length > 1: a registry that drifted down to
          local-only is exactly the state where a user needs to see and change
          the launch behavior, and hiding the control there left hand-editing
          connections.json as the only recourse (#90174). */}
      {!loading && registry && (
        <div className="mt-6 border-t border-border/60 pt-4">
          <ToggleRow
            checked={registry.launchMode === 'last-used'}
            description={s.launchModeDesc}
            disabled={launchModeBusy || !bridge?.setLaunchMode}
            label={s.launchModeTitle}
            onChange={enabled => void setLaunchMode(enabled ? 'last-used' : 'primary')}
          />
        </div>
      )}

      <ConfirmDialog
        confirmLabel={s.removeConnection}
        description={removeTarget ? s.removeConfirmDesc(removeTarget.label) : ''}
        destructive
        onClose={() => setRemoveTarget(null)}
        onConfirm={() => remove()}
        open={Boolean(removeTarget)}
        title={s.removeConfirmTitle}
      />
    </div>
  )
}
