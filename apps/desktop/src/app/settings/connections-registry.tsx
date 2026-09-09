import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Input } from '@/components/ui/input'
import type {
  DesktopConnectionKind,
  DesktopConnectionProbeResult,
  DesktopConnectionsRegistry,
  DesktopRegistryConnection,
  DesktopRegistryConnectionInput
} from '@/global'
import { useI18n } from '@/i18n'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionMatchesQuery,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { deriveRemoteAuthProviderShape } from '@/lib/desktop-remote-auth'
import { triggerHaptic } from '@/lib/haptics'
import {
  Check,
  Cloud,
  Globe,
  Loader2,
  LogIn,
  Monitor,
  Pencil,
  Plus,
  RefreshCw,
  SearchIcon,
  Terminal,
  Trash2
} from '@/lib/icons'
import { coerceRemoteUrlScheme } from '@/lib/remote-url'
import { $activeConnectionId, setConnectionsRegistry } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'

import { EmptyState, ListRow, Pill, SectionHeading, ToggleRow } from './primitives'

const KIND_ICONS: Record<DesktopConnectionKind, typeof Globe> = {
  cloud: Cloud,
  local: Monitor,
  remote: Globe,
  ssh: Terminal
}

interface EditorState {
  // null id → creating a new connection.
  id: null | string
  kind: DesktopConnectionKind
  label: string
  url: string
  authMode: 'oauth' | 'token'
  token: string
  host: string
  keyPath: string
  // ssh remote profile, hydrated on edit so the duplicate key matches the
  // main-process one (user@host:port + profile); the editor doesn't expose it.
  remoteProfile: string
  // Extra gateway headers (access proxies such as Cloudflare Access).
  // `value: ''` on a row that came from storage means "keep the saved
  // secret" (sent as null); a typed value replaces it; a removed row clears
  // it. `stored` marks rows hydrated from headerNames so the placeholder can
  // say "saved" instead of demanding a value.
  headers: { name: string; stored: boolean; value: string }[]
}

function editorFromConnection(conn: DesktopRegistryConnection): EditorState {
  return {
    id: conn.id,
    kind: conn.kind,
    label: conn.label,
    url: conn.url || '',
    authMode: conn.authMode || 'token',
    token: '',
    // Reconstruct the composite the single ssh host field displays. The save
    // payload sends ONLY this string (never separate user/port), because
    // normalizeSshConfig gives explicit user/port fields precedence over the
    // parsed host string — sending stored user/port alongside a retyped host
    // would silently resurrect the old values.
    host: conn.host ? `${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}` : '',
    keyPath: conn.keyPath || '',
    remoteProfile: conn.remoteProfile || '',
    headers: (conn.headerNames || []).map(name => ({ name, stored: true, value: '' }))
  }
}

function emptyEditor(kind: DesktopConnectionKind): EditorState {
  return {
    id: null,
    kind,
    label: '',
    url: '',
    authMode: 'token',
    token: '',
    host: '',
    keyPath: '',
    remoteProfile: '',
    headers: []
  }
}

/** Dedupe key for a remote/cloud gateway URL: trim, drop trailing slashes, lowercase. */
export function normalizeGatewayUrl(url: string): string {
  return url.trim().replace(/\/+$/, '').toLowerCase()
}

/**
 * Dedupe key for the composite ssh host string the editor shows
 * (`user@host:port`, user/port optional): normalized to `user@host:port` with
 * the default port made explicit so `box` and `box:22` collide.
 */
export function sshCompositeKey(composite: string): string {
  const raw = composite.trim().toLowerCase()

  if (!raw) {
    return ''
  }

  const at = raw.lastIndexOf('@')
  const user = at > 0 ? raw.slice(0, at) : ''
  const rest = at >= 0 ? raw.slice(at + 1) : raw
  const portMatch = rest.match(/^(.*?):(\d+)$/)
  const host = portMatch ? portMatch[1] : rest
  const port = portMatch ? portMatch[2] : '22'

  if (!host) {
    return ''
  }

  return `${user}@${host}:${port}`
}

/**
 * The renderer-side duplicate rule, mirrored by the main process
 * (normalizeConnectionInput enforces the same keys in the save path):
 *  - at most ONE local entry, ever;
 *  - remote/cloud entries are duplicates when their normalized URLs match;
 *  - ssh entries are duplicates on user@host:port + remote profile.
 * Returns the existing entry the candidate collides with, or null.
 */
export function findDuplicateConnection(
  editor: Pick<EditorState, 'host' | 'id' | 'kind' | 'remoteProfile' | 'url'>,
  connections: DesktopRegistryConnection[]
): DesktopRegistryConnection | null {
  if (editor.kind === 'local') {
    return connections.find(c => c.kind === 'local' && c.id !== editor.id) ?? null
  }

  if (editor.kind === 'remote' || editor.kind === 'cloud') {
    const key = normalizeGatewayUrl(editor.url)

    if (!key) {
      return null
    }

    return (
      connections.find(
        c =>
          (c.kind === 'remote' || c.kind === 'cloud') && c.id !== editor.id && normalizeGatewayUrl(c.url || '') === key
      ) ?? null
    )
  }

  const key = sshCompositeKey(editor.host)

  if (!key) {
    return null
  }

  const profile = editor.remoteProfile.trim()

  return (
    connections.find(
      c =>
        c.kind === 'ssh' &&
        c.id !== editor.id &&
        sshCompositeKey(`${c.user ? `${c.user}@` : ''}${c.host ?? ''}${c.port ? `:${c.port}` : ''}`) === key &&
        (c.remoteProfile || '').trim() === profile
    ) ?? null
  )
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

/**
 * The connections registry section of Settings → Gateways: manage the named
 * agent sources (local runtime + any number of remote gateways / Hermes Cloud
 * instances / SSH hosts). Storage-level management — the active/primary
 * switchover UX is the connection-mode controls above this section.
 */
export function ConnectionsRegistrySection() {
  const { t } = useI18n()
  const s = t.settings.connections
  const activeConnectionId = useStore($activeConnectionId)
  const [registry, setRegistry] = useState<DesktopConnectionsRegistry | null>(null)
  const [loading, setLoading] = useState(true)
  const [editor, setEditor] = useState<EditorState | null>(null)
  const [saving, setSaving] = useState(false)
  const [busyId, setBusyId] = useState<null | string>(null)
  const [testingId, setTestingId] = useState<null | string>(null)
  const [removeTarget, setRemoveTarget] = useState<DesktopRegistryConnection | null>(null)
  const [plainTextConfirm, setPlainTextConfirm] = useState(false)
  const [launchModeBusy, setLaunchModeBusy] = useState(false)
  const [updatingAll, setUpdatingAll] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const searchInputRef = useRef<HTMLInputElement>(null)
  const pendingSearchTopRef = useRef<null | number>(null)
  // Inline duplicate rejection from the save path (dedupe is also enforced in
  // the main process, so a crafted payload can't slip past the UI check).
  const [dupeError, setDupeError] = useState<null | string>(null)
  // A gated remote gateway (OAuth, or username/password) never accepts a
  // session token: it authenticates with a browser sign-in and keeps the
  // session itself. Probe the edited URL so this row can name the provider,
  // and remember whether the login round-trip actually completed.
  const [authProbe, setAuthProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const [signingIn, setSigningIn] = useState(false)
  const [oauthConnected, setOauthConnected] = useState(false)
  const probeSeq = useRef(0)

  const bridge = window.hermesDesktop?.connections

  const hasLocal = Boolean(registry?.connections.some(c => c.kind === 'local'))

  const publishRegistry = useCallback((next: DesktopConnectionsRegistry) => {
    setRegistry(next)
    setConnectionsRegistry(next)
  }, [])

  const editorUrl = editor?.kind === 'remote' ? coerceRemoteUrlScheme(editor.url) : ''
  const editorWantsOauth = editor?.kind === 'remote' && editor.authMode === 'oauth'
  const authProviderShape = deriveRemoteAuthProviderShape(authProbe?.providers, t.boot.failure.identityProvider)

  // Probe only while the sign-in row is on screen, and debounce it so typing a
  // URL doesn't fire a request per keystroke. Best-effort: a failed probe just
  // leaves the generic provider label, it never blocks signing in.
  useEffect(() => {
    if (!editorWantsOauth || !editorUrl || !window.hermesDesktop?.probeConnectionConfig) {
      setAuthProbe(null)

      return
    }

    const seq = ++probeSeq.current
    // Staleness is covered by probeSeq, but not unmount: a probe resolving
    // after the editor closes would still call setAuthProbe on an unmounted
    // component. Harmless in React 18, still worth not doing.
    let cancelled = false

    const timer = setTimeout(() => {
      window.hermesDesktop
        .probeConnectionConfig(editorUrl)
        .then(result => {
          if (!cancelled && seq === probeSeq.current) {
            setAuthProbe(result)
          }
        })
        .catch(() => {
          if (!cancelled && seq === probeSeq.current) {
            setAuthProbe(null)
          }
        })
    }, 400)

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [editorUrl, editorWantsOauth])

  // The session is scoped to an origin, so pointing the editor at a different
  // URL invalidates the "signed in" state this row is reporting. Flipping the
  // auth mode invalidates it too: a saved row edited token -> oauth must not
  // present a stale "Signed in" pill from an earlier oauth stint.
  useEffect(() => {
    setOauthConnected(false)
  }, [editorUrl, editorWantsOauth])

  // Open the gateway's own login window and let the main process keep whatever
  // it mints (native PKCE bearer tokens, or the legacy session cookies). This
  // is the same IPC the first-run form and the gateway panel use — the
  // registry editor simply had no affordance to reach it.
  //
  // The draft's identity rides along because this sign-in can run BEFORE the
  // connection is saved: the main process settles the id the save will reuse
  // and writes the session into that connection's own cookie jar. An unsaved
  // draft has no on-disk entry to URL-match, so without the identity the
  // session would land in the legacy shared jar the saved connection never
  // reads. Pin the settled id into the draft so Save reuses it.
  const signInOauth = useCallback(async () => {
    if (!editorUrl) {
      notify({ kind: 'warning', title: t.settings.gateway.authTitle, message: t.settings.gateway.enterUrlFirst })

      return
    }

    setSigningIn(true)

    try {
      const result = await window.hermesDesktop.oauthLoginConnectionConfig(editorUrl, {
        connectionId: editor?.id ?? null,
        label: editor?.label ?? ''
      })

      if (result.connectionId) {
        const settledId = result.connectionId

        setEditor(prev => (prev && !prev.id ? { ...prev, id: settledId } : prev))
      }

      setOauthConnected(Boolean(result.connected))

      if (result.connected) {
        notify({
          title: t.settings.gateway.signedIn,
          message: t.settings.gateway.connectedTo(authProviderShape.providerLabel)
        })
      } else {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })
      }
    } catch (err) {
      notifyError(err, t.settings.gateway.signInFailed)
    } finally {
      setSigningIn(false)
    }
  }, [authProviderShape.providerLabel, editor?.id, editor?.label, editorUrl, t])

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

  const openEditor = (next: EditorState | null) => {
    setDupeError(null)
    setEditor(next)
  }

  const save = useCallback(
    async (allowPlainTextToken = false) => {
      if (!bridge || !editor) {
        return
      }

      // Duplicate prevention lives in the save path (not just a disabled
      // button): reject a candidate that collides with an existing entry with
      // an inline error before anything crosses the IPC boundary.
      const dupe = findDuplicateConnection(editor, registry?.connections ?? [])

      if (dupe) {
        setDupeError(
          editor.kind === 'local'
            ? s.duplicateLocal
            : editor.kind === 'ssh'
              ? s.duplicateSsh(dupe.label)
              : s.duplicateUrl(dupe.label)
        )

        return
      }

      setDupeError(null)
      setSaving(true)

      try {
        const payload: DesktopRegistryConnectionInput = {
          kind: editor.kind,
          label: editor.label
        }

        if (editor.id) {
          payload.id = editor.id
        }

        if (editor.kind === 'remote' || editor.kind === 'cloud') {
          payload.url = editor.url
          payload.authMode = editor.authMode

          if (editor.token.trim()) {
            payload.token = editor.token.trim()
          }

          if (allowPlainTextToken) {
            payload.allowPlainTextToken = true
          }

          // Authoritative header map: typed value → new secret; empty value on
          // a stored row → null (keep saved secret); a removed row is simply
          // absent, which clears it server-side.
          const headerEntries = editor.headers
            .map(row => ({ name: row.name.trim(), stored: row.stored, value: row.value.trim() }))
            .filter(row => row.name)

          if (headerEntries.length > 0 || editor.headers.length === 0) {
            payload.headers = Object.fromEntries(
              headerEntries.map(row => [row.name, row.value ? row.value : row.stored ? null : ''])
            )
          }
        } else if (editor.kind === 'ssh') {
          // The composite host string (user@host:port) is the single source
          // of truth — never send separate user/port (see editorFromConnection).
          payload.host = editor.host
          payload.keyPath = editor.keyPath || undefined
        }

        const result = await bridge.save(payload)
        publishRegistry(result.registry)
        setEditor(null)
        setPlainTextConfirm(false)
      } catch (err) {
        // Keyring-less machine and the user hasn't consented to plain-text
        // storage yet: raise the same opt-in dialog the connection-mode form
        // uses instead of dead-ending the save.
        if (
          !allowPlainTextToken &&
          registry?.secureTokenStorage === false &&
          editor.kind === 'remote' &&
          editor.authMode === 'token' &&
          editor.token.trim()
        ) {
          setPlainTextConfirm(true)

          return
        }

        notifyError(err, s.saveFailed)
      } finally {
        setSaving(false)
      }
    },
    [bridge, editor, publishRegistry, registry?.connections, registry?.secureTokenStorage, s]
  )

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

  const kindMeta: Record<DesktopConnectionKind, { label: string; desc: string }> = {
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
    <div className="mt-8 border-t border-border/60 pt-6">
      <SectionHeading icon={Globe} title={s.title} />
      <p className="mb-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{s.intro}</p>
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
        <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <Loader2 className="size-4 animate-spin" />
        </div>
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
                    {testingId === conn.id ? <Loader2 className="size-3.5 animate-spin" /> : s.testConnection}
                  </Button>
                  {!isPrimary && (
                    <Button disabled={busy} onClick={() => void makePrimary(conn.id)} size="sm" variant="outline">
                      {s.makePrimary}
                    </Button>
                  )}
                  {conn.kind !== 'local' && (
                    <>
                      <Button
                        aria-label={s.editConnection}
                        onClick={() => openEditor(editorFromConnection(conn))}
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
                        {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
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

      {editor ? (
        <div className="mt-4 space-y-3 rounded-lg border border-border/60 p-4">
          <div className="grid grid-cols-2 gap-2 @2xl:grid-cols-4">
            {/* Kind is fixed once created (buttons disable on edit). On create
                every kind is offered; Local is disabled while the managed
                local entry exists (the registry holds at most one). */}
            {(editor.id ? ([editor.kind] as const) : (['local', 'cloud', 'remote', 'ssh'] as const)).map(kind => (
              <Button
                disabled={Boolean(editor.id) || (kind === 'local' && hasLocal)}
                key={kind}
                onClick={() => {
                  setDupeError(null)
                  setEditor({ ...editor, kind })
                }}
                size="sm"
                variant={editor.kind === kind ? 'default' : 'outline'}
              >
                {kindMeta[kind].label}
              </Button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">{kindMeta[editor.kind].desc}</p>
          {!editor.id && hasLocal ? <p className="text-xs text-muted-foreground">{s.localAddHint}</p> : null}
          {!editor.id && editor.kind === 'cloud' ? (
            <p className="text-xs text-muted-foreground">{s.cloudAddHint}</p>
          ) : null}

          <ListRow
            action={
              <Input
                onChange={e => setEditor({ ...editor, label: e.target.value })}
                placeholder={s.labelPlaceholder}
                value={editor.label}
              />
            }
            description={s.labelDesc}
            title={s.labelTitle}
          />

          {(editor.kind === 'remote' || editor.kind === 'cloud') && (
            <ListRow
              action={
                <Input
                  onChange={e => {
                    setDupeError(null)
                    setEditor({ ...editor, url: e.target.value })
                  }}
                  placeholder="http://homelab.lan:9119"
                  value={editor.url}
                />
              }
              title={s.urlTitle}
            />
          )}

          {editor.kind === 'remote' && (
            <>
              <ListRow
                action={
                  <div className="flex gap-2">
                    {(['token', 'oauth'] as const).map(mode => (
                      <Button
                        key={mode}
                        onClick={() => setEditor({ ...editor, authMode: mode })}
                        size="sm"
                        variant={editor.authMode === mode ? 'default' : 'outline'}
                      >
                        {mode === 'token' ? t.settings.gateway.tokenTitle : 'OAuth'}
                      </Button>
                    ))}
                  </div>
                }
                title={t.settings.gateway.authTitle}
              />
              {editor.authMode === 'token' && (
                <ListRow
                  action={
                    <Input
                      onChange={e => setEditor({ ...editor, token: e.target.value })}
                      placeholder={t.settings.gateway.pasteSessionToken}
                      type="password"
                      value={editor.token}
                    />
                  }
                  description={t.settings.gateway.tokenDesc}
                  title={t.settings.gateway.tokenTitle}
                />
              )}
              {editor.authMode === 'oauth' && (
                <ListRow
                  action={
                    oauthConnected ? (
                      <Pill tone="primary">
                        <Check className="size-3" /> {t.settings.gateway.signedIn}
                      </Pill>
                    ) : (
                      <Button disabled={signingIn || !editorUrl} onClick={() => void signInOauth()} size="sm">
                        {signingIn ? <Loader2 className="size-4 animate-spin" /> : <LogIn className="size-4" />}
                        {authProviderShape.isPassword
                          ? t.settings.gateway.signIn
                          : t.settings.gateway.signInWith(authProviderShape.providerLabel)}
                      </Button>
                    )
                  }
                  description={
                    oauthConnected
                      ? authProviderShape.isPassword
                        ? t.settings.gateway.authSignedInPassword
                        : t.settings.gateway.authSignedInOauth
                      : authProviderShape.isPassword
                        ? t.settings.gateway.authNeedsPassword
                        : t.settings.gateway.authNeedsOauth(authProviderShape.providerLabel)
                  }
                  title={t.settings.gateway.authTitle}
                />
              )}
            </>
          )}

          {(editor.kind === 'remote' || editor.kind === 'cloud') && (
            <div className="grid gap-2">
              <div>
                <div className="text-sm font-medium">{s.headersTitle}</div>
                <p className="mt-1 text-xs text-muted-foreground">{s.headersDesc}</p>
              </div>
              {editor.headers.map((row, index) => (
                <div className="flex items-center gap-2" key={index}>
                  <Input
                    className="flex-1"
                    onChange={e =>
                      setEditor({
                        ...editor,
                        headers: editor.headers.map((h, i) => (i === index ? { ...h, name: e.target.value } : h))
                      })
                    }
                    placeholder="CF-Access-Client-Id"
                    value={row.name}
                  />
                  <Input
                    className="flex-1"
                    onChange={e =>
                      setEditor({
                        ...editor,
                        headers: editor.headers.map((h, i) => (i === index ? { ...h, value: e.target.value } : h))
                      })
                    }
                    placeholder={row.stored ? s.headerValueSaved : s.headerValuePlaceholder}
                    type="password"
                    value={row.value}
                  />
                  <Button
                    onClick={() => setEditor({ ...editor, headers: editor.headers.filter((_, i) => i !== index) })}
                    size="sm"
                    variant="ghost"
                  >
                    {s.headerRemove}
                  </Button>
                </div>
              ))}
              <div>
                <Button
                  onClick={() =>
                    setEditor({ ...editor, headers: [...editor.headers, { name: '', stored: false, value: '' }] })
                  }
                  size="sm"
                  variant="outline"
                >
                  {s.headerAdd}
                </Button>
              </div>
            </div>
          )}

          {editor.kind === 'ssh' && (
            <ListRow
              action={
                <Input
                  onChange={e => {
                    setDupeError(null)
                    setEditor({ ...editor, host: e.target.value })
                  }}
                  placeholder="user@host:22"
                  value={editor.host}
                />
              }
              title={s.sshHostTitle}
            />
          )}

          {dupeError ? <p className="text-xs text-destructive">{dupeError}</p> : null}

          <div className="flex justify-end gap-2">
            <Button disabled={saving} onClick={() => openEditor(null)} size="sm" variant="ghost">
              {s.cancel}
            </Button>
            <Button disabled={saving || !editor.label.trim()} onClick={() => void save()} size="sm">
              {saving ? s.saving : s.save}
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex items-center gap-2">
          <Button
            onClick={() => {
              triggerHaptic('selection')
              openEditor(emptyEditor('remote'))
            }}
            size="sm"
            variant="outline"
          >
            <Plus className="size-3.5" /> {s.addConnection}
          </Button>
          {bridge?.updateAll && (registry?.connections.length ?? 0) > 1 && (
            <Button
              disabled={updatingAll}
              onClick={() => {
                triggerHaptic('selection')
                void updateAll()
              }}
              size="sm"
              variant="outline"
            >
              {updatingAll ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" /> {s.updateAllRunning}
                </>
              ) : (
                <>
                  <RefreshCw className="size-3.5" /> {s.updateAll}
                </>
              )}
            </Button>
          )}
        </div>
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

      {/* Keyring-less opt-in: same consent flow as the connection-mode form. */}
      <ConfirmDialog
        confirmLabel={t.settings.gateway.plainTextConfirmAction}
        description={t.settings.gateway.plainTextConfirmDesc}
        destructive
        onClose={() => setPlainTextConfirm(false)}
        onConfirm={() => save(true)}
        open={plainTextConfirm}
        title={t.settings.gateway.plainTextConfirmTitle}
      />
    </div>
  )
}
