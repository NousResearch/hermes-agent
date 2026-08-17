import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Input } from '@/components/ui/input'
import type {
  DesktopConnectionKind,
  DesktopConnectionsRegistry,
  DesktopRegistryConnection,
  DesktopRegistryConnectionInput
} from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Cloud, Globe, Loader2, Monitor, Pencil, Plus, RefreshCw, Terminal, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { EmptyState, ListRow, Pill, SectionHeading, SettingsContent, SettingsSkeleton } from './primitives'

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
    headers: (conn.headerNames || []).map(name => ({ name, stored: true, value: '' }))
  }
}

function emptyEditor(kind: DesktopConnectionKind): EditorState {
  return { id: null, kind, label: '', url: '', authMode: 'token', token: '', host: '', keyPath: '', headers: [] }
}

/**
 * Settings → Connections: manage the registry of named agent sources (local
 * runtime + any number of remote gateways / Hermes Cloud instances / SSH
 * hosts). Storage-level management only — the active/primary switchover UX
 * stays in Settings → Gateway until the routing generalization lands.
 */
export function ConnectionsSettings() {
  const { t } = useI18n()
  const s = t.settings.connections
  const [registry, setRegistry] = useState<DesktopConnectionsRegistry | null>(null)
  const [loading, setLoading] = useState(true)
  const [editor, setEditor] = useState<EditorState | null>(null)
  const [saving, setSaving] = useState(false)
  const [busyId, setBusyId] = useState<null | string>(null)
  const [testingId, setTestingId] = useState<null | string>(null)
  const [removeTarget, setRemoveTarget] = useState<DesktopRegistryConnection | null>(null)
  const [plainTextConfirm, setPlainTextConfirm] = useState(false)
  const [updatingAll, setUpdatingAll] = useState(false)

  const bridge = window.hermesDesktop?.connections

  const load = useCallback(async () => {
    if (!bridge) {
      setLoading(false)

      return
    }

    setLoading(true)

    try {
      setRegistry(await bridge.list())
    } catch (err) {
      notifyError(err, s.loadFailed)
    } finally {
      setLoading(false)
    }
  }, [bridge, s.loadFailed])

  useEffect(() => {
    void load()
  }, [load])

  const save = useCallback(
    async (allowPlainTextToken = false) => {
      if (!bridge || !editor) {
        return
      }

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
        setRegistry(result.registry)
        setEditor(null)
        setPlainTextConfirm(false)
      } catch (err) {
        // Keyring-less machine and the user hasn't consented to plain-text
        // storage yet: raise the same opt-in dialog Settings → Gateway uses
        // instead of dead-ending the save.
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
    [bridge, editor, registry?.secureTokenStorage, s.saveFailed]
  )

  const remove = useCallback(async () => {
    if (!bridge || !removeTarget) {
      return
    }

    setBusyId(removeTarget.id)

    try {
      const result = await bridge.remove(removeTarget.id)
      setRegistry(result.registry)
    } catch (err) {
      notifyError(err, s.removeFailed)
    } finally {
      setBusyId(null)
      setRemoveTarget(null)
    }
  }, [bridge, removeTarget, s.removeFailed])

  const makePrimary = useCallback(
    async (id: string) => {
      if (!bridge) {
        return
      }

      setBusyId(id)

      try {
        const result = await bridge.setPrimary(id)
        setRegistry(result.registry)
      } catch (err) {
        notifyError(err, s.saveFailed)
      } finally {
        setBusyId(null)
      }
    },
    [bridge, s.saveFailed]
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

  if (loading) {
    return <SettingsSkeleton />
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Globe} title={s.title} />
      <p className="mb-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{s.intro}</p>
      {/* Storage-only slice: be explicit that routing consumption is staged so
          "Make primary" isn't read as an immediate connection switch. */}
      <p className="mb-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.stagedNote}
      </p>

      {!registry || registry.connections.length === 0 ? (
        <EmptyState title={s.empty} />
      ) : (
        registry.connections.map(conn => {
          const Icon = KIND_ICONS[conn.kind]
          const isPrimary = registry.primary === conn.id
          const busy = busyId === conn.id

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
                        onClick={() => setEditor(editorFromConnection(conn))}
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
                conn.kind === 'ssh'
                  ? `${kindMeta[conn.kind].label} · ${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}`
                  : conn.url
                    ? `${kindMeta[conn.kind].label} · ${conn.url}`
                    : kindMeta[conn.kind].desc
              }
              key={conn.id}
              title={
                <span className="flex items-center gap-2">
                  <Icon className="size-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{conn.label}</span>
                  {isPrimary && <Pill tone="primary">{s.primaryPill}</Pill>}
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
            {/* Cloud creation is deliberately absent: a dialable cloud entry
                comes from the Hermes Cloud sign-in/discovery flow (Settings →
                Gateway), not a hand-typed URL. Migrated/discovered cloud
                entries remain editable (kind buttons are disabled on edit). */}
            {(editor.kind === 'cloud' ? (['cloud'] as const) : (['remote', 'ssh'] as const)).map(kind => (
              <Button
                disabled={Boolean(editor.id)}
                key={kind}
                onClick={() => setEditor({ ...editor, kind })}
                size="sm"
                variant={editor.kind === kind ? 'default' : 'outline'}
              >
                {kindMeta[kind].label}
              </Button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">{kindMeta[editor.kind].desc}</p>

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
                  onChange={e => setEditor({ ...editor, url: e.target.value })}
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
                  onChange={e => setEditor({ ...editor, host: e.target.value })}
                  placeholder="user@host:22"
                  value={editor.host}
                />
              }
              title={s.sshHostTitle}
            />
          )}

          <div className="flex justify-end gap-2">
            <Button disabled={saving} onClick={() => setEditor(null)} size="sm" variant="ghost">
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
              setEditor(emptyEditor('remote'))
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

      <ConfirmDialog
        confirmLabel={s.removeConnection}
        description={removeTarget ? s.removeConfirmDesc(removeTarget.label) : ''}
        destructive
        onClose={() => setRemoveTarget(null)}
        onConfirm={() => remove()}
        open={Boolean(removeTarget)}
        title={s.removeConfirmTitle}
      />

      {/* Keyring-less opt-in: same consent flow as Settings → Gateway. */}
      <ConfirmDialog
        confirmLabel={t.settings.gateway.plainTextConfirmAction}
        description={t.settings.gateway.plainTextConfirmDesc}
        destructive
        onClose={() => setPlainTextConfirm(false)}
        onConfirm={() => save(true)}
        open={plainTextConfirm}
        title={t.settings.gateway.plainTextConfirmTitle}
      />
    </SettingsContent>
  )
}
