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
import { Cloud, Globe, Loader2, Monitor, Pencil, Plus, Terminal, Trash2 } from '@/lib/icons'
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
  user: string
  port: string
  keyPath: string
}

function editorFromConnection(conn: DesktopRegistryConnection): EditorState {
  return {
    id: conn.id,
    kind: conn.kind,
    label: conn.label,
    url: conn.url || '',
    authMode: conn.authMode || 'token',
    token: '',
    host: conn.host || '',
    user: conn.user || '',
    port: conn.port ? String(conn.port) : '',
    keyPath: conn.keyPath || ''
  }
}

function emptyEditor(kind: DesktopConnectionKind): EditorState {
  return { id: null, kind, label: '', url: '', authMode: 'token', token: '', host: '', user: '', port: '', keyPath: '' }
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

  const save = useCallback(async () => {
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
      } else if (editor.kind === 'ssh') {
        payload.host = editor.host
        payload.user = editor.user || undefined
        payload.port = editor.port.trim() ? Number(editor.port) : null
        payload.keyPath = editor.keyPath || undefined
      }

      const result = await bridge.save(payload)
      setRegistry(result.registry)
      setEditor(null)
    } catch (err) {
      notifyError(err, s.saveFailed)
    } finally {
      setSaving(false)
    }
  }, [bridge, editor, s.saveFailed])

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
          notifyError(new Error(result.error || s.testFailed), conn.label)
        }
      } catch (err) {
        notifyError(err, s.testFailed)
      } finally {
        setTestingId(null)
      }
    },
    [bridge, s.testFailed, s.testOk]
  )

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
      <p className="mb-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{s.intro}</p>

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
            {(['remote', 'cloud', 'ssh'] as const).map(kind => (
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
        <div className="mt-4">
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
    </SettingsContent>
  )
}
