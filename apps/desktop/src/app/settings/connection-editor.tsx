import { useCallback, useRef, useState } from 'react'

import { RemoteSetupFields } from '@/components/remote-setup/fields'
import { type RemoteCredentials, type RemoteSetup, useRemoteSetup } from '@/components/remote-setup/use-remote-setup'
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
import { Loader2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, Pill } from './primitives'

export interface EditorState {
  // null id → creating a new connection.
  id: null | string
  kind: DesktopConnectionKind
  label: string
  host: string
  keyPath: string
  remoteHermesPath: string
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
    // Reconstruct the composite the single ssh host field displays. The save
    // payload sends ONLY this string (never separate user/port), because
    // normalizeSshConfig gives explicit user/port fields precedence over the
    // parsed host string — sending stored user/port alongside a retyped host
    // would silently resurrect the old values.
    host: conn.host ? `${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}` : '',
    keyPath: conn.keyPath || '',
    remoteHermesPath: conn.remoteHermesPath || '',
    remoteProfile: conn.remoteProfile || '',
    headers: (conn.headerNames || []).map(name => ({ name, stored: true, value: '' }))
  }
}

function emptyEditor(kind: DesktopConnectionKind): EditorState {
  return {
    id: null,
    kind,
    label: '',
    host: '',
    keyPath: '',
    remoteHermesPath: '',
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
  editor: Pick<EditorState, 'host' | 'id' | 'kind' | 'remoteProfile'> & { url: string },
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
 * Authoritative header map: typed value → new secret; empty value on a stored row → null (keep
 * saved secret); a removed row is simply absent, which clears it server-side. Undefined leaves the
 * saved headers alone (only blank rows were left).
 */
function headersPayload(headers: EditorState['headers']): Record<string, null | string> | undefined {
  const entries = headers
    .map(row => ({ name: row.name.trim(), stored: row.stored, value: row.value.trim() }))
    .filter(row => row.name)

  if (entries.length === 0 && headers.length > 0) {
    return undefined
  }

  return Object.fromEntries(entries.map(row => [row.name, row.value ? row.value : row.stored ? null : '']))
}

/** The save request for the draft. */
function connectionPayload(
  editor: EditorState,
  remote: Pick<RemoteSetup, 'credentials' | 'payload'>,
  allowPlainTextToken: boolean
): DesktopRegistryConnectionInput {
  const payload: DesktopRegistryConnectionInput = { kind: editor.kind, label: editor.label }

  if (editor.id) {
    payload.id = editor.id
  }

  if (editor.kind === 'remote' || editor.kind === 'cloud') {
    payload.url = remote.payload.remoteUrl
    payload.authMode = editor.kind === 'cloud' ? 'oauth' : remote.credentials.authMode

    if (editor.kind === 'remote' && remote.payload.remoteToken) {
      payload.token = remote.payload.remoteToken
    }

    if (allowPlainTextToken) {
      payload.allowPlainTextToken = true
    }

    const headers = headersPayload(editor.headers)

    if (headers) {
      payload.headers = headers
    }
  } else if (editor.kind === 'ssh') {
    // The composite host string (user@host:port) is the single source
    // of truth — never send separate user/port (see editorFromConnection).
    payload.host = editor.host
    payload.keyPath = editor.keyPath || undefined
    payload.remoteHermesPath = editor.remoteHermesPath.trim()
  }

  return payload
}

/** What the editor opens on: the draft, and the saved credentials of the connection it edits. */
export interface ConnectionDraft {
  credentials: Partial<RemoteCredentials>
  editor: EditorState
}

/** A draft to edit `conn`, or (no `conn`) to add a new connection. */
export function connectionDraft(conn?: DesktopRegistryConnection): ConnectionDraft {
  return {
    credentials: {
      url: conn?.url || '',
      authMode: conn?.kind === 'cloud' ? 'oauth' : conn?.authMode || 'token',
      tokenSet: conn?.tokenSet ?? false,
      tokenPreview: conn?.tokenPreview ?? null
    },
    editor: conn ? editorFromConnection(conn) : emptyEditor('remote')
  }
}

export type KindMeta = Record<DesktopConnectionKind, { desc: string; label: string }>

interface ConnectionEditorProps {
  bridge: NonNullable<Window['hermesDesktop']['connections']>
  draft: ConnectionDraft
  kindMeta: KindMeta
  onClose: () => void
  onSaved: (registry: DesktopConnectionsRegistry) => void
  registry: DesktopConnectionsRegistry | null
  /** A save is in flight, from this draft or one opened before it. */
  saving: boolean
  /** Runs one save under the list's save lock (`useSaveLock`); a call while one runs does nothing. */
  runSave: (work: () => Promise<void>) => Promise<void>
}

/** A Cloud draft signs in before it saves: the account's browser sign-in, or its password. */
function CloudSignInRow({ remote, saving }: { remote: RemoteSetup; saving: boolean }) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const connected = remote.credentials.oauthConnected

  const description = connected
    ? remote.isPassword
      ? g.authSignedInPassword
      : g.authSignedInOauth
    : remote.isPassword
      ? g.authNeedsPassword
      : g.authNeedsOauth(remote.providerLabel)

  return (
    <ListRow
      action={
        connected ? (
          <Pill tone="primary">{g.signedIn}</Pill>
        ) : (
          <Button
            disabled={saving || remote.signingIn || !remote.payload.remoteUrl}
            onClick={() => void remote.signIn()}
            size="sm"
          >
            {remote.signingIn ? <Loader2 className="animate-spin" /> : null}
            {remote.isPassword ? g.signIn : g.signInWith(remote.providerLabel)}
          </Button>
        )
      }
      description={description}
      title={g.authTitle}
    />
  )
}

/** Extra gateway headers (access proxies such as Cloudflare Access), one name/secret pair a row. */
function HeaderRows({
  headers,
  onChange
}: {
  headers: EditorState['headers']
  onChange: (headers: EditorState['headers']) => void
}) {
  const { t } = useI18n()
  const s = t.settings.connections

  const update = (index: number, patch: Partial<EditorState['headers'][number]>) =>
    onChange(headers.map((h, i) => (i === index ? { ...h, ...patch } : h)))

  return (
    <div className="grid gap-2">
      <div>
        <div className="text-sm font-medium">{s.headersTitle}</div>
        <p className="mt-1 text-xs text-muted-foreground">{s.headersDesc}</p>
      </div>
      {headers.map((row, index) => (
        <div className="flex items-center gap-2" key={index}>
          <Input
            className="flex-1"
            onChange={e => update(index, { name: e.target.value })}
            placeholder="CF-Access-Client-Id"
            value={row.name}
          />
          <Input
            className="flex-1"
            onChange={e => update(index, { value: e.target.value })}
            placeholder={row.stored ? s.headerValueSaved : s.headerValuePlaceholder}
            type="password"
            value={row.value}
          />
          <Button onClick={() => onChange(headers.filter((_, i) => i !== index))} size="sm" variant="ghost">
            {s.headerRemove}
          </Button>
        </div>
      ))}
      <div>
        <Button
          onClick={() => onChange([...headers, { name: '', stored: false, value: '' }])}
          size="sm"
          variant="outline"
        >
          {s.headerAdd}
        </Button>
      </div>
    </div>
  )
}

/**
 * The add/edit form of Settings → Gateways: the draft, its credentials and sign-in, and the save
 * with its plain-text consent. Mounted only while open, and only on a list this page may edit.
 */
export function ConnectionEditor({
  bridge,
  draft,
  kindMeta,
  onClose,
  onSaved,
  registry,
  runSave,
  saving
}: ConnectionEditorProps) {
  const { t } = useI18n()
  const s = t.settings.connections
  const [editor, setEditor] = useState<EditorState>(draft.editor)
  const [plainTextConfirm, setPlainTextConfirm] = useState(false)
  // Inline duplicate rejection from the save path (dedupe is also enforced in
  // the main process, so a crafted payload can't slip past the UI check).
  const [dupeError, setDupeError] = useState<null | string>(null)

  // The draft's auth mode lives inside the remote-setup state below; the
  // sign-in identity callback runs at login time (after that state exists),
  // so it reads the mode through this ref rather than the render-time closure.
  const remoteAuthModeRef = useRef<'oauth' | 'token'>('token')

  const remote = useRemoteSetup({
    host: 'registry',
    enabled: editor.kind === 'remote' || editor.kind === 'cloud',
    initial: draft.credentials,
    onNotice: notify,
    // The registry draft can sign in BEFORE it is saved: the login carries
    // the draft's identity so the main process settles the id the save will
    // reuse and writes the session into the jar the saved connection reads.
    // The kind/authMode gate in oauth-partition.ts decides WHICH jar that is:
    // a cookie-auth remote draft gets its own; cloud and token drafts share
    // the legacy jar, exactly what they resolve to after the save. Pin the
    // settled id into the draft so Save reuses it.
    oauthLoginIdentity: () => ({
      connectionId: editor.id,
      label: editor.label,
      kind: editor.kind,
      authMode: editor.kind === 'cloud' ? 'oauth' : remoteAuthModeRef.current
    }),
    onOAuthLoginSettled: settledId => {
      setEditor(prev => (prev.id ? prev : { ...prev, id: settledId }))
    }
  })

  remoteAuthModeRef.current = remote.credentials.authMode

  const hasLocal = Boolean(registry?.connections.some(c => c.kind === 'local'))

  const save = useCallback(
    async (allowPlainTextToken: boolean = false): Promise<void> => {
      // Duplicate prevention lives in the save path (not just a disabled
      // button): reject a candidate that collides with an existing entry with
      // an inline error before anything crosses the IPC boundary.
      const dupe = findDuplicateConnection({ ...editor, url: remote.credentials.url }, registry?.connections ?? [])

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

      await runSave(async () => {
        try {
          const payload = connectionPayload(editor, remote, allowPlainTextToken)

          const result = await bridge.save(payload)
          onSaved(result.registry)
        } catch (err) {
          // Keyring-less machine and the user hasn't consented to plain-text
          // storage yet: raise the same opt-in dialog the connection-mode form
          // uses instead of dead-ending the save.
          if (
            !allowPlainTextToken &&
            registry?.secureTokenStorage === false &&
            editor.kind === 'remote' &&
            remote.credentials.authMode === 'token' &&
            remote.credentials.token.trim()
          ) {
            setPlainTextConfirm(true)

            return
          }

          notifyError(err, s.saveFailed)
        }
      })
    },
    [bridge, editor, onSaved, remote, registry?.connections, registry?.secureTokenStorage, runSave, s]
  )

  return (
    <>
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

                // Cloud uses browser sign-in even after a token-auth remote edit.
                if (kind === 'cloud') {
                  remote.setAuthMode('oauth')
                }

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
          <RemoteSetupFields
            disabled={saving}
            onUrlChange={() => setDupeError(null)}
            setup={remote}
            urlOnly={editor.kind === 'cloud'}
          />
        )}

        {editor.kind === 'cloud' && <CloudSignInRow remote={remote} saving={saving} />}

        {(editor.kind === 'remote' || editor.kind === 'cloud') && (
          <HeaderRows headers={editor.headers} onChange={headers => setEditor({ ...editor, headers })} />
        )}

        {editor.kind === 'ssh' && (
          <>
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
            <ListRow
              action={
                <Input
                  onChange={e => setEditor({ ...editor, remoteHermesPath: e.target.value })}
                  placeholder={t.settings.gateway.sshHermesPathPlaceholder}
                  value={editor.remoteHermesPath}
                />
              }
              description={t.settings.gateway.sshHermesPathDesc}
              title={t.settings.gateway.sshHermesPathTitle}
            />
          </>
        )}

        {dupeError ? <p className="text-xs text-destructive">{dupeError}</p> : null}

        <div className="flex justify-end gap-2">
          <Button disabled={saving} onClick={onClose} size="sm" variant="ghost">
            {s.cancel}
          </Button>
          <Button disabled={saving || !editor.label.trim()} onClick={() => void save()} size="sm">
            {saving ? s.saving : s.save}
          </Button>
        </div>
      </div>

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
    </>
  )
}
