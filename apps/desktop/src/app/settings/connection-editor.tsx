import { RemoteSetupFields } from '@/components/remote-setup/fields'
import type { useRemoteSetup } from '@/components/remote-setup/use-remote-setup'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'

import type { ConnectionKindMeta } from './connection-row'
import type { EditorState } from './connections-registry'
import { ListRow, Pill } from './primitives'

interface ConnectionEditorFieldsProps {
  editor: EditorState
  hasLocal: boolean
  kindMeta: ConnectionKindMeta
  remote: ReturnType<typeof useRemoteSetup>
  saving: boolean
  dupeError: null | string
  onEditorChange: (editor: EditorState) => void
  onDupeErrorChange: (error: null | string) => void
  onCancel: () => void
  onSave: () => void
}

/** The add/edit form for one saved connection: kind, name, endpoint and auth, headers, SSH host. */
export function ConnectionEditorFields({
  editor,
  hasLocal,
  kindMeta,
  remote,
  saving,
  dupeError,
  onEditorChange,
  onDupeErrorChange,
  onCancel,
  onSave
}: ConnectionEditorFieldsProps) {
  const { t } = useI18n()
  const s = t.settings.connections

  return (
    <>
      <div className="grid grid-cols-2 gap-2 @2xl:grid-cols-4">
        {/* Kind is fixed once created (buttons disable on edit). On create
                every kind is offered; Local is disabled while the managed
                local entry exists (the registry holds at most one). */}
        {(editor.id ? ([editor.kind] as const) : (['local', 'cloud', 'remote', 'ssh'] as const)).map(kind => (
          <Button
            disabled={Boolean(editor.id) || (kind === 'local' && hasLocal)}
            key={kind}
            onClick={() => {
              onDupeErrorChange(null)

              // Cloud uses browser sign-in even after a token-auth remote edit.
              if (kind === 'cloud') {
                remote.setAuthMode('oauth')
              }

              onEditorChange({ ...editor, kind })
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
      {!editor.id && editor.kind === 'cloud' ? <p className="text-xs text-muted-foreground">{s.cloudAddHint}</p> : null}

      <ListRow
        action={
          <Input
            onChange={e => onEditorChange({ ...editor, label: e.target.value })}
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
          onUrlChange={() => onDupeErrorChange(null)}
          setup={remote}
          urlOnly={editor.kind === 'cloud'}
        />
      )}

      {editor.kind === 'cloud' && <CloudSignInRow remote={remote} saving={saving} />}

      {(editor.kind === 'remote' || editor.kind === 'cloud') && (
        <ConnectionHeadersField headers={editor.headers} onChange={headers => onEditorChange({ ...editor, headers })} />
      )}

      {editor.kind === 'ssh' && (
        <>
          <ListRow
            action={
              <Input
                onChange={e => {
                  onDupeErrorChange(null)
                  onEditorChange({ ...editor, host: e.target.value })
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
                onChange={e => onEditorChange({ ...editor, remoteHermesPath: e.target.value })}
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
        <Button disabled={saving} onClick={onCancel} size="sm" variant="ghost">
          {s.cancel}
        </Button>
        <Button disabled={saving || !editor.label.trim()} onClick={onSave} size="sm">
          {saving ? s.saving : s.save}
        </Button>
      </div>
    </>
  )
}

interface ConnectionHeadersFieldProps {
  headers: EditorState['headers']
  onChange: (headers: EditorState['headers']) => void
}

/** Extra HTTP headers sent to a remote or cloud gateway (access proxies such as Cloudflare Access). */
function ConnectionHeadersField({ headers, onChange }: ConnectionHeadersFieldProps) {
  const { t } = useI18n()
  const s = t.settings.connections

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
            onChange={e => onChange(headers.map((h, i) => (i === index ? { ...h, name: e.target.value } : h)))}
            placeholder="CF-Access-Client-Id"
            value={row.name}
          />
          <Input
            className="flex-1"
            onChange={e => onChange(headers.map((h, i) => (i === index ? { ...h, value: e.target.value } : h)))}
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

/** Hermes Cloud drafts sign in through the browser before they are saved. */
function CloudSignInRow({ remote, saving }: { remote: ReturnType<typeof useRemoteSetup>; saving: boolean }) {
  const { t } = useI18n()

  return (
    <ListRow
      action={
        remote.credentials.oauthConnected ? (
          <Pill tone="primary">{t.settings.gateway.signedIn}</Pill>
        ) : (
          <Button
            disabled={saving || remote.signingIn || !remote.payload.remoteUrl}
            onClick={() => void remote.signIn()}
            size="sm"
          >
            {remote.signingIn ? <Loader2 className="animate-spin" /> : null}
            {remote.isPassword ? t.settings.gateway.signIn : t.settings.gateway.signInWith(remote.providerLabel)}
          </Button>
        )
      }
      description={
        remote.credentials.oauthConnected
          ? remote.isPassword
            ? t.settings.gateway.authSignedInPassword
            : t.settings.gateway.authSignedInOauth
          : remote.isPassword
            ? t.settings.gateway.authNeedsPassword
            : t.settings.gateway.authNeedsOauth(remote.providerLabel)
      }
      title={t.settings.gateway.authTitle}
    />
  )
}
