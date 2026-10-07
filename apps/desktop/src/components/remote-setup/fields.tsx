import type { ReactElement, ReactNode } from 'react'

import { ListRow, Pill } from '@/app/settings/primitives'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertCircle, Check, Loader2, LogIn } from '@/lib/icons'

import type { RemoteSetup } from './use-remote-setup'

interface FieldProps {
  stacked: boolean
  title: string
  description?: string
  children: ReactNode
}

function Field({ stacked, title, description, children }: FieldProps): ReactElement {
  return stacked ? (
    <div className="grid gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{title}</span>
      {children}
      {description ? <span className="text-xs text-muted-foreground">{description}</span> : null}
    </div>
  ) : (
    <ListRow action={children} description={description} title={title} />
  )
}

interface RemoteSetupFieldsProps {
  setup: RemoteSetup
  disabled?: boolean
  /** Pins only the URL input: an env-owned remote (HERMES_DESKTOP_REMOTE_URL) still signs in here. */
  urlDisabled?: boolean
  urlOnly?: boolean
  onUrlChange?: () => void
}

export function RemoteSetupFields({
  setup,
  disabled = false,
  urlDisabled = false,
  urlOnly = false,
  onUrlChange
}: RemoteSetupFieldsProps): ReactElement {
  const { t } = useI18n()
  const g = t.settings.gateway
  const firstRun = setup.host === 'first-run'
  const registry = setup.host === 'registry'
  const copy = firstRun ? t.install : g
  const { credentials, isPassword, providerLabel } = setup
  const urlTitle = registry ? t.settings.connections.urlTitle : copy.remoteUrlTitle

  const authDescription = firstRun
    ? credentials.oauthConnected
      ? t.install.authSignedIn
      : t.install.authNeedsOauth(providerLabel)
    : credentials.oauthConnected
      ? isPassword
        ? g.authSignedInPassword
        : g.authSignedInOauth
      : isPassword
        ? g.authNeedsPassword
        : g.authNeedsOauth(providerLabel)

  return (
    <div className="grid gap-3">
      <Field description={registry ? undefined : copy.remoteUrlDesc} stacked={firstRun} title={urlTitle}>
        <Input
          aria-label={urlTitle}
          autoComplete="url"
          disabled={disabled || urlDisabled}
          onChange={event => {
            setup.setUrl(event.target.value)
            onUrlChange?.()
          }}
          placeholder={registry ? 'http://homelab.lan:9119' : t.install.remoteUrlPlaceholder}
          value={credentials.url}
        />
      </Field>
      {!registry && setup.probeStatus === 'probing' ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          {copy.probing}
        </div>
      ) : null}
      {!registry && setup.probeStatus === 'error' ? (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="size-4" />
          {copy.probeError}
        </div>
      ) : null}
      {registry && !urlOnly ? (
        <Field stacked={false} title={g.authTitle}>
          <div className="flex gap-2">
            {(['token', 'oauth'] as const).map(mode => (
              <Button
                disabled={disabled}
                key={mode}
                onClick={() => setup.setAuthMode(mode)}
                size="sm"
                variant={credentials.authMode === mode ? 'default' : 'outline'}
              >
                {mode === 'token' ? g.tokenTitle : g.signIn}
              </Button>
            ))}
          </div>
        </Field>
      ) : null}
      {!urlOnly && setup.authResolved && credentials.authMode === 'oauth' ? (
        <Field description={authDescription} stacked={firstRun} title={copy.authTitle}>
          {credentials.oauthConnected ? (
            <div className="flex items-center gap-2">
              <Pill tone="primary">
                <Check className="size-3" />
                {firstRun ? t.install.connected : g.signedIn}
              </Pill>
              {setup.host === 'settings' ? (
                <Button disabled={disabled || setup.signingIn} onClick={() => void setup.signOut()} variant="outline">
                  {g.signOut}
                </Button>
              ) : null}
            </div>
          ) : (
            <Button
              disabled={disabled || setup.signingIn || !setup.payload.remoteUrl}
              onClick={() => void setup.signIn()}
              size="sm"
            >
              {setup.signingIn ? <Loader2 className="animate-spin" /> : <LogIn />}
              {isPassword ? copy.signIn : copy.signInWith(providerLabel)}
            </Button>
          )}
        </Field>
      ) : null}
      {!urlOnly && setup.authResolved && credentials.authMode === 'token' ? (
        <Field description={copy.tokenDesc} stacked={firstRun} title={copy.tokenTitle}>
          <Input
            aria-label={copy.tokenTitle}
            autoComplete="off"
            disabled={disabled}
            onChange={event => setup.setToken(event.target.value)}
            placeholder={
              credentials.tokenSet ? g.existingToken(credentials.tokenPreview ?? g.savedToken) : copy.pasteSessionToken
            }
            type="password"
            value={credentials.token}
          />
        </Field>
      ) : null}
      {setup.error ? <div className="text-sm text-destructive">{setup.error}</div> : null}
      {setup.success ? <div className="text-sm text-primary">{setup.success}</div> : null}
    </div>
  )
}
