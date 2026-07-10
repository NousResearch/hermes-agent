import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertCircle, Check, Loader2, LogIn } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { CONTROL_TEXT } from './constants'
import { ListRow, Pill } from './primitives'
import type { RemoteConnectionForm } from './use-remote-connection-form'

// Presentational remote-connection form: the URL entry row, the probe
// status/error rows, and the auth-resolved OAuth-sign-in vs session-token
// controls. Pure pass-through of the shared hook so Settings → Gateway and the
// first-run overlay render one identical form.
export function RemoteConnectForm({ className, form }: { className?: string; form: RemoteConnectionForm }) {
  const { t } = useI18n()
  const g = t.settings.gateway

  const {
    state,
    setState,
    remoteToken,
    setRemoteToken,
    signingIn,
    probeStatus,
    trimmedUrl,
    authMode,
    authResolved,
    providerLabel,
    isPasswordProvider,
    oauthConnected,
    signIn,
    signOut
  } = form

  return (
    <div className={cn('grid gap-1', className)}>
      <ListRow
        action={
          <Input
            className={cn('h-8', CONTROL_TEXT)}
            disabled={state.envOverride}
            onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
            placeholder="https://gateway.example.com/hermes"
            value={state.remoteUrl}
          />
        }
        description={g.remoteUrlDesc}
        title={g.remoteUrlTitle}
      />

      {probeStatus === 'probing' ? (
        <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <Loader2 className="size-4 animate-spin" />
          {g.probing}
        </div>
      ) : null}

      {probeStatus === 'error' ? (
        <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          {g.probeError}
        </div>
      ) : null}

      {/* OAuth / password gateways: present a sign-in button + connection status. */}
      {authResolved && authMode === 'oauth' ? (
        <ListRow
          action={
            oauthConnected ? (
              <div className="flex items-center gap-2">
                <Pill tone="primary">
                  <Check className="size-3" /> {g.signedIn}
                </Pill>
                <Button disabled={signingIn || state.envOverride} onClick={() => void signOut()} variant="outline">
                  {signingIn ? <Loader2 className="animate-spin" /> : null}
                  {g.signOut}
                </Button>
              </div>
            ) : (
              <Button disabled={signingIn || state.envOverride || !trimmedUrl} onClick={() => void signIn()}>
                {signingIn ? <Loader2 className="animate-spin" /> : <LogIn />}
                {isPasswordProvider ? g.signIn : g.signInWith(providerLabel)}
              </Button>
            )
          }
          description={
            oauthConnected
              ? isPasswordProvider
                ? g.authSignedInPassword
                : g.authSignedInOauth
              : isPasswordProvider
                ? g.authNeedsPassword
                : g.authNeedsOauth(providerLabel)
          }
          title={g.authTitle}
        />
      ) : null}

      {/* Session-token gateways: keep the existing token entry box. */}
      {authResolved && authMode === 'token' ? (
        <ListRow
          action={
            <Input
              autoComplete="off"
              className={cn('h-8 font-mono', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setRemoteToken(event.target.value)}
              placeholder={
                state.remoteTokenSet ? g.existingToken(state.remoteTokenPreview ?? g.savedToken) : g.pasteSessionToken
              }
              type="password"
              value={remoteToken}
            />
          }
          description={g.tokenDesc}
          title={g.tokenTitle}
        />
      ) : null}
    </div>
  )
}
