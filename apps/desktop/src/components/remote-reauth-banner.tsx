import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $remoteReauthExhausted, clearRemoteReauthExhausted, remoteReauthOrigin } from '@/store/remote-reauth'
import { $connection } from '@/store/session'

/**
 * "Reconnection required" for the ACTIVE gateway.
 *
 * The main process stopped opening automatic sign-in windows for this origin
 * after its bounded retries exhausted (a dead credential must not be
 * hammered). This banner is the manual way back: one visible, user-requested
 * sign-in. It is deliberately non-modal and never steals focus — the
 * exhausted-recovery state offers; it does not hijack.
 */
export function RemoteReauthBanner() {
  const { t } = useI18n()
  const exhausted = useStore($remoteReauthExhausted)
  const connection = useStore($connection)
  const [signingIn, setSigningIn] = useState(false)

  const activeOrigin = connection ? remoteReauthOrigin(connection.baseUrl) : null
  const latched = activeOrigin ? exhausted[activeOrigin] : undefined

  if (!connection || !activeOrigin || !latched) {
    return null
  }

  let host = activeOrigin

  try {
    host = new URL(connection.baseUrl).host
  } catch {
    // the origin is already a readable fallback
  }

  const signIn = () => {
    setSigningIn(true)

    void window.hermesDesktop
      .oauthLoginConnectionConfig(connection.baseUrl)
      .then(result => {
        if (result?.ok) {
          clearRemoteReauthExhausted(connection.baseUrl)
        } else {
          notifyError(new Error(result?.error || 'Sign-in did not complete'), t.remoteReauthBanner.failed)
        }
      })
      .catch(error => notifyError(error, t.remoteReauthBanner.failed))
      .finally(() => setSigningIn(false))
  }

  return (
    <div
      className="fixed inset-x-0 top-3 z-(--z-over-modal) mx-auto flex w-fit max-w-lg items-center gap-4 rounded-lg border bg-background px-4 py-3 shadow-lg"
      role="alert"
    >
      <div className="grid gap-0.5">
        <div className="text-sm font-medium text-foreground">{t.remoteReauthBanner.title}</div>
        <div className="text-xs text-muted-foreground">{t.remoteReauthBanner.body(host)}</div>
      </div>
      <Button disabled={signingIn} onClick={signIn} size="sm" type="button" variant="secondary">
        {signingIn ? t.remoteReauthBanner.signingIn : t.remoteReauthBanner.action}
      </Button>
    </div>
  )
}
