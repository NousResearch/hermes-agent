import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Check, Loader2 } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { ListRow, Pill, SettingsContent } from './primitives'

interface ServingHost {
  signedIn: boolean
  url: string
}

/**
 * Gateways in the Webapp. The page is served by one Hermes host and can only
 * talk to it, so the native connection controls (mode, Cloud, saved
 * connections, logs) have nothing to act on. What does apply is which host
 * this is and, behind the sign-in gate, signing out of it.
 */
export function WebappGatewaySettings({ embedded }: { embedded: boolean }) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const [host, setHost] = useState<null | ServingHost>(null)
  const [signingOut, setSigningOut] = useState(false)

  useEffect(() => {
    let cancelled = false

    void window.hermesDesktop
      .getConnectionConfig(null)
      .then(config => {
        if (!cancelled) {
          setHost({
            signedIn: config.remoteAuthMode === 'oauth' && config.remoteOauthConnected,
            url: config.remoteUrl
          })
        }
      })
      .catch(err => notifyError(err, g.failedLoad))

    return () => void (cancelled = true)
  }, [g.failedLoad])

  // Success leaves the page for the login screen, so only a failure re-arms.
  const signOut = async (url: string) => {
    setSigningOut(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(url)
    } catch (err) {
      notifyError(err, g.signOutFailed)
      setSigningOut(false)
    }
  }

  return (
    <SettingsContent bare={embedded}>
      {host ? (
        <ListRow
          action={
            host.signedIn ? (
              <div className="flex items-center gap-2">
                <Pill tone="primary">
                  <Check className="size-3" />
                  {g.signedIn}
                </Pill>
                <Button disabled={signingOut} onClick={() => void signOut(host.url)} variant="outline">
                  {signingOut ? <Loader2 className="animate-spin" /> : null}
                  {g.signOut}
                </Button>
              </div>
            ) : null
          }
          description={host.url}
          title={g.webappHostTitle}
        />
      ) : null}
      <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {g.webappHostDesc}
      </p>
    </SettingsContent>
  )
}
