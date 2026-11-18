import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Check, Loader2 } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import { usePortalShell } from '@/store/shell-variant'

import { ConnectionsRegistrySection } from './connections-registry'
import { ListRow, Pill, SettingsContent } from './primitives'
import { SettingsCard, SettingsGroup, SettingsOverview, SettingsPageDescription } from './settings-card'
import { useSettingsPagesCopy } from './settings-pages-copy'

interface ServingHost {
  signedIn: boolean
  url: string
}

/** How often the open page re-reads the saved connections, which native Desktop may change too. */
const SAVED_CONNECTIONS_POLL_MS = 5_000

/**
 * Gateways in the Webapp: which host serves the page and, behind the sign-in gate, signing out of
 * it, then native Desktop's own Saved connections page over the one list this machine keeps. The
 * native connection-mode controls (local runtime, Cloud login, logs) stay with the native app.
 */
export function WebappGatewaySettings({ embedded }: { embedded: boolean }) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const pages = useSettingsPagesCopy()
  const redesign = usePortalShell()
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

  // An edit in native Desktop or another tab shows here while the page is open (the read publishes
  // what changed, which the section and the gateway switcher follow).
  useEffect(() => {
    const list = window.hermesDesktop?.connections?.list

    if (!list) {
      return
    }

    const timer = window.setInterval(() => void list().catch(() => undefined), SAVED_CONNECTIONS_POLL_MS)

    return () => window.clearInterval(timer)
  }, [])

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

  const hostRow = host ? (
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
  ) : null

  if (redesign) {
    return (
      <SettingsContent bare={embedded}>
        <SettingsPageDescription>{pages.intro.gateways}</SettingsPageDescription>
        <SettingsOverview>
          <SettingsGroup description={g.webappHostDesc} title={pages.gateway.thisHost}>
            {hostRow && <SettingsCard>{hostRow}</SettingsCard>}
          </SettingsGroup>
          <ConnectionsRegistrySection />
        </SettingsOverview>
      </SettingsContent>
    )
  }

  return (
    <SettingsContent bare={embedded}>
      {hostRow}
      <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {g.webappHostDesc}
      </p>
      <ConnectionsRegistrySection />
    </SettingsContent>
  )
}
