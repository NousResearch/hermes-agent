import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect } from 'react'

import { UpdateStatusCard, VersionHero } from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connection } from '@/store/session'
import { $desktopVersion, checkBackendUpdates, refreshDesktopVersion } from '@/store/updates'

import { SectionHeading, SettingsContent } from './primitives'
import { UninstallSection } from './uninstall-section'

export function AboutSettings(): ReactElement {
  const { t } = useI18n()
  const version = useStore($desktopVersion)
  const connection = useStore($connection)
  const remote = connection?.mode === 'remote'

  // Refresh the running version when About opens or the active gateway changes.
  useEffect((): void => {
    void refreshDesktopVersion()

    if (remote) {
      void checkBackendUpdates()
    }
  }, [connection, remote])

  return (
    <SettingsContent>
      <VersionHero version={version} />

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title={t.settings.about.updates} />

        <div className="grid gap-3">
          <UpdateStatusCard target="client" />
          {/* Client and remote backend updates are independent. Only the client has release notes. */}
          {remote && <UpdateStatusCard showReleaseNotes={false} target="backend" />}
        </div>

        {version && <VersionDetails version={version} />}

        <UninstallSection />
      </div>
    </SettingsContent>
  )
}