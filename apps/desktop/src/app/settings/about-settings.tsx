import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { UpdateStatusCard, VersionHero } from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connection } from '@/store/session'
import { $desktopVersion, checkBackendUpdates, refreshDesktopVersion } from '@/store/updates'

import { SectionHeading, SettingsContent } from './primitives'
import { UninstallSection } from './uninstall-section'

export function AboutSettings() {
  const { t } = useI18n()
  const u = t.updates
  const version = useStore($desktopVersion)
  const connection = useStore($connection)
  const remote = connection?.mode === 'remote'

  // The version atom is loaded once at app boot, which makes About show a
  // stale number after a self-update (the running binary is current, the
  // displayed string is not). Re-read on mount so opening About always
  // reflects the running build. In remote mode also seed the backend update
  // state so the backend card opens answered instead of on "never checked".
  useEffect(() => {
    void refreshDesktopVersion()

    if (remote) {
      void checkBackendUpdates()
    }
  }, [remote])

  return (
    <SettingsContent>
      <VersionHero version={version} />

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title={t.settings.about.updates} />

        <div className="grid gap-3">
          <UpdateStatusCard target="client" />
          {/* The desktop client and a remote backend update independently — in
              remote mode the statusbar shows both pills, so About shows both
              states too. The backend has no GitHub release notes link of its
              own; the client card already carries it. */}
          {remote && <UpdateStatusCard showReleaseNotes={false} target="backend" />}
        </div>

        {version && <VersionDetails version={version} />}

        <UninstallSection />
      </div>
    </SettingsContent>
  )
}
