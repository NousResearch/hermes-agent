import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import {
  UpdateStatusCard,
  type UpdateStatusView,
  useUpdateStatusView,
  VersionBundleNotice,
  VersionHero
} from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import type { DesktopVersionInfo } from '@/global'
import { useI18n } from '@/i18n'
import { Loader2, RefreshCw } from '@/lib/icons'
import { shortVersion } from '@/lib/version-label'
import { $cloudOwnsUpdates } from '@/store/cloud-updates'
import { $connection } from '@/store/session'
import { usePortalShell } from '@/store/shell-variant'
import {
  $desktopVersion,
  checkBackendUpdates,
  openUpdateOverlayFor,
  refreshDesktopVersion,
  startActiveUpdate,
  type UpdateTarget
} from '@/store/updates'

import { ListRow, SectionHeading, SettingsContent } from './primitives'
import {
  SettingsCard,
  SettingsDisclosure,
  SettingsGroup,
  SettingsOverview,
  SettingsPageDescription
} from './settings-card'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { useSettingsPagesCopy } from './settings-pages-copy'
import { UninstallSection } from './uninstall-section'
import { UpdateChannelRow } from './update-channel-row'
import { useSettingDeepLink } from './use-setting-deep-link'

interface AboutSettingsProps {
  subpage?: string
}

export function AboutSettings({ subpage }: AboutSettingsProps = {}): ReactElement {
  // The redesign is one page (856:6778): uninstall follows the version card instead of its own subpage.
  const redesign = usePortalShell()
  useSettingDeepLink('about', page => subpage === undefined || page === subpage)

  if (subpage === 'uninstall') {
    return (
      <SettingsContent>
        <UninstallSection />
      </SettingsContent>
    )
  }

  return <AppUpdatesSettings includeUninstall={redesign || subpage === undefined} />
}

interface AppUpdatesSettingsProps {
  includeUninstall: boolean
}

function AppUpdatesSettings({ includeUninstall }: AppUpdatesSettingsProps): ReactElement {
  const { t } = useI18n()
  const redesign = usePortalShell()
  const version = useStore($desktopVersion)
  const connection = useStore($connection)
  const remote = connection?.mode === 'remote'
  // Hermes Cloud updates its agents itself: no checker, channel or Update button for what it owns.
  const cloudOwned = useStore($cloudOwnsUpdates)
  const showBackend = remote && !cloudOwned.backend

  // Refresh the running version when About opens or the active gateway changes.
  useEffect((): void => {
    void refreshDesktopVersion()

    if (showBackend) {
      void checkBackendUpdates()
    }
  }, [connection, showBackend])

  if (redesign) {
    return (
      <RedesignAbout
        clientOwned={cloudOwned.client}
        includeUninstall={includeUninstall}
        showBackend={showBackend}
        version={version}
      />
    )
  }

  return (
    <SettingsContent>
      <VersionHero version={version} />
      <div className="mx-auto mt-4 w-full max-w-2xl">
        {!cloudOwned.client && (
          <>
            <SectionHeading icon={RefreshCw} title={t.settings.about.updates} />
            <div className="grid gap-3" id={settingElementId(SETTING_IDS.about.updates)}>
              <UpdateStatusCard target="client" version={version} />
              <UpdateChannelRow />
              {/* Client and remote backend updates are independent. Only the client has release notes. */}
              {showBackend && <UpdateStatusCard showReleaseNotes={false} target="backend" version={version} />}
            </div>
          </>
        )}
        {version && <VersionDetails version={version} />}
        {includeUninstall && <UninstallSection />}
      </div>
    </SettingsContent>
  )
}

interface RedesignAboutProps {
  clientOwned: boolean
  includeUninstall: boolean
  showBackend: boolean
  version: DesktopVersionInfo | null
}

/** About & Updates (856:6778): one Version card (this app, the remote gateway, the update channel),
 *  build details under Advanced, then uninstall. Applying an update still runs in the updates overlay. */
function RedesignAbout({ clientOwned, includeUninstall, showBackend, version }: RedesignAboutProps): ReactElement {
  const { t } = useI18n()
  const pages = useSettingsPagesCopy()

  return (
    <SettingsContent>
      <SettingsPageDescription>{pages.intro.about}</SettingsPageDescription>
      <SettingsOverview>
        <VersionBundleNotice version={version} />
        <SettingsGroup id={settingElementId(SETTING_IDS.about.updates)} title={pages.version}>
          <SettingsCard>
            <AppVersionRow clientOwned={clientOwned} separateGateway={showBackend} version={version} />
            {showBackend && <GatewayVersionRow />}
            {!clientOwned && <UpdateChannelRow />}
          </SettingsCard>
        </SettingsGroup>
        {version && (
          <SettingsDisclosure count={1} description={pages.about.detailsDesc} title={t.settings.sections.advanced}>
            <VersionDetails version={version} />
          </SettingsDisclosure>
        )}
        {includeUninstall && <UninstallSection />}
      </SettingsOverview>
    </SettingsContent>
  )
}

interface UpdateRowActionProps {
  target: UpdateTarget
  /** Offer a re-check; the app row leaves it to the gateway row when a remote gateway has its own. */
  canCheck: boolean
}

/** Update now when one is waiting, otherwise Check for updates while the target supports checking. */
function UpdateRowAction({ canCheck, target }: UpdateRowActionProps): ReactElement | null {
  const pages = useSettingsPagesCopy()
  const { t } = useI18n()
  const { checking, handleCheck, view } = useUpdateStatusView(target)

  if (view.applying || view.tone === 'unsupported') {
    return null
  }

  if (view.updateAvailable) {
    return (
      <Button onClick={() => startActiveUpdate(target)} size="sm">
        {t.updates.updateNow}
      </Button>
    )
  }

  return canCheck ? (
    <Button disabled={checking} onClick={() => void handleCheck()} size="sm" variant="outline">
      {checking ? <Loader2 className="animate-spin" /> : null}
      {checking ? t.updates.checkingShort : pages.about.checkForUpdates}
    </Button>
  ) : null
}

/** "Version 1.2 · You're up to date · Last checked 5 min ago" — the status card folded into a row. */
function statusDescription(lead: string, view: UpdateStatusView, lastChecked: string): string {
  return [lead, view.line, view.tone === 'unsupported' ? null : lastChecked].filter(Boolean).join(' · ')
}

interface AppVersionRowProps {
  clientOwned: boolean
  separateGateway: boolean
  version: DesktopVersionInfo | null
}

function AppVersionRow({ clientOwned, separateGateway, version }: AppVersionRowProps): ReactElement {
  const { t } = useI18n()
  const u = t.updates
  const pages = useSettingsPagesCopy()
  const { lastChecked, view } = useUpdateStatusView('client')
  const lead = version?.appVersion ? u.version(shortVersion(version.appVersion)) : u.versionUnavailable

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          {!clientOwned && <UpdateRowAction canCheck={!separateGateway} target="client" />}
          <Button onClick={() => openUpdateOverlayFor('client')} size="sm" variant="outline">
            {pages.about.whatsNew}
          </Button>
        </div>
      }
      description={clientOwned ? lead : statusDescription(lead, view, lastChecked)}
      title={u.appName}
    />
  )
}

function GatewayVersionRow(): ReactElement {
  const pages = useSettingsPagesCopy()
  const { lastChecked, status, view } = useUpdateStatusView('backend')
  const lead = status?.sourceVersion ? `v${status.sourceVersion}` : ''

  return (
    <ListRow
      action={<UpdateRowAction canCheck target="backend" />}
      description={statusDescription(lead, view, lastChecked)}
      title={pages.about.gateway}
    />
  )
}
