import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle, CheckCircle2, ExternalLink, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $automaticUpdateChecksEnabled,
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkUpdates,
  openUpdatesWindow,
  refreshDesktopVersion,
  setAutomaticUpdateChecksEnabled,
  startActiveUpdate
} from '@/store/updates'
import { UpdateStatusCard, VersionHero } from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connection } from '@/store/session'
import { $desktopVersion, checkBackendUpdates, refreshDesktopVersion } from '@/store/updates'

import { SectionHeading, SettingsContent, ToggleRow } from './primitives'
import { SectionHeading, SettingsContent } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { UninstallSection } from './uninstall-section'
import { useSettingDeepLink } from './use-setting-deep-link'

interface AboutSettingsProps {
  subpage?: string
}

export function AboutSettings({ subpage }: AboutSettingsProps = {}): ReactElement {
  useSettingDeepLink('about', page => subpage === undefined || page === subpage)

  if (subpage === 'uninstall') {
    return (
      <SettingsContent>
        <UninstallSection />
      </SettingsContent>
    )
  }

  return <AppUpdatesSettings includeUninstall={subpage === undefined} />
}

interface AppUpdatesSettingsProps {
  includeUninstall: boolean
}

function AppUpdatesSettings({ includeUninstall }: AppUpdatesSettingsProps): ReactElement {
  const { t } = useI18n()
  const version = useStore($desktopVersion)
  const status = useStore($updateStatus)
  const apply = useStore($updateApply)
  const checking = useStore($updateChecking)
  const automaticUpdateChecksEnabled = useStore($automaticUpdateChecksEnabled)
  const [justChecked, setJustChecked] = useState(false)
  const connection = useStore($connection)
  const remote = connection?.mode === 'remote'

  // Refresh the running version when About opens or the active gateway changes.
  useEffect((): void => {
    void refreshDesktopVersion()

  const behind = status?.behind ?? 0
  // behind is null when the exact count is unknowable (shallow clone): the
  // backend flags that case via updateAvailable instead of a number.
  const updateAvailable = behind > 0 || Boolean(status?.updateAvailable)
  const supported = status?.supported !== false
  const applying = apply.applying || apply.stage === 'restart'
  // A dirty working tree is safe to stash on the configured branch. The
  // updater is parked only when the checkout is on a different branch/ref,
  // which is the condition that requires an explicit operator choice.

  const updateParked =
    updateAvailable && Boolean(status?.currentBranch && status?.branch && status.currentBranch !== status.branch)

  const handleCheck = async () => {
    setJustChecked(false)
    const next = await checkUpdates({ force: true })
    setJustChecked(Boolean(next))
  }

  let statusLine: string
  let statusTone: 'idle' | 'available' | 'error' = 'idle'

  if (!supported) {
    statusLine = status?.message ?? a.cantUpdate
    statusTone = 'error'
  } else if (status?.error) {
    // A git that never ran is a local problem; leading with "couldn't reach
    // the update server" would misdiagnose it as a network failure.
    statusLine = [status.error === 'git-unusable' ? '' : a.cantReach, status.message].filter(Boolean).join(' ')
    statusTone = 'error'
  } else if (applying) {
    statusLine = a.installing
    statusTone = 'available'
  } else if (updateAvailable) {
    statusLine = behind > 0 ? a.updateReady(behind) : a.updateReadyUnknown
    statusTone = 'available'
  } else if (status) {
    statusLine = a.onLatest
  } else {
    statusLine = a.tapCheck
  }
    if (remote) {
      void checkBackendUpdates()
    }
  }, [connection, remote])

  return (
    <SettingsContent>
      <VersionHero version={version} />
      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title={a.updates} />

        <div
          className={cn(
            'rounded-xl border px-4 py-3 text-sm',
            statusTone === 'available' && 'border-primary/30 bg-primary/5 text-foreground',
            statusTone === 'error' && 'border-destructive/35 bg-destructive/5 text-destructive',
            statusTone === 'idle' && 'border-border/70 bg-muted/20 text-foreground'
          )}
        >
          <div className="flex items-start gap-2">
            {statusTone === 'available' ? (
              <Codicon className="mt-0.5 size-4 shrink-0 text-primary" name="cloud-download" size="1rem" />
            ) : statusTone === 'error' ? null : (
              <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
            )}
            <div className="min-w-0">
              <p className="font-medium">{statusLine}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {a.lastChecked(relativeTime(status?.fetchedAt, a))}
                {justChecked && !checking ? a.justNowSuffix : ''}
              </p>
              {updateParked && (
                <div className="mt-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-800 dark:text-amber-200">
                  <p className="font-medium">{a.updateParked}</p>
                  <p className="mt-0.5 text-amber-900/70 dark:text-amber-100/70">{a.updateParkedDesc}</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-4">
            <Button
              disabled={checking || applying || !supported}
              onClick={() => void handleCheck()}
              size="sm"
              variant="textStrong"
            >
              {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
              {checking ? a.checking : a.checkNow}
            </Button>

            {updateAvailable && supported && !applying && (
              <>
                <Button onClick={() => startActiveUpdate()} size="sm">
                  {a.updateNow}
                </Button>
                <Button onClick={() => openUpdatesWindow('client')} size="sm" variant="textStrong">
                  {a.seeWhatsNew}
                </Button>
              </>
            )}

            <Button asChild className="ml-auto" size="sm" variant="text">
              <a
                href={RELEASE_NOTES_URL}
                onClick={event => {
                  event.preventDefault()
                  void window.hermesDesktop?.openExternal?.(RELEASE_NOTES_URL)
                }}
                rel="noreferrer"
                target="_blank"
              >
                <ExternalLink className="size-3" />
                {a.releaseNotes}
              </a>
            </Button>
          </div>
        <SectionHeading icon={RefreshCw} title={t.settings.about.updates} />
        <div className="grid gap-3" id={settingElementId(SETTING_IDS.about.updates)}>
          <UpdateStatusCard target="client" />
          {/* Client and remote backend updates are independent. Only the client has release notes. */}
          {remote && <UpdateStatusCard showReleaseNotes={false} target="backend" />}
        </div>

        <ToggleRow
          checked={automaticUpdateChecksEnabled}
          description={a.automaticUpdatesDesc}
          hint={`${a.updateSource}: ${status?.repository ? `${status.repository} · ` : ''}${a.branchCommit(`origin/${status?.branch ?? 'unknown'}`, status?.currentSha?.slice(0, 7) ?? 'unknown')}`}
          id={settingElementId(SETTING_IDS.about.automaticUpdates)}
          label={a.automaticUpdates}
          onChange={setAutomaticUpdateChecksEnabled}
        />

        {version && <VersionDetails version={version} />}
        {includeUninstall && <UninstallSection />}
      </div>
    </SettingsContent>
  )
}
