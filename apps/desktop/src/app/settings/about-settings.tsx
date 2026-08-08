import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { type Translations, useI18n } from '@/i18n'
import { CheckCircle2, ExternalLink, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $connection } from '@/store/session'
import {
  $backendUpdateApply,
  $backendUpdateChecking,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkBackendUpdates,
  checkUpdates,
  openUpdatesWindow,
  refreshDesktopVersion,
  startActiveUpdate
} from '@/store/updates'

import { ListRow, SectionHeading, SettingsContent } from './primitives'
import { UninstallSection } from './uninstall-section'

const RELEASE_NOTES_URL = 'https://github.com/NousResearch/hermes-agent/releases'

function relativeTime(ms: number | undefined, a: Translations['settings']['about']) {
  if (!ms) {
    return a.never
  }

  const diff = Date.now() - ms

  if (diff < 60_000) {
    return a.justNow
  }

  if (diff < 3_600_000) {
    return a.minAgo(Math.round(diff / 60_000))
  }

  if (diff < 86_400_000) {
    return a.hoursAgo(Math.round(diff / 3_600_000))
  }

  return a.daysAgo(Math.round(diff / 86_400_000))
}

export function AboutSettings() {
  const { t } = useI18n()
  const a = t.settings.about
  const version = useStore($desktopVersion)
  const status = useStore($updateStatus)
  const apply = useStore($updateApply)
  const clientChecking = useStore($updateChecking)
  const connection = useStore($connection)
  const backendStatus = useStore($backendUpdateStatus)
  const backendApply = useStore($backendUpdateApply)
  const backendChecking = useStore($backendUpdateChecking)
  const [justChecked, setJustChecked] = useState(false)

  // Remote gateway: the desktop shell drives a remote backend, but this app is
  // still built from the LOCAL checkout. Both are updatable — show both, and
  // "Update now" (startActiveUpdate) chains backend-then-client.
  const isRemote = connection?.mode === 'remote'

  // The version atom is loaded once at app boot, which makes About show a
  // stale number after a self-update (the running binary is current, the
  // displayed string is not). Re-read on mount so opening About always
  // reflects the running build.
  useEffect(() => {
    void refreshDesktopVersion()
  }, [])

  const behind = status?.behind ?? 0
  const supported = status?.supported !== false
  const backendBehind = isRemote && backendStatus?.supported !== false && !backendStatus?.error ? (backendStatus?.behind ?? 0) : 0
  const checking = clientChecking || backendChecking
  const applying = apply.applying || apply.stage === 'restart' || backendApply.applying || backendApply.stage === 'restart'
  const updateReady = (behind > 0 && supported) || backendBehind > 0

  const handleCheck = async () => {
    setJustChecked(false)
    const [next] = await Promise.all([checkUpdates(), isRemote ? checkBackendUpdates() : Promise.resolve(null)])
    setJustChecked(Boolean(next))
  }

  let statusLine: string
  let statusTone: 'idle' | 'available' | 'error' = 'idle'

  if (!supported && backendBehind <= 0) {
    statusLine = status?.message ?? a.cantUpdate
    statusTone = 'error'
  } else if (status?.error && backendBehind <= 0) {
    statusLine = a.cantReach
    statusTone = 'error'
  } else if (applying) {
    statusLine = a.installing
    statusTone = 'available'
  } else if (behind > 0 && supported && backendBehind > 0) {
    statusLine = a.updateReadyBoth(backendBehind, behind)
    statusTone = 'available'
  } else if (backendBehind > 0) {
    statusLine = a.backendUpdateReady(backendBehind)
    statusTone = 'available'
  } else if (behind > 0) {
    statusLine = a.updateReady(behind)
    statusTone = 'available'
  } else if (status || (isRemote && backendStatus)) {
    statusLine = a.onLatest
  } else {
    statusLine = a.tapCheck
  }

  return (
    <SettingsContent>
      <div className="flex flex-col items-center gap-3 pt-6 pb-2 text-center">
        <BrandMark className="size-16" />
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{a.heading}</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {version?.appVersion ? a.version(version.appVersion) : a.versionUnavailable}
          </p>
        </div>
      </div>

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
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-4">
            <Button
              disabled={checking || applying || (!supported && !isRemote)}
              onClick={() => void handleCheck()}
              size="sm"
              variant="textStrong"
            >
              {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
              {checking ? a.checking : a.checkNow}
            </Button>

            {updateReady && !applying && (
              <>
                <Button onClick={() => startActiveUpdate()} size="sm">
                  {a.updateNow}
                </Button>
                <Button onClick={() => openUpdatesWindow()} size="sm" variant="textStrong">
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
        </div>

        <ListRow
          description={a.automaticUpdatesDesc}
          hint={
            a.branchCommit(status?.branch ?? 'unknown', status?.currentSha?.slice(0, 7) ?? 'unknown') +
            (isRemote && backendStatus?.currentVersion ? ` · ${a.backendVersion(backendStatus.currentVersion)}` : '')
          }
          title={a.automaticUpdates}
        />

        <UninstallSection />
      </div>
    </SettingsContent>
  )
}
