import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { type Translations, useI18n } from '@/i18n'
import { CheckCircle2, ExternalLink, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateStatus,
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
  const checking = useStore($updateChecking)
  const [justChecked, setJustChecked] = useState(false)

  // The version atom is loaded once at app boot, which makes About show a
  // stale number after a self-update (the running binary is current, the
  // displayed string is not). Re-read on mount so opening About always
  // reflects the running build.
  useEffect(() => {
    void refreshDesktopVersion()
  }, [])

  const behind = status?.behind ?? 0
  const supported = status?.supported !== false
  const applying = apply.applying || apply.stage === 'restart'

  const handleCheck = async () => {
    setJustChecked(false)
    const next = await checkUpdates()
    setJustChecked(Boolean(next))
  }

  let statusLine: string
  let statusTone: 'idle' | 'available' | 'error' = 'idle'

  if (!supported) {
    statusLine = status?.message ?? a.cantUpdate
    statusTone = 'error'
  } else if (status?.error) {
    statusLine = a.cantReach
    statusTone = 'error'
  } else if (applying) {
    statusLine = a.installing
    statusTone = 'available'
  } else if (behind > 0) {
    statusLine = a.updateReady(behind)
    statusTone = 'available'
  } else if (status) {
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
              disabled={checking || applying || !supported}
              onClick={() => void handleCheck()}
              size="sm"
              variant="textStrong"
            >
              {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
              {checking ? a.checking : a.checkNow}
            </Button>

            {behind > 0 && supported && !applying && (
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
          hint={a.branchCommit(status?.branch ?? 'unknown', status?.currentSha?.slice(0, 7) ?? 'unknown')}
          title={a.automaticUpdates}
        />

        <UninstallSection />
      </div>
    </SettingsContent>
  )
}
