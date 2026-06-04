import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Loader2, RefreshCw, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkUpdates,
  openUpdatesWindow,
  refreshDesktopVersion
} from '@/store/updates'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

const RELEASE_NOTES_URL = 'https://github.com/NousResearch/hermes-agent/releases'

function relativeTime(ms: number | undefined) {
  if (!ms) {
    return 'never'
  }

  const diff = Date.now() - ms

  if (diff < 60_000) {
    return 'just now'
  }

  if (diff < 3_600_000) {
    return `${Math.round(diff / 60_000)} min ago`
  }

  if (diff < 86_400_000) {
    return `${Math.round(diff / 3_600_000)} hours ago`
  }

  return `${Math.round(diff / 86_400_000)} days ago`
}

export function AboutSettings() {
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
    statusLine = status?.message ?? "This build can't update itself from inside the app."
    statusTone = 'error'
  } else if (status?.error) {
    statusLine = "We couldn't reach the update server."
    statusTone = 'error'
  } else if (applying) {
    statusLine = 'An update is currently installing.'
    statusTone = 'available'
  } else if (behind > 0) {
    statusLine = `A new update is ready (${behind} change${behind === 1 ? '' : 's'} included).`
    statusTone = 'available'
  } else if (status) {
    statusLine = "You're on the latest version."
  } else {
    statusLine = 'Tap "Check now" to look for updates.'
  }

  return (
    <SettingsContent>
      <div className="flex flex-col items-center gap-3 pt-6 pb-2 text-center">
        <span className="flex size-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Sparkles className="size-8" />
        </span>
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Hermes Desktop</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {version?.appVersion ? `Version ${version.appVersion}` : 'Version unavailable'}
          </p>
        </div>
      </div>

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title="Updates" />

        <div
          className={cn(
            'rounded-xl border px-4 py-3 text-sm',
            statusTone === 'available' && 'border-primary/30 bg-primary/5 text-foreground',
            statusTone === 'error' && 'border-destructive/35 bg-destructive/5 text-destructive',
            statusTone === 'idle' && 'border-border/70 bg-muted/20 text-foreground'
          )}
        >
          <div className="min-w-0">
            <p className="font-medium">{statusLine}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Last checked {relativeTime(status?.fetchedAt)}
              {justChecked && !checking ? ' · just now' : ''}
            </p>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-4">
            <Button
              disabled={checking || applying || !supported}
              onClick={() => void handleCheck()}
              size="sm"
              variant="textStrong"
            >
              {checking ? <Loader2 className="size-3 animate-spin" /> : null}
              {checking ? 'Checking…' : 'Check now'}
            </Button>

            {behind > 0 && supported && !applying && (
              <Button onClick={() => openUpdatesWindow()} size="sm">
                See what&apos;s new
              </Button>
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
                Release notes
              </a>
            </Button>
          </div>
        </div>

        <ListRow
          description="Hermes checks for updates automatically in the background and lets you know when one is ready."
          hint={`Branch ${status?.branch ?? 'unknown'} · Commit ${status?.currentSha?.slice(0, 7) ?? 'unknown'}`}
          title="Automatic updates"
        />
      </div>
    </SettingsContent>
  )
}
