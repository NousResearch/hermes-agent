import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { DesktopUpdateStatus, DesktopVersionInfo } from '@/global'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle, CheckCircle2, ExternalLink, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $backendUpdateApply,
  $backendUpdateChecking,
  $backendUpdateStatus,
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkBackendUpdates,
  checkUpdates,
  openUpdateOverlayFor,
  startActiveUpdate,
  type UpdateApplyState,
  type UpdateTarget
} from '@/store/updates'

const RELEASE_NOTES_URL = 'https://github.com/NousResearch/hermes-agent/releases'
const INSTALLER_URL = 'https://hermes-agent.nousresearch.com/'

export type UpdateStatusTone = 'idle' | 'available' | 'error' | 'unsupported'

export interface UpdateStatusView {
  line: string
  tone: UpdateStatusTone
  error?: string
  updateAvailable: boolean
  applying: boolean
  supported: boolean
}

/**
 * One status derivation for every "am I up to date?" surface (About page,
 * updates overlay). Pure so the tone/copy contract is unit-testable.
 */
export function deriveUpdateStatus({
  apply,
  checking,
  status,
  target,
  u
}: {
  apply: UpdateApplyState
  checking: boolean
  status: DesktopUpdateStatus | null
  target: UpdateTarget
  u: Translations['updates']
}): UpdateStatusView {
  const behind = status?.behind ?? 0
  // behind is null when the exact count is unknowable (shallow clone): the
  // backend flags that case via updateAvailable instead of a number.
  const updateAvailable = behind > 0 || Boolean(status?.updateAvailable)
  const supported = status?.supported !== false
  const applying = apply.applying || apply.stage === 'restart'

  if (!supported) {
    return { applying, line: status?.message ?? u.unsupportedMessage, supported, tone: 'unsupported', updateAvailable }
  }

  if (status?.error) {
    return {
      applying,
      error: status.message,
      line: u.cantReach,
      supported,
      tone: 'error',
      updateAvailable
    }
  }

  if (applying) {
    return { applying, line: u.installing, supported, tone: 'available', updateAvailable }
  }

  if (updateAvailable) {
    return {
      applying,
      line: behind > 0 ? u.updateReady(behind) : u.updateReadyUnknown,
      supported,
      tone: 'available',
      updateAvailable
    }
  }

  if (status) {
    return {
      applying,
      line: target === 'backend' ? u.latestBodyBackend : u.latestBody,
      supported,
      tone: 'idle',
      updateAvailable
    }
  }

  return { applying, line: checking ? u.checking : u.tapCheck, supported, tone: 'idle', updateAvailable }
}

function relativeTime(ms: number | undefined, u: Translations['updates']): string {
  if (!ms) {
    return u.never
  }

  const diff = Date.now() - ms

  if (diff < 60_000) {
    return u.justNow
  }

  if (diff < 3_600_000) {
    return u.minAgo(Math.round(diff / 60_000))
  }

  if (diff < 86_400_000) {
    return u.hoursAgo(Math.round(diff / 3_600_000))
  }

  return u.daysAgo(Math.round(diff / 86_400_000))
}

/**
 * The "Hermes Desktop / version / brand mark" hero shared by the About page
 * and the updates overlay, including the bundle-out-of-sync warning. The
 * heading render is injectable so a dialog surface can emit a DialogTitle
 * for its accessible name while About keeps a plain h2.
 */
export function VersionHero({
  renderHeading,
  version
}: {
  renderHeading?: (heading: string) => ReactNode
  version: DesktopVersionInfo | null
}) {
  const { t } = useI18n()
  const u = t.updates

  return (
    <div className="flex flex-col items-center gap-3 pt-6 pb-2 text-center">
      <BrandMark className="size-16" />
      <div>
        {renderHeading ? (
          renderHeading(u.appName)
        ) : (
          <h2 className="text-lg font-semibold tracking-tight">{u.appName}</h2>
        )}
        <p className="mt-1 text-xs text-muted-foreground">
          {version?.appVersion ? u.version(version.appVersion) : u.versionUnavailable}
        </p>
      </div>
      {version?.bundleOutOfSync && (
        <div className="mx-auto w-full max-w-2xl rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-left text-sm">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
            <div className="min-w-0">
              <p className="font-medium">{u.bundleOutOfSync}</p>
              <p className="mt-1 text-xs text-muted-foreground">{u.bundleOutOfSyncDesc}</p>
              <Button asChild className="mt-2" size="sm" variant="textStrong">
                <a
                  href={INSTALLER_URL}
                  onClick={event => {
                    event.preventDefault()
                    void window.hermesDesktop?.openExternal?.(INSTALLER_URL)
                  }}
                  rel="noreferrer"
                  target="_blank"
                >
                  <ExternalLink className="size-3" />
                  {u.bundleOutOfSyncAction}
                </a>
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * The bordered update-state card (status line, last-checked age, check /
 * update / release-notes actions) in the About page's visual language.
 * Target-aware: 'client' reads the desktop self-update atoms, 'backend' the
 * remote backend's. The actual apply flow stays in the updates overlay —
 * "Update now" opens it and starts the install there.
 */
export function UpdateStatusCard({
  showReleaseNotes = true,
  target
}: {
  showReleaseNotes?: boolean
  target: UpdateTarget
}) {
  const { t } = useI18n()
  const u = t.updates
  const isBackend = target === 'backend'
  const status = useStore(isBackend ? $backendUpdateStatus : $updateStatus)
  const checking = useStore(isBackend ? $backendUpdateChecking : $updateChecking)
  const apply = useStore(isBackend ? $backendUpdateApply : $updateApply)
  const [justChecked, setJustChecked] = useState(false)

  const view = deriveUpdateStatus({ apply, checking, status, target, u })

  const handleCheck = async (): Promise<void> => {
    setJustChecked(false)
    const next = await (isBackend ? checkBackendUpdates() : checkUpdates())
    setJustChecked(Boolean(next))
  }

  return (
    <div
      className={cn(
        'rounded-xl border px-4 py-3 text-sm',
        view.tone === 'available' && 'border-primary/30 bg-primary/5 text-foreground',
        view.tone === 'error' && 'border-destructive/35 bg-destructive/5 text-destructive',
        (view.tone === 'idle' || view.tone === 'unsupported') && 'border-border/70 bg-muted/20 text-foreground'
      )}
    >
      <div className="flex items-start gap-2">
        {view.tone === 'available' ? (
          <Codicon className="mt-0.5 size-4 shrink-0 text-primary" name="cloud-download" size="1rem" />
        ) : view.tone === 'error' || view.tone === 'unsupported' ? null : (
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
        )}
        <div className="min-w-0">
          <p className="font-medium">{view.line}</p>
          {view.error && <p className="mt-1 text-xs text-muted-foreground">{view.error}</p>}
          {view.tone !== 'unsupported' && (
            <p className="mt-1 text-xs text-muted-foreground">
              {u.lastChecked(relativeTime(status?.fetchedAt, u))}
              {justChecked && !checking ? u.justNowSuffix : ''}
            </p>
          )}
        </div>
      </div>

      {view.tone !== 'unsupported' && (
        <div className="mt-3 flex flex-wrap items-center gap-4">
          <Button
            disabled={checking || view.applying}
            onClick={() => void handleCheck()}
            size="sm"
            variant="textStrong"
          >
            {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
            {checking ? u.checkingShort : u.checkNow}
          </Button>

          {view.updateAvailable && view.supported && !view.applying && (
            <>
              <Button onClick={() => startActiveUpdate()} size="sm">
                {u.updateNow}
              </Button>
              <Button onClick={() => openUpdateOverlayFor(target)} size="sm" variant="textStrong">
                {u.seeWhatsNew}
              </Button>
            </>
          )}

          {showReleaseNotes && (
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
                {u.releaseNotes}
              </a>
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
