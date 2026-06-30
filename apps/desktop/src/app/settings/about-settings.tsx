import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle, CheckCircle2, Cpu, ExternalLink, Loader2, RefreshCw, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $runtimeUpdateApplying,
  $runtimeUpdateCheck,
  $runtimeUpdateChecking,
  $runtimeVersion,
  applyRuntimeUpdate,
  checkRuntimeUpdate,
  loadRuntimeVersion
} from '@/store/runtime-update'
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

// R5 / R6 — desktop opt-in engine (runtime) update. Shows the currently
// installed engine version and, on demand (opt-in: the user must click), checks
// the admin-set default and offers a confirmed apply. Reuses the existing IPC
// bridge (window.hermesDesktop.runtime.*) via the runtime-update store — no
// mechanism lives here.
function EngineUpdateSection() {
  const { t } = useI18n()
  const a = t.settings.about
  const installed = useStore($runtimeVersion)
  const check = useStore($runtimeUpdateCheck)
  const checking = useStore($runtimeUpdateChecking)
  const applying = useStore($runtimeUpdateApplying)
  const [confirmOpen, setConfirmOpen] = useState(false)

  // R6: read the installed engine version on open. Local marker read only —
  // no network, no opt-in violation (the update *check* still waits for a click).
  useEffect(() => {
    void loadRuntimeVersion()
  }, [])

  // Prefer the version from a fresh opt-in check when present; otherwise the
  // locally-loaded marker version shown on open.
  const currentVersion = check?.current?.version ?? installed?.version ?? null
  const latest = check?.updateAvailable ? check.latest : null
  const compatNotes = latest?.compatibilityNotes?.trim() || ''
  // ok:false from a check means we couldn't reach the admin/latest endpoint.
  // updateAvailable:false with ok:true means we reached it and we're current.
  const reachable = check ? check.ok : true

  let statusLine: string
  let statusTone: 'available' | 'error' | 'idle' = 'idle'

  if (!check) {
    statusLine = a.engineTapCheck
  } else if (!reachable) {
    statusLine = a.engineCantReach
    statusTone = 'error'
  } else if (latest) {
    statusLine = latest.version ? a.engineFound(latest.version) : a.engineFoundGeneric
    statusTone = 'available'
  } else {
    statusLine = a.engineUpToDate
  }

  const handleApply = async () => {
    const result = await applyRuntimeUpdate()

    // On success the pin is re-armed; reload to drive the bootstrap re-run.
    if (result.reloadRequired) {
      window.location.reload()
    }
  }

  return (
    <div className="mt-6">
      <SectionHeading icon={Cpu} title={a.engineSection} />

      <div
        className={cn(
          'p5-panel px-4 py-3.5 text-sm',
          statusTone === 'error' ? 'text-destructive' : 'text-foreground'
        )}
        data-tone={statusTone}
      >
        <div className="flex items-start gap-2">
          {statusTone === 'available' ? (
            <Sparkles className="mt-0.5 size-4 shrink-0 text-[var(--ui-blue)]" />
          ) : statusTone === 'error' ? (
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          ) : (
            <Cpu className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          )}
          <div className="min-w-0">
            <p className="font-medium">{statusLine}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {currentVersion ? a.engineVersion(currentVersion) : a.engineVersionUnavailable}
            </p>
            {latest && compatNotes && (
              <p className="mt-2 whitespace-pre-line text-xs text-muted-foreground">
                <span className="font-medium text-foreground">{a.engineCompatNotes}: </span>
                {compatNotes}
              </p>
            )}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-4">
          <Button
            disabled={checking || applying}
            onClick={() => void checkRuntimeUpdate()}
            size="sm"
            variant="textStrong"
          >
            {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
            {checking ? a.engineChecking : a.engineCheck}
          </Button>

          {latest && (
            <Button disabled={applying} onClick={() => setConfirmOpen(true)} size="sm">
              {applying ? <Loader2 className="size-3 animate-spin" /> : null}
              {applying ? a.engineApplying : a.engineApply}
            </Button>
          )}
        </div>
      </div>

      <ConfirmDialog
        cancelLabel={t.common.cancel}
        confirmLabel={a.engineConfirmApply}
        description={latest?.version ? a.engineConfirmBody(latest.version) : a.engineConfirmBodyGeneric}
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleApply}
        open={confirmOpen}
        title={a.engineConfirmTitle}
      />
    </div>
  )
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
      <div className="flex flex-col items-center gap-3.5 pt-8 pb-3 text-center">
        <BrandMark className="size-16" />
        <div>
          <h2 className="text-[1.375rem] font-semibold tracking-tight text-foreground">{a.heading}</h2>
          <p className="mt-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {version?.appVersion ? a.version(version.appVersion) : a.versionUnavailable}
          </p>
        </div>
      </div>

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title={a.updates} />

        <div
          className={cn(
            'p5-panel px-4 py-3.5 text-sm',
            statusTone === 'error' ? 'text-destructive' : 'text-foreground'
          )}
          data-tone={statusTone}
        >
          <div className="flex items-start gap-2">
            {statusTone === 'available' ? (
              <Sparkles className="mt-0.5 size-4 shrink-0 text-[var(--ui-blue)]" />
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
              <Button onClick={() => openUpdatesWindow()} size="sm">
                {a.seeWhatsNew}
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

        <EngineUpdateSection />

        <UninstallSection />
      </div>
    </SettingsContent>
  )
}
