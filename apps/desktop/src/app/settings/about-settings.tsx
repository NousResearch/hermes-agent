import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { AlertTriangle, Cpu, Loader2, RefreshCw, Sparkles } from '@/lib/icons'
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
import { $desktopVersion, refreshDesktopVersion } from '@/store/updates'

import { SectionHeading, SettingsContent } from './primitives'
import { UninstallSection } from './uninstall-section'

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

// The full About block (brand header, app updates, engine updates, uninstall)
// WITHOUT the SettingsContent scroll wrapper, so 个性化 (PersonalizationSettings)
// can embed it below the 人格 picker. AboutSettings keeps wrapping it for the
// still-functional `?tab=about` deep link.
export function AboutSettingsBody() {
  const { t } = useI18n()
  const a = t.settings.about
  const version = useStore($desktopVersion)

  // The version atom is loaded once at app boot, which makes About show a
  // stale number after a self-update (the running binary is current, the
  // displayed string is not). Re-read on mount so opening About always
  // reflects the running build.
  useEffect(() => {
    void refreshDesktopVersion()
  }, [])

  // Upstream's app self-update block (git-checkout based `hermes update`) is
  // intentionally NOT rendered: ApexNodes installs are COS source tarballs
  // with no .git, so that flow can only ever error. Shell updates ship as new
  // installers; the ENGINE update below (R5/R6) is our supported path.
  return (
    <>
      {/* hc: brand mark + "APEX 桌面版" heading removed per Kael — 个性化页不再顶品牌大头;
          版本号(功能信息)保留,居中小字。 */}
      <div className="pt-4 pb-2 text-center">
        <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {version?.appVersion ? a.version(version.appVersion) : a.versionUnavailable}
        </p>
      </div>

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <EngineUpdateSection />

        <UninstallSection />
      </div>
    </>
  )
}

export function AboutSettings() {
  return (
    <SettingsContent>
      <AboutSettingsBody />
    </SettingsContent>
  )
}
