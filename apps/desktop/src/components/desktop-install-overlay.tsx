import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import type {
  DesktopBootstrapEvent,
  DesktopBootstrapStageDescriptor,
  DesktopBootstrapStageResult,
  DesktopBootstrapStageState,
  DesktopBootstrapState
} from '@/global'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, ChevronDown, ChevronRight, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

/**
 * DesktopInstallOverlay
 *
 * Renders the first-launch install progress for Hermes Agent. Mounted always;
 * shows itself only when main.cjs reports an in-flight bootstrap (state.active)
 * OR an error from a completed-failed bootstrap (state.error). When the
 * bootstrap finishes successfully the overlay fades out and the rest of the
 * app (existing onboarding overlay -> main UI) takes over.
 *
 * Subscribes to two channels:
 *   - getBootstrapState()           -- initial snapshot on mount
 *   - onBootstrapEvent(callback)    -- live event stream
 *
 * The reducer is intentionally simple: every event mutates an in-component
 * snapshot the same way main.cjs mutates its server-side snapshot. We don't
 * try to reconcile -- if we miss an event (shouldn't happen) the initial
 * getBootstrapState() call will resync the picture on the next render.
 *
 * Stages flagged needs_user_input render with a deliberately subdued style:
 * they're expected to come back as skipped=true (install.ps1 short-circuits
 * them under -NonInteractive). The post-install configuration flow that
 * those stages cover (API key, model, persona, gateway autostart) is handled
 * by the existing DesktopOnboardingOverlay, NOT by the install overlay.
 */

interface DesktopInstallOverlayProps {
  /** When false, the overlay never renders -- useful for dev when we want
   * to suppress it entirely. */
  enabled?: boolean
}

interface StageRowProps {
  descriptor: DesktopBootstrapStageDescriptor
  result: DesktopBootstrapStageResult | undefined
  isCurrent: boolean
  now: number
}

function formatStageName(name: string): string {
  // 'system-packages' -> 'System packages'; 'uv' stays 'uv'
  if (name.length <= 3) {
    return name
  }

  return name
    .split('-')
    .map((word, i) => (i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word))
    .join(' ')
}

function formatDuration(ms: number | null | undefined): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) {
    return ''
  }

  if (ms < 1000) {
    return `${ms} ms`
  }

  const s = ms / 1000

  if (s < 60) {
    return `${s.toFixed(1)}s`
  }

  const m = Math.floor(s / 60)
  const rs = Math.round(s - m * 60)

  return `${m}m ${rs}s`
}

// Live elapsed for a running stage, as m:ss (or s for sub-minute).
function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))

  if (s < 60) {
    return `${s}s`
  }

  const m = Math.floor(s / 60)

  return `${m}:${String(s - m * 60).padStart(2, '0')}`
}

function StageRow({ descriptor, result, isCurrent, now }: StageRowProps) {
  const { t } = useI18n()
  const copy = t.install
  const state: DesktopBootstrapStageState = result?.state || 'pending'

  const elapsed =
    state === 'running' && typeof result?.startedAt === 'number' ? formatElapsed(now - result.startedAt) : ''

  const icon = useMemo(() => {
    switch (state) {
      case 'running':
        return <Loader2 className="h-4 w-4 animate-spin text-primary" />

      case 'succeeded':
        return <Check className="h-4 w-4 text-emerald-600" />

      case 'skipped':
        return <Check className="h-4 w-4 text-muted-foreground" />

      case 'failed':
        return <AlertTriangle className="h-4 w-4 text-destructive" />

      case 'pending':

      default:
        return <div className="h-2 w-2 rounded-full border border-muted-foreground/40" />
    }
  }, [state])

  const reason = result?.json?.reason || result?.error || null

  return (
    <li
      className={cn(
        'flex items-start gap-3 rounded-md px-3 py-2 transition-colors',
        isCurrent && 'bg-muted/60',
        state === 'failed' && 'bg-destructive/10'
      )}
    >
      <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center">{icon}</div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className={cn('truncate text-sm font-medium', state === 'pending' && 'text-muted-foreground')}>
            {formatStageName(descriptor.name)}
          </span>
          <span className="flex-shrink-0 text-xs tabular-nums text-muted-foreground">
            {state === 'running'
              ? elapsed
                ? `${copy.stageStates[state]} · ${elapsed}`
                : copy.stageStates[state]
              : null}
            {state === 'succeeded' || state === 'skipped' ? formatDuration(result?.durationMs) : null}
            {state === 'failed' ? copy.stageStates[state] : null}
          </span>
        </div>
        {reason && state !== 'pending' && <p className="mt-0.5 truncate text-xs text-muted-foreground">{reason}</p>}
      </div>
    </li>
  )
}

const EMPTY_STATE: DesktopBootstrapState = {
  active: false,
  manifest: null,
  stages: {},
  error: null,
  log: [],
  startedAt: null,
  completedAt: null,
  unsupportedPlatform: null
}

function applyEvent(state: DesktopBootstrapState, ev: DesktopBootstrapEvent): DesktopBootstrapState {
  if (ev.type === 'manifest') {
    const stages: Record<string, DesktopBootstrapStageResult> = {}

    for (const stage of ev.stages) {
      stages[stage.name] = { state: 'pending', durationMs: null, startedAt: null, json: null, error: null }
    }

    return {
      ...state,
      active: true,
      manifest: { type: 'manifest', stages: ev.stages, protocolVersion: ev.protocolVersion },
      stages,
      error: null,
      startedAt: state.startedAt || Date.now()
    }
  }

  if (ev.type === 'stage') {
    const prev = state.stages[ev.name]

    return {
      ...state,
      stages: {
        ...state.stages,
        [ev.name]: {
          state: ev.state,
          durationMs: ev.durationMs ?? null,
          // Stamp the start time on the running transition so the UI can show
          // a live elapsed timer; preserve it across repeated running events.
          startedAt: ev.state === 'running' ? (prev?.startedAt ?? Date.now()) : (prev?.startedAt ?? null),
          json: ev.json ?? null,
          error: ev.error ?? null
        }
      }
    }
  }

  if (ev.type === 'log') {
    const next = state.log.concat({ ts: Date.now(), stage: ev.stage ?? null, line: ev.line, stream: ev.stream })

    while (next.length > 500) {
      next.shift()
    }

    return { ...state, log: next }
  }

  if (ev.type === 'complete') {
    return { ...state, active: false, completedAt: Date.now(), error: null }
  }

  if (ev.type === 'failed') {
    return { ...state, active: false, error: ev.error || 'unknown error' }
  }

  if (ev.type === 'unsupported-platform') {
    return {
      ...state,
      active: false,
      unsupportedPlatform: {
        platform: ev.platform,
        activeRoot: ev.activeRoot,
        installCommand: ev.installCommand,
        docsUrl: ev.docsUrl
      }
    }
  }

  return state
}

export function DesktopInstallOverlay({ enabled = true }: DesktopInstallOverlayProps) {
  const { t } = useI18n()
  const copy = t.install
  const [state, setState] = useState<DesktopBootstrapState>(EMPTY_STATE)
  const [logOpen, setLogOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const logEndRef = useRef<HTMLDivElement | null>(null)

  // Tick once a second while a bootstrap is in flight so running steps show a
  // live elapsed timer. Stops when nothing is active to avoid idle renders.
  useEffect(() => {
    if (!state.active) {
      return
    }

    const id = window.setInterval(() => setNow(Date.now()), 1000)

    return () => window.clearInterval(id)
  }, [state.active])

  // Subscribe to bootstrap events + load initial snapshot
  useEffect(() => {
    if (!enabled) {
      return
    }

    const desktop = window.hermesDesktop

    if (!desktop || typeof desktop.onBootstrapEvent !== 'function') {
      return
    }

    let cancelled = false

    desktop
      .getBootstrapState()
      .then(snapshot => {
        if (!cancelled && snapshot) {
          setState(snapshot)
        }
      })
      .catch(() => {
        // Older Electron build without the IPC handler -- bootstrap UI just
        // stays empty, app falls through to existing onboarding flow.
      })

    const off = desktop.onBootstrapEvent(ev => setState(prev => applyEvent(prev, ev)))

    return () => {
      cancelled = true
      off?.()
    }
  }, [enabled])

  // Autoscroll log to bottom when new lines arrive AND the log is open
  useEffect(() => {
    if (logOpen && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'auto', block: 'end' })
    }
  }, [state.log.length, logOpen])

  // Auto-expand the log panel when a bootstrap fails so the user immediately
  // sees the install.ps1 output. Without this, the failure block shows just
  // the top-level error message and the user has to click "Show installer
  // output" to see WHY the stage failed.
  useEffect(() => {
    if (state.error) {
      setLogOpen(true)
    }
  }, [state.error])

  // Mount logic: show whenever a bootstrap is in flight, completed-with-error,
  // or actively running with a manifest. Hide entirely after a successful
  // completion so the rest of the UI can take over.
  const shouldShow = useMemo(() => {
    if (!enabled) {
      return false
    }

    if (state.active) {
      return true
    }

    if (state.error) {
      return true
    }

    if (state.unsupportedPlatform) {
      return true
    }

    return false
  }, [enabled, state.active, state.error, state.unsupportedPlatform])

  if (!shouldShow) {
    return null
  }

  // Unsupported-platform branch: macOS/Linux packaged builds hit this when
  // there's no Hermes Agent installed yet and we can't drive install.sh
  // (no stage protocol equivalent yet). Show a copy-paste install command
  // and the docs URL; user runs it from Terminal and relaunches the app.
  if (state.unsupportedPlatform) {
    const ups = state.unsupportedPlatform
    const platformLabel = ups.platform === 'darwin' ? 'macOS' : ups.platform === 'linux' ? 'Linux' : ups.platform

    return (
      <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-background/90 backdrop-blur-md">
        <div className="w-full max-w-xl rounded-xl border bg-card p-8 shadow-xl">
          <h2 className="text-2xl font-semibold tracking-tight">{copy.oneTimeTitle}</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            {copy.unsupportedDesc(platformLabel)}
          </p>

          <div className="mt-4">
            <div className="mb-1.5 text-xs font-medium text-muted-foreground">{copy.installCommand}</div>
            <pre className="overflow-x-auto rounded-md border bg-muted/50 px-3 py-2.5 font-mono text-[12px]">
              <code>{ups.installCommand}</code>
            </pre>
            <div className="mt-2 flex items-center gap-2">
              <Button
                onClick={() => {
                  void navigator.clipboard?.writeText(ups.installCommand).catch(() => {})
                }}
                size="sm"
                variant="secondary"
              >
                {copy.copyCommand}
              </Button>
              <Button
                onClick={() => {
                  window.hermesDesktop?.openExternal?.(ups.docsUrl)
                }}
                size="sm"
                variant="ghost"
              >
                {copy.viewDocs}
              </Button>
            </div>
          </div>

          <div className="mt-6 flex items-center justify-between border-t pt-4">
            <span className="text-xs text-muted-foreground">
              {copy.installTo} <code className="rounded bg-muted/50 px-1 py-0.5 font-mono">{ups.activeRoot}</code>
            </span>
            <Button onClick={() => window.location.reload()} size="sm" variant="default">
              {copy.retryAfterRun}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  const stages = state.manifest?.stages || []
  const currentStage = stages.find(s => state.stages[s.name]?.state === 'running')?.name

  const completedCount = stages.filter(
    s => state.stages[s.name]?.state === 'succeeded' || state.stages[s.name]?.state === 'skipped'
  ).length

  const totalCount = stages.length
  const failed = Boolean(state.error)
  const progressPct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0
  const currentStartedAt = currentStage ? state.stages[currentStage]?.startedAt : null
  const currentElapsed = typeof currentStartedAt === 'number' ? formatElapsed(now - currentStartedAt) : ''

  return (
    <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-background/90 backdrop-blur-md p-4">
      <div className="flex w-full max-w-2xl max-h-[90vh] flex-col rounded-xl border bg-card shadow-xl">
        {/* Header -- always visible, never scrolls */}
        <div className="flex-shrink-0 p-8 pb-4">
          <h2 className="text-2xl font-semibold tracking-tight">
            {failed ? copy.failedTitle : state.active ? copy.settingUpTitle : copy.finishingTitle}
          </h2>
          <p className="mt-1.5 text-sm text-muted-foreground">
            {failed ? copy.failedDesc : copy.activeDesc}
          </p>
        </div>

        {/* Scrollable middle: progress, stages, error block, log */}
        <div className="min-h-0 flex-1 overflow-y-auto px-8 pb-2">
          {totalCount > 0 && (
            <div className="mb-4">
              <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  {copy.progress(completedCount, totalCount)}
                  {currentStage && copy.currentStage(formatStageName(currentStage))}
                  {currentElapsed && ` (${currentElapsed})`}
                </span>
                <span className="tabular-nums">{progressPct}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className={cn('h-full transition-all duration-300', failed ? 'bg-destructive' : 'bg-primary')}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          )}

          {totalCount === 0 && state.active && (
            <div className="mb-4 flex items-center gap-2 rounded-md border border-dashed bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{copy.fetchingManifest}</span>
            </div>
          )}

          {failed && state.error && (
            <div className="mb-4 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm">
              <div className="mb-1 flex items-center gap-1.5 font-medium text-destructive">
                <AlertTriangle className="h-4 w-4" />
                <span>{copy.error}</span>
              </div>
              <p className="whitespace-pre-wrap break-words text-foreground/90">{state.error}</p>
            </div>
          )}

          {stages.length > 0 && (
            <ol className="mb-4 space-y-1">
              {stages.map(stage => (
                <StageRow
                  descriptor={stage}
                  isCurrent={stage.name === currentStage}
                  key={stage.name}
                  now={now}
                  result={state.stages[stage.name]}
                />
              ))}
            </ol>
          )}

          <div className="border-t pt-3">
            <button
              className="flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
              onClick={() => setLogOpen(v => !v)}
              type="button"
            >
              {logOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              <span>{logOpen ? copy.hideOutput : copy.showOutput}</span>
              <span className="ml-1 tabular-nums">
                ({copy.lines(state.log.length)})
              </span>
            </button>

            {logOpen && (
              <div
                className={cn(
                  'mt-2 overflow-auto rounded-md border bg-muted/30 p-2 font-mono text-[11px] leading-relaxed',
                  failed ? 'max-h-96' : 'max-h-64'
                )}
              >
                {state.log.length === 0 ? (
                  <div className="text-muted-foreground">{copy.noOutput}</div>
                ) : (
                  <>
                    {state.log.map((entry, i) => (
                      <div
                        className={cn(
                          'whitespace-pre-wrap break-words',
                          entry.stream === 'stderr' && 'text-muted-foreground'
                        )}
                        key={i}
                      >
                        {entry.stage ? <span className="text-muted-foreground/70">[{entry.stage}] </span> : null}
                        <span>{entry.line}</span>
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Active footer: let the user actually cancel a running install. */}
        {state.active && !failed && (
          <div className="flex-shrink-0 border-t bg-card p-4">
            <div className="flex items-center justify-end">
              <Button
                disabled={cancelling}
                onClick={async () => {
                  setCancelling(true)

                  try {
                    await window.hermesDesktop?.cancelBootstrap?.()
                  } catch {
                    // ignore -- the failed/cancelled event will surface the result
                  }
                }}
                size="sm"
                variant="ghost"
              >
                {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                {cancelling ? copy.cancelling : copy.cancelInstall}
              </Button>
            </div>
          </div>
        )}

        {/* Footer -- always visible, never scrolls; only renders on failure */}
        {failed && (
          <div className="flex-shrink-0 border-t bg-card p-4">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">
                {copy.transcriptSaved}{' '}
                <code className="rounded bg-muted/50 px-1 py-0.5 font-mono">%LOCALAPPDATA%\hermes\logs\</code>
              </span>
              <div className="flex gap-2">
                <Button
                  onClick={async () => {
                    const text = state.log
                      .map(entry => (entry.stage ? `[${entry.stage}] ${entry.line}` : entry.line))
                      .join('\n')

                    const fullText = state.error ? `Error: ${state.error}\n\n${text}` : text

                    try {
                      await navigator.clipboard.writeText(fullText)
                      setCopied(true)
                      window.setTimeout(() => setCopied(false), 1500)
                    } catch {
                      // ignore -- some environments forbid clipboard writes
                    }
                  }}
                  size="sm"
                  variant="secondary"
                >
                  {copied ? copy.copiedOutput : copy.copyOutput}
                </Button>
                <Button
                  onClick={async () => {
                    // Tell main.cjs to clear its latched failure BEFORE we
                    // reload. Otherwise the renderer reload calls getConnection
                    // and main short-circuits to the latched error without
                    // re-running install.ps1.
                    try {
                      await window.hermesDesktop?.resetBootstrap?.()
                    } catch {
                      // best-effort -- continue with reload regardless
                    }

                    window.location.reload()
                  }}
                  size="sm"
                  variant="default"
                >
                  {copy.reloadRetry}
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
