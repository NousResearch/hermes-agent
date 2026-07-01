import { useEffect, useMemo, useRef, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorIcon } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import { LogView } from '@/components/ui/log-view'
import type {
  DesktopBootstrapEvent,
  DesktopBootstrapStageDescriptor,
  DesktopBootstrapStageResult,
  DesktopBootstrapStageState,
  DesktopBootstrapState,
  DesktopConnectionProbeResult
} from '@/global'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, ChevronDown, ChevronRight, iconSize, Loader2, LogIn } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { deriveProviderShape, signInLabel } from './boot-failure-reauth'

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
  now: number
}

type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'

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

function StageRow({ descriptor, result, now }: StageRowProps) {
  const { t } = useI18n()
  const copy = t.install
  const state: DesktopBootstrapStageState = result?.state || 'pending'

  const elapsed =
    state === 'running' && typeof result?.startedAt === 'number' ? formatElapsed(now - result.startedAt) : ''

  const icon = useMemo(() => {
    switch (state) {
      case 'running':
        return <Loader className="size-6" type="fourier-flow" />

      case 'succeeded':

      case 'skipped':
        return <Codicon className="text-muted-foreground" name="check" size="0.8125rem" />

      case 'failed':
        return <ErrorIcon size="1rem" />

      case 'pending':

      default:
        return <div className="size-1.5 rounded-full border border-(--ui-stroke-secondary)" />
    }
  }, [state])

  const reason = result?.json?.reason || result?.error || null

  return (
    <li className="flex items-center gap-3 px-3 py-1">
      {state === 'running' && (
        <div className="-mr-2 -ml-4 flex size-6 flex-shrink-0 items-center justify-center">{icon}</div>
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className={cn('truncate text-sm', state === 'running' ? 'font-medium' : 'text-muted-foreground')}>
            {formatStageName(descriptor.name)}
          </span>
          {state !== 'running' && <span className="flex size-4 shrink-0 items-center justify-center">{icon}</span>}
        </div>
        {reason && state !== 'pending' && <p className="mt-0.5 truncate text-xs text-muted-foreground">{reason}</p>}
      </div>
      <span className="flex-shrink-0 text-xs tabular-nums text-muted-foreground">
        {state === 'running'
          ? elapsed
            ? `${copy.stageStates[state]} · ${elapsed}`
            : copy.stageStates[state]
          : null}
        {state === 'succeeded' || state === 'skipped' ? formatDuration(result?.durationMs) : null}
        {state === 'failed' ? copy.stageStates[state] : null}
      </span>
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

function RemoteConnectCard() {
  const { t } = useI18n()
  const copy = t.install
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteToken, setRemoteToken] = useState('')
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [messageKind, setMessageKind] = useState<'error' | 'success' | null>(null)
  const [signingIn, setSigningIn] = useState(false)
  const [testing, setTesting] = useState(false)
  const [applying, setApplying] = useState(false)
  const probeSeq = useRef(0)
  const trimmedUrl = remoteUrl.trim()

  useEffect(() => {
    if (!trimmedUrl || !/^https?:\/\//i.test(trimmedUrl)) {
      setProbeStatus('idle')
      setProbe(null)

      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.probeConnectionConfig) {
      return
    }

    const seq = ++probeSeq.current
    setProbeStatus('probing')
    setMessage(null)
    setMessageKind(null)

    const timer = window.setTimeout(() => {
      void desktop
        .probeConnectionConfig(trimmedUrl)
        .then(result => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(result)
          setProbeStatus(result.reachable ? 'done' : 'error')
        })
        .catch(error => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(null)
          setProbeStatus('error')
          setMessage(error instanceof Error ? error.message : String(error))
          setMessageKind('error')
        })
    }, 400)

    return () => window.clearTimeout(timer)
  }, [trimmedUrl])

  const authResolved = probeStatus === 'done' && probe?.authMode !== 'unknown'
  const authMode = authResolved ? probe.authMode : 'token'
  const providerShape = deriveProviderShape(probe?.providers)
  const signInActionLabel = signInLabel(
    { url: trimmedUrl, ...providerShape },
    {
      identityProvider: t.boot.failure.identityProvider,
      remoteGateway: copy.remoteSignIn,
      withProvider: copy.remoteSignInWithProvider
    }
  )
  const oauthReady = authMode === 'oauth' && Boolean(trimmedUrl) && messageKind === 'success'
  const tokenReady = authMode === 'token' && Boolean(trimmedUrl) && Boolean(remoteToken.trim())
  const canTest = authMode === 'oauth' ? oauthReady : tokenReady

  const payload = () => ({
    mode: 'remote' as const,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl
  })

  const testConnection = async () => {
    if (!canTest) {
      setMessage(authMode === 'oauth' ? copy.remoteSignInRequired : copy.remoteConnectFirst)
      setMessageKind('error')

      return
    }

    setTesting(true)

    try {
      const result = await window.hermesDesktop.testConnectionConfig(payload())
      setMessage(copy.remoteConnected(result.baseUrl, result.version ?? undefined))
      setMessageKind('success')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
      setMessageKind('error')
    } finally {
      setTesting(false)
    }
  }

  const connectRemote = async () => {
    if (!canTest) {
      setMessage(authMode === 'oauth' ? copy.remoteSignInRequired : copy.remoteConnectFirst)
      setMessageKind('error')

      return
    }

    setApplying(true)

    try {
      await window.hermesDesktop.applyConnectionConfig(payload())
    } catch (error) {
      setMessage(error instanceof Error ? error.message : copy.remoteApplyFailed)
      setMessageKind('error')
      setApplying(false)
    }
  }

  const signIn = async () => {
    if (!trimmedUrl) {
      setMessage(copy.remoteConnectFirst)
      setMessageKind('error')

      return
    }

    setSigningIn(true)

    try {
      await window.hermesDesktop.saveConnectionConfig({
        mode: 'remote',
        remoteAuthMode: 'oauth',
        remoteUrl: trimmedUrl
      })

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)

      if (result.connected) {
        setMessage(copy.remoteSignInConnected)
        setMessageKind('success')

        return
      }

      setMessage(copy.remoteSignInRequired)
      setMessageKind('error')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
      setMessageKind('error')
    } finally {
      setSigningIn(false)
    }
  }

  const probeCopy = useMemo(() => {
    if (probeStatus === 'probing') {
      return copy.remoteProbeChecking
    }

    if (probeStatus === 'error') {
      return probe?.error || copy.remoteProbeFailed
    }

    if (probeStatus === 'done') {
      if (probe?.authMode === 'oauth') {
        return `${copy.remoteProbeOauth}${probe?.providers?.length ? ` ${signInActionLabel}.` : ''}`
      }

      if (probe?.authMode === 'token') {
        return copy.remoteProbeToken
      }
    }

    return copy.remoteProbeIdle
  }, [copy, probe, probeStatus, signInActionLabel])

  return (
    <div className="mt-6 rounded-xl border border-(--stroke-nous) bg-muted/30 p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">{copy.connectExistingTitle}</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{copy.connectExistingDesc}</p>
        </div>
      </div>

      <div className="mt-4 grid gap-4">
        <div className="grid gap-1.5">
          <label className="text-xs font-medium text-muted-foreground" htmlFor="remote-dashboard-url">
            {copy.remoteUrlLabel}
          </label>
          <Input
            id="remote-dashboard-url"
            onChange={event => setRemoteUrl(event.target.value)}
            placeholder={copy.remoteUrlPlaceholder}
            type="url"
            value={remoteUrl}
          />
        </div>

        <div className="rounded-lg border border-border/70 bg-background/70 px-3 py-2 text-xs text-muted-foreground">
          <div className="font-medium text-foreground">{copy.remoteProbeLabel}</div>
          <div className="mt-1">{probeCopy}</div>
        </div>

        {authResolved && authMode === 'oauth' ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button disabled={!trimmedUrl || signingIn || applying} onClick={() => void signIn()} size="sm">
              {signingIn ? <Loader2 className="animate-spin" /> : <LogIn />}
              {signInActionLabel}
            </Button>
            {oauthReady ? <span className="text-xs text-emerald-700">{copy.remoteSignInConnected}</span> : null}
          </div>
        ) : null}

        {authResolved && authMode === 'token' ? (
          <div className="grid gap-1.5">
            <label className="text-xs font-medium text-muted-foreground" htmlFor="remote-session-token">
              {copy.remoteTokenLabel}
            </label>
            <Input
              id="remote-session-token"
              onChange={event => setRemoteToken(event.target.value)}
              placeholder={copy.remoteTokenPlaceholder}
              type="password"
              value={remoteToken}
            />
          </div>
        ) : null}

        {message ? (
          <div
            className={cn(
              'rounded-lg px-3 py-2 text-xs',
              messageKind === 'error' ? 'border border-destructive/30 bg-destructive/10 text-destructive' : '',
              messageKind === 'success' ? 'border border-emerald-300/60 bg-emerald-500/10 text-emerald-700' : ''
            )}
          >
            {message}
          </div>
        ) : null}

        <div className="flex flex-wrap gap-2">
          <Button disabled={!canTest || testing || applying} onClick={() => void testConnection()} size="sm" variant="secondary">
            {testing ? <Loader2 className="animate-spin" /> : null}
            {copy.remoteTest}
          </Button>
          <Button disabled={!canTest || applying} onClick={() => void connectRemote()} size="sm">
            {applying ? <Loader2 className="animate-spin" /> : null}
            {applying ? copy.remoteApplying : copy.remoteConnect}
          </Button>
        </div>
      </div>
    </div>
  )
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
        <div className="w-full max-w-2xl rounded-xl border border-(--stroke-nous) bg-card p-8 shadow-nous">
          <h2 className="text-2xl font-semibold tracking-tight">{copy.oneTimeTitle}</h2>
          <p className="mt-2 text-sm text-muted-foreground">{copy.unsupportedDesc(platformLabel)}</p>

          <div className="mt-4">
            <div className="mb-1.5 text-xs font-medium text-muted-foreground">{copy.installCommand}</div>
            <pre className="overflow-x-auto rounded-md border border-(--stroke-nous) px-3 py-2.5 font-mono text-[12px]">
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

          <div className="mt-6 flex items-center justify-between pt-2">
            <span className="text-xs text-muted-foreground">
              {copy.installTo} <code className="font-mono text-(--ui-text-secondary)">{ups.activeRoot}</code>
            </span>
            <Button onClick={() => window.location.reload()} size="sm" variant="default">
              {copy.retryAfterRun}
            </Button>
          </div>

          <RemoteConnectCard />
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
  // Count the running stage as half-done so the bar advances *during* a long
  // stage instead of sitting frozen at the last completed step while its logs
  // stream (e.g. "0 of 2" pinned at 0% for the whole first stage).
  const progressUnits = completedCount + (!failed && currentStage ? 0.5 : 0)
  const progressPct = totalCount > 0 ? Math.round((progressUnits / totalCount) * 100) : 0
  const currentStartedAt = currentStage ? state.stages[currentStage]?.startedAt : null
  const currentElapsed = typeof currentStartedAt === 'number' ? formatElapsed(now - currentStartedAt) : ''

  return (
    <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-background/90 backdrop-blur-md p-4">
      <div className="flex w-full max-w-2xl max-h-[90vh] flex-col rounded-xl border border-(--stroke-nous) bg-card shadow-nous">
        {/* Header -- always visible, never scrolls */}
        <div className="flex flex-shrink-0 items-start gap-4 p-8 pb-4">
          {!failed && <BrandMark className="size-11 shrink-0" />}
          <div className="min-w-0">
            <h2 className="text-xl font-semibold tracking-tight">
              {failed ? copy.failedTitle : state.active ? copy.settingUpTitle : copy.finishingTitle}
            </h2>
            <p className="mt-1.5 text-sm text-muted-foreground">{failed ? copy.failedDesc : copy.activeDesc}</p>
          </div>
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
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
                <div
                  className={cn('h-full transition-all duration-300', failed ? 'bg-destructive' : 'bg-primary')}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          )}

          {totalCount === 0 && state.active && (
            <div className="mb-4 flex items-center gap-2.5 text-sm text-muted-foreground">
              <Loader className="size-5" type="fourier-flow" />
              <span>{copy.fetchingManifest}</span>
            </div>
          )}

          {failed && state.error && (
            <div className="mb-4 flex items-start gap-2 text-sm">
              <ErrorIcon className="mt-0.5 shrink-0" size="1rem" />
              <div className="min-w-0">
                <div className="font-medium text-destructive">{copy.error}</div>
                <p className="mt-0.5 whitespace-pre-wrap break-words text-foreground/90">{state.error}</p>
              </div>
            </div>
          )}

          {stages.length > 0 && (
            <ol className="mb-4 space-y-0.5">
              {stages.map(stage => (
                <StageRow descriptor={stage} key={stage.name} now={now} result={state.stages[stage.name]} />
              ))}
            </ol>
          )}

          <div className="pt-3">
            <Button
              className="-ml-2 text-muted-foreground hover:text-foreground"
              onClick={() => setLogOpen(v => !v)}
              size="xs"
              type="button"
              variant="ghost"
            >
              {logOpen ? <ChevronDown className={iconSize.sm} /> : <ChevronRight className={iconSize.sm} />}
              <span>{logOpen ? copy.hideOutput : copy.showOutput}</span>
              <span className="ml-1 tabular-nums">({copy.lines(state.log.length)})</span>
            </Button>

            {logOpen && (
              <LogView className={cn('mt-2', failed ? 'max-h-96' : 'max-h-64')}>
                {state.log.length === 0 ? (
                  <div>{copy.noOutput}</div>
                ) : (
                  <>
                    {state.log.map((entry, i) => (
                      <div className={cn(entry.stream === 'stderr' && 'text-muted-foreground/70')} key={i}>
                        {entry.stage ? <span className="text-muted-foreground/60">[{entry.stage}] </span> : null}
                        <span>{entry.line}</span>
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </>
                )}
              </LogView>
            )}
          </div>
        </div>

        {/* Active footer: let the user actually cancel a running install. */}
        {state.active && !failed && (
          <div className="flex-shrink-0 bg-card p-4">
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
                {cancelling ? <Loader className="size-4" type="fourier-flow" /> : null}
                {cancelling ? copy.cancelling : copy.cancelInstall}
              </Button>
            </div>
          </div>
        )}

        {/* Footer -- always visible, never scrolls; only renders on failure */}
        {failed && (
          <div className="flex-shrink-0 bg-card p-4">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">
                {copy.transcriptSaved}{' '}
                <code className="font-mono text-(--ui-text-secondary)">%LOCALAPPDATA%\hermes\logs\</code>
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
