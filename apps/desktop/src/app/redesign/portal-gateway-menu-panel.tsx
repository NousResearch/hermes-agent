import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { StatusDot } from '@/components/status-dot'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { LogView } from '@/components/ui/log-view'
import { Tip } from '@/components/ui/tooltip'
import { getLogs } from '@/hermes'
import { useI18n } from '@/i18n'
import { LayoutDashboard, Power, RefreshCw } from '@/lib/icons'
import { platformStatusTone } from '@/lib/platform-status'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import { $activeConnectionId, $connectionsRegistry, selectConnection } from '@/store/connections'
import { $fleetRoster, refreshFleetRoster } from '@/store/fleet-roster'
import { requestGatewayForAgent } from '@/store/gateway'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { recordIaAction } from '@/store/ia-metrics'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import { runGatewayRestart } from '@/store/system-actions'
import type { StatusResponse } from '@/types/hermes'

import { COMMAND_CENTER_ROUTE, SETTINGS_ROUTE } from '../routes'
import { useStatusSnapshot } from '../shell/hooks/use-status-snapshot'
import { useStableNavigate } from '../stable-navigate'

interface GatewayMenuPanelProps {
  /** The machine's name as the opener shows it (a sidebar machine row); else the registry label. */
  name?: string
  gatewayState: string
  inferenceStatus: RuntimeReadinessResult | null
  onClose: () => void
  onOpenSystem: () => void
  statusSnapshot: StatusResponse | null
}

const LOG_TAIL = 120
const LOG_VISIBLE = 40
const LOG_POLL_MS = 3_000

// Per-connection WebSocket churn (accept/close/heartbeat) drowns out anything
// useful — strip it so the tail reads as real gateway activity at a glance.
const LOG_NOISE_RE = /\bws (?:accepted|closed|response sent|ping|pong)\b/i

// Live tail while the popover is mounted (i.e. open): poll on a tight cadence
// and stop on unmount, instead of a global always-on status poll.
function useGatewayLogTail() {
  const [state, setState] = useState<{ lines: string[]; error: string | null; loading: boolean }>({
    lines: [],
    error: null,
    loading: true
  })

  useEffect(() => {
    let cancelled = false

    // async: getLogs THROWS (not rejects) when the desktop bridge is missing
    // (plain-browser mode) — a sync throw here would take down the root
    // error boundary before the .catch even attaches.
    const load = async () => {
      try {
        const res = await getLogs({ file: 'gui', lines: LOG_TAIL })

        if (cancelled) {
          return
        }

        setState({
          lines: res.lines
            .map(line => line.trim())
            .filter(line => line && !LOG_NOISE_RE.test(line))
            .slice(-LOG_VISIBLE),
          error: null,
          loading: false
        })
      } catch (error) {
        if (!cancelled) {
          setState(previous => ({ ...previous, error: String(error), loading: false }))
        }
      }
    }

    void load()
    const timer = window.setInterval(load, LOG_POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  return state
}

const prettyState = (state: string) => state.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

// Strip leading "YYYY-MM-DD HH:MM:SS,mmm " and "[runtime_id] " prefixes from
// log lines so they don't dominate the display. Full text preserved on hover.
const TIMESTAMP_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+/
const RUNTIME_BRACKET_RE = /^\[[^\]]+]\s+/
const trimLogLine = (raw: string) => raw.trim().replace(TIMESTAMP_RE, '').replace(RUNTIME_BRACKET_RE, '')

/** The machine's name and its header actions (616:1685): Reconnect always; with developer chrome
 *  also the system panel and, set apart, Restart (asks first). The offline frame (616:1781) keeps only
 *  the name: its one action is the full-width Reconnect below. */
function GatewayHeader({ actions, name }: { actions?: ReactNode; name: string }) {
  return (
    <div className="flex min-h-[2.125rem] items-center justify-between gap-2">
      <h2 className="min-w-0 truncate text-[0.8125rem] font-semibold" title={name}>
        {name}
      </h2>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </div>
  )
}

function IconAction({
  children,
  danger,
  disabled,
  label,
  onClick
}: {
  children: ReactNode
  danger?: boolean
  disabled?: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        className={GATEWAY_ICON_BUTTON}
        disabled={disabled}
        onClick={onClick}
        size="icon-compact"
        variant={danger ? 'destructiveOutline' : 'outline'}
      >
        {children}
      </Button>
    </Tip>
  )
}

/** The status popover of the machine this window is connected to: live connection and inference
 *  state, its messaging platforms and recent gateway log lines. Both entry points (the rail's System
 *  popover and a machine row's status in the sidebar) render this one panel. */
export function PortalGatewayMenuPanel({
  name,
  gatewayState,
  inferenceStatus,
  onClose,
  onOpenSystem,
  statusSnapshot
}: GatewayMenuPanelProps) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu
  const advanced = useStore($showsAdvancedChrome)
  const connectionId = useStore($activeConnectionId)
  const registry = useStore($connectionsRegistry)
  const scope = useStore($apiRequestScope)

  const gatewayName =
    name ?? registry?.connections.find(connection => connection.id === connectionId)?.label ?? copy.gateway

  const [activityOpen, setActivityOpen] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)
  const openedState = useRef(gatewayState)
  // Offline (616:1781): what the machine's platforms and logs say is unknown, so neither is shown.
  const offline = gatewayState !== 'open' && gatewayState !== 'connecting'
  useEffect(() => {
    const state = openedState.current
    recordIaAction(
      state === 'open'
        ? 'ia.gateway.popover_open'
        : state === 'connecting'
          ? 'ia.gateway.popover_connecting'
          : 'ia.gateway.popover_offline',
      'state'
    )
  }, [])

  // Both jumps open the system panel, which owns the full view — so dismiss the
  // little status popover on the way out.
  const openSystem = () => {
    recordIaAction('ia.gateway.system')
    onClose()
    onOpenSystem()
  }

  // A restart drops every running turn on the machine, so it asks first. The shared helper never
  // rejects and reports progress in the gateway indicator.
  const restart = async () => {
    const confirmed = await confirm({
      title: copy.restartTitle,
      description: copy.restartBody,
      confirmLabel: t.commandCenter.restartGateway,
      destructive: true
    })

    if (confirmed) {
      onClose()
      void runGatewayRestart()
    }
  }

  const reconnect = () => {
    if (reconnecting) {
      return
    }

    recordIaAction('ia.gateway.reconnect')
    setReconnecting(true)
    void reconnectGateway()
      .catch(err => notifyError(err, copy.reconnectGateway))
      .finally(() => setReconnecting(false))
  }

  const platforms = Object.entries(statusSnapshot?.gateway_platforms || {}).sort(([l], [r]) => l.localeCompare(r))

  if (offline) {
    return (
      <div className="space-y-3 p-3 text-xs">
        <GatewayHeader name={gatewayName} />
        <GatewayReadiness gatewayState={gatewayState} inferenceStatus={inferenceStatus} />
        <Button className={GATEWAY_PRIMARY} disabled={reconnecting} onClick={reconnect} size="sm" variant="default">
          <RefreshCw className={cn(reconnecting && 'animate-spin')} />
          {copy.reconnectGateway}
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-3 p-3 text-xs">
      <GatewayHeader
        actions={
          <>
            {/* An open transport can still be wedged; recovery must remain reachable. */}
            <IconAction disabled={reconnecting} label={copy.reconnectGateway} onClick={reconnect}>
              <RefreshCw className={cn(reconnecting && 'animate-spin')} />
            </IconAction>
            {advanced && (
              <>
                <IconAction label={copy.openSystem} onClick={openSystem}>
                  <LayoutDashboard />
                </IconAction>
                {/* Keep the disruptive restart visibly distinct from connection recovery. */}
                <span aria-hidden className="h-4 w-px bg-[color-mix(in_srgb,var(--ui-base)_15%,transparent)]" />
                <IconAction danger label={t.commandCenter.restartGateway} onClick={() => void restart()}>
                  <Power />
                </IconAction>
              </>
            )}
          </>
        }
        name={gatewayName}
      />

      <GatewayReadiness gatewayState={gatewayState} inferenceStatus={inferenceStatus} />
      {platforms.length > 0 && (
        <section className="space-y-1.5">
          <SectionLabel>{copy.messagingPlatforms}</SectionLabel>
          <ul className={cn(GATEWAY_CARD, 'space-y-0.5')}>
            {platforms.map(([name, platform]) => (
              <li className="flex min-h-6 items-center justify-between gap-2" key={name}>
                <span className="truncate font-medium capitalize">{name}</span>
                <span className="flex shrink-0 items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)">
                  <StatusDot tone={platformStatusTone({ state: platform.state })} />
                  {prettyState(platform.state)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {advanced ? (
        <div className="flex items-center justify-between gap-2">
          <SectionLabel>{copy.recentActivity}</SectionLabel>
          <Button className="text-(--portal-brand-ink)" onClick={openSystem} size="xs" variant="text">
            {copy.viewAllLogs}
          </Button>
        </div>
      ) : (
        <Button
          aria-expanded={activityOpen}
          className={cn(GATEWAY_CARD, 'min-h-[2.125rem] w-full justify-between font-medium hover:bg-[color-mix(in_srgb,var(--ui-base)_7%,transparent)]')}
          onClick={() => setActivityOpen(value => !value)}
          size="sm"
          variant="ghost"
        >
          {copy.recentActivity}
          <Codicon className="text-muted-foreground" name={activityOpen ? 'chevron-down' : 'chevron-right'} />
        </Button>
      )}
      {(advanced || activityOpen) && (
        <RecentGatewayActivity key={`${scope.connectionId ?? ''}:${scope.profile ?? ''}`} />
      )}
    </div>
  )
}

/** The connected machine's panel for a surface that holds no status snapshot of its own (a machine
 *  row in the sidebar): reads the same live status the statusbar does, only while it is open. */
export function ActiveMachineStatusPanel({ name, onClose }: { name: string; onClose: () => void }) {
  const gatewayState = useStore($gatewayState)
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const navigate = useStableNavigate()

  const requestStatus = useCallback(
    <T,>(method: string, params?: Record<string, unknown>) =>
      requestGatewayForAgent<T>(connectionId, profile, method, params),
    [connectionId, profile]
  )

  const scopeKey = `${connectionId ?? ''}\0${profile}`
  const { inferenceStatus, statusSnapshot } = useStatusSnapshot(gatewayState, requestStatus, scopeKey)

  return (
    <PortalGatewayMenuPanel
      gatewayState={gatewayState}
      inferenceStatus={inferenceStatus}
      key={scopeKey}
      name={name}
      onClose={onClose}
      onOpenSystem={() => navigate(`${COMMAND_CENTER_ROUTE}?section=system`)}
      statusSnapshot={statusSnapshot}
    />
  )
}

interface MachineStatusPanelProps {
  /** The registry connection this machine is, or null for this app's own machine. */
  connectionId: null | string
  label: string
  onClose: () => void
}

/** The status popover of a machine this window is NOT connected to. Only what the app actually
 *  knows about it is shown: the fleet probe's reachability. Inference, platforms and logs are only
 *  readable over a connection, so they are omitted rather than guessed (the offline frame's rule);
 *  the actions are the real ones for it: probe again, switch to it, or manage it in Settings. */
export function MachineStatusPanel({ connectionId, label, onClose }: MachineStatusPanelProps) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu
  const navigate = useStableNavigate()
  const roster = useStore($fleetRoster)
  const registered = useStore($connectionsRegistry)?.connections.some(connection => connection.id === connectionId)
  const source = roster?.sources.find(entry => entry.connectionId === connectionId)
  const [checking, setChecking] = useState(false)
  const [switching, setSwitching] = useState(false)

  const check = () => {
    setChecking(true)
    void refreshFleetRoster({ force: true }).finally(() => setChecking(false))
  }

  const switchTo = () => {
    if (!connectionId || switching) {
      return
    }

    setSwitching(true)
    void selectConnection(connectionId)
      .then(onClose)
      .catch(err => notifyError(err, copy.switchToMachine))
      .finally(() => setSwitching(false))
  }

  const state = source ? (source.reachable ? 'open' : 'closed') : 'connecting'

  return (
    <div className="space-y-3 p-3 text-xs">
      <GatewayHeader
        actions={
          <IconAction disabled={checking} label={copy.checkAgain} onClick={check}>
            <RefreshCw className={cn(checking && 'animate-spin')} />
          </IconAction>
        }
        name={label}
      />
      <dl className={cn(GATEWAY_CARD, 'space-y-1')}>
        <StatusRow
          good={state === 'open'}
          label={copy.connectionLabel}
          value={state === 'open' ? copy.connected : state === 'closed' ? copy.offline : copy.connecting}
          waiting={state === 'connecting'}
        />
      </dl>
      {source?.error && <p className="text-muted-foreground">{source.error}</p>}
      {registered && (
        <Button className={GATEWAY_PRIMARY} disabled={switching} onClick={switchTo} size="sm" variant="default">
          {copy.switchToMachine}
        </Button>
      )}
      <Button
        className="text-(--portal-brand-ink)"
        onClick={() => {
          onClose()
          navigate(`${SETTINGS_ROUTE}?tab=connections`)
        }}
        size="xs"
        variant="text"
      >
        {copy.machineSettings}
      </Button>
    </div>
  )
}

function GatewayReadiness({
  gatewayState,
  inferenceStatus
}: Pick<GatewayMenuPanelProps, 'gatewayState' | 'inferenceStatus'>) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu
  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true

  const connectionLabel = gatewayOpen ? copy.connected : gatewayConnecting ? copy.connecting : copy.offline

  const inferenceKnown = inferenceStatus && inferenceStatus.source !== 'fallback'

  const inferenceLabel = !gatewayOpen
    ? copy.inferenceNotReady
    : !inferenceKnown
      ? copy.checkingInference
      : inferenceStatus.ready
        ? copy.inferenceReady
        : copy.inferenceNotReady

  return (
    <>
      <dl className={cn(GATEWAY_CARD, 'space-y-1')}>
        <StatusRow
          good={gatewayOpen}
          label={copy.connectionLabel}
          value={connectionLabel}
          waiting={gatewayConnecting}
        />
        <StatusRow
          good={inferenceReady}
          label={copy.inferenceLabel}
          value={inferenceLabel}
          waiting={gatewayOpen && !inferenceKnown}
        />
      </dl>
      {gatewayOpen && inferenceStatus?.reason && <p className="text-muted-foreground">{inferenceStatus.reason}</p>}
    </>
  )
}

function StatusRow({ label, good, waiting, value }: { label: string; good: boolean; waiting: boolean; value: string }) {
  return (
    <div className="flex min-h-5 items-center justify-between gap-3">
      <dt className="text-[0.6875rem] text-(--ui-text-tertiary)">{label}</dt>
      <dd
        className={cn(
          'flex items-center gap-2 text-right font-medium',
          good ? 'text-(--ui-success)' : 'text-(--ui-text-secondary)'
        )}
      >
        <StatusDot
          className={
            good
              ? 'bg-(--ui-success)'
              : waiting
                ? undefined
                : 'bg-transparent ring-1 ring-[color-mix(in_srgb,var(--ui-base)_60%,transparent)] ring-inset'
          }
          tone={good ? 'good' : waiting ? 'warn' : 'muted'}
        />
        {value}
      </dd>
    </div>
  )
}

function RecentGatewayActivity() {
  const { t } = useI18n()
  const { lines, error, loading } = useGatewayLogTail()
  const scrollRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines])

  return (
    <div className="space-y-2">
      {error && (
        <p className="text-destructive" role="alert">
          {error}
        </p>
      )}
      {lines.length > 0 ? (
        <LogView
          className="max-h-44 rounded-[6px] border-[color-mix(in_srgb,var(--ui-base)_8%,transparent)] text-[0.625rem] leading-normal"
          ref={scrollRef}
        >
          {lines.map(trimLogLine).join('\n')}
        </LogView>
      ) : (
        !error && (
          <p className="text-muted-foreground">{loading ? t.commandCenter.loadingStatus : t.commandCenter.noLogs}</p>
        )
      )}
    </div>
  )
}

// Gateway popover geometry (616:1685): 28px r6 icon controls and r6 inset cards on 4% fill.
const GATEWAY_ICON_BUTTON = 'rounded-[6px]'

// The offline frame's one action (616:1781): full-width brand fill, r6.
const GATEWAY_PRIMARY =
  'w-full rounded-[6px] bg-(--portal-brand-fill) text-(--portal-on-brand) hover:bg-[color-mix(in_srgb,var(--portal-brand-fill)_86%,var(--ui-base))]'

const GATEWAY_CARD =
  'rounded-[6px] border border-[color-mix(in_srgb,var(--ui-base)_8%,transparent)] bg-[color-mix(in_srgb,var(--ui-base)_4%,transparent)] px-2.5 py-2'

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--portal-brand-ink)">{children}</div>
  )
}
