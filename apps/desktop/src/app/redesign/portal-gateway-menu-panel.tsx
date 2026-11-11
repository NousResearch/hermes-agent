import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

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
import { $activeConnectionId, $connectionsRegistry } from '@/store/connections'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { recordIaAction } from '@/store/ia-metrics'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { notifyError } from '@/store/notifications'
import { runGatewayRestart } from '@/store/system-actions'
import type { StatusResponse } from '@/types/hermes'

interface GatewayMenuPanelProps {
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

export function PortalGatewayMenuPanel({
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
  const gatewayName = registry?.connections.find(connection => connection.id === connectionId)?.label ?? copy.gateway
  const [activityOpen, setActivityOpen] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)
  const openedState = useRef(gatewayState)
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

  // Shared restart helper: never rejects and surfaces progress in the statusbar
  // gateway indicator, so just fire and close.
  const restart = () => {
    onClose()
    void runGatewayRestart()
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

  return (
    <div className="space-y-3 p-3 text-xs">
      <div className="flex min-h-[2.125rem] items-center justify-between gap-2">
        <h2 className="min-w-0 truncate text-[0.8125rem] font-semibold" title={gatewayName}>
          {gatewayName}
        </h2>
        <div className="flex shrink-0 items-center gap-1.5">
          {/* An open transport can still be wedged; recovery must remain reachable. */}
          <Tip label={copy.reconnectGateway}>
            <Button
              aria-label={copy.reconnectGateway}
              className={GATEWAY_ICON_BUTTON}
              disabled={reconnecting}
              onClick={reconnect}
              size="icon-compact"
              variant="outline"
            >
              <RefreshCw className={cn(reconnecting && 'animate-spin')} />
            </Button>
          </Tip>
          {advanced && (
            <>
              <Tip label={copy.openSystem}>
                <Button
                  aria-label={copy.openSystem}
                  className={GATEWAY_ICON_BUTTON}
                  onClick={openSystem}
                  size="icon-compact"
                  variant="outline"
                >
                  <LayoutDashboard />
                </Button>
              </Tip>
              {/* Keep the disruptive restart visibly distinct from connection recovery. */}
              <span aria-hidden className="h-4 w-px bg-[color-mix(in_srgb,var(--ui-base)_15%,transparent)]" />
              <Tip label={t.commandCenter.restartGateway}>
                <Button
                  aria-label={t.commandCenter.restartGateway}
                  className={GATEWAY_ICON_BUTTON}
                  onClick={restart}
                  size="icon-compact"
                  variant="destructiveOutline"
                >
                  <Power />
                </Button>
              </Tip>
            </>
          )}
        </div>
      </div>

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
          className="min-h-[2.125rem] w-full justify-between rounded-[6px] border border-dashed border-[color-mix(in_srgb,var(--ui-base)_12%,transparent)] bg-[color-mix(in_srgb,var(--ui-base)_2%,transparent)] px-2.5 hover:bg-[color-mix(in_srgb,var(--ui-base)_5%,transparent)]"
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

function GatewayReadiness({
  gatewayState,
  inferenceStatus
}: Pick<GatewayMenuPanelProps, 'gatewayState' | 'inferenceStatus'>) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu
  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true

  const connectionLabel = gatewayOpen
    ? copy.connected
    : gatewayConnecting
      ? copy.connecting
      : prettyState(gatewayState || copy.offline)

  const inferenceKnown = inferenceStatus && inferenceStatus.source !== 'fallback'

  const inferenceLabel = !gatewayOpen
    ? copy.disconnected
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
          className={good ? 'bg-(--ui-success)' : undefined}
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

const GATEWAY_CARD =
  'rounded-[6px] border border-[color-mix(in_srgb,var(--ui-base)_8%,transparent)] bg-[color-mix(in_srgb,var(--ui-base)_4%,transparent)] px-2.5 py-2'

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--portal-brand-ink)">{children}</div>
  )
}
