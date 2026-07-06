import { useEffect, useRef, useState } from 'react'

import { StatusDot, type StatusTone } from '@/components/status-dot'
import { Button } from '@/components/ui/button'
import { LogView } from '@/components/ui/log-view'
import { Tip } from '@/components/ui/tooltip'
import { getLogs } from '@/hermes'
import { useI18n } from '@/i18n'
import { LayoutDashboard, RefreshCw } from '@/lib/icons'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
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
function useGatewayLogTail(): string[] {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false

    const load = () =>
      getLogs({ file: 'gui', lines: LOG_TAIL })
        .then(res => {
          if (cancelled) {
            return
          }

          setLines(
            res.lines
              .map(line => line.trim())
              .filter(line => line && !LOG_NOISE_RE.test(line))
              .slice(-LOG_VISIBLE)
          )
        })
        .catch(() => {})

    void load()
    const timer = window.setInterval(load, LOG_POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  return lines
}

const PLATFORM_TONE: Record<string, StatusTone> = {
  connected: 'good',
  connecting: 'warn',
  retrying: 'warn',
  pending_restart: 'warn',
  startup_failed: 'bad',
  fatal: 'bad'
}

const prettyState = (state: string) => state.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

// Strip leading "YYYY-MM-DD HH:MM:SS,mmm " and "[runtime_id] " prefixes from
// log lines so they don't dominate the display. Full text preserved on hover.
const TIMESTAMP_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+/
const RUNTIME_BRACKET_RE = /^\[[^\]]+]\s+/
const trimLogLine = (raw: string) => raw.trim().replace(TIMESTAMP_RE, '').replace(RUNTIME_BRACKET_RE, '')

export function GatewayMenuPanel({
  gatewayState,
  inferenceStatus,
  onClose,
  onOpenSystem,
  statusSnapshot
}: GatewayMenuPanelProps) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu

  // Both jumps open the system panel, which owns the full view — so dismiss the
  // little status popover on the way out.
  const openSystem = () => {
    onClose()
    onOpenSystem()
  }

  // Shared restart helper: never rejects and surfaces progress in the statusbar
  // gateway indicator, so just fire and close.
  const restart = () => {
    onClose()
    void runGatewayRestart()
  }

  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true

  const connectionLabel = gatewayOpen
    ? copy.connected
    : gatewayConnecting
      ? copy.connecting
      : prettyState(gatewayState || copy.offline)

  const inferenceLabel = gatewayOpen
    ? inferenceStatus?.ready
      ? copy.inferenceReady
      : inferenceStatus
        ? copy.inferenceNotReady
        : copy.checkingInference
    : copy.disconnected

  const platforms = Object.entries(statusSnapshot?.gateway_platforms || {}).sort(([l], [r]) => l.localeCompare(r))
  const recentLogs = useGatewayLogTail()

  // Keep the tail pinned to the latest line as it streams.
  const logScrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = logScrollRef.current

    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [recentLogs])

  return (
    <div className="text-sm">
      <div className="flex items-center justify-between gap-3 px-3 py-2">
        <div className="flex min-w-0 flex-col gap-1 text-[0.7rem] leading-none">
          <span className="flex items-center gap-1.5 font-medium">
            <StatusDot tone={gatewayOpen ? 'good' : gatewayConnecting ? 'warn' : 'bad'} />
            {connectionLabel}
          </span>
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <StatusDot tone={inferenceReady ? 'good' : gatewayOpen ? 'warn' : 'bad'} />
            {inferenceLabel}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <Tip label={t.commandCenter.restartGateway}>
            <Button
              aria-label={t.commandCenter.restartGateway}
              className="text-muted-foreground hover:text-foreground"
              onClick={restart}
              size="icon-xs"
              variant="ghost"
            >
              <RefreshCw />
            </Button>
          </Tip>
          <Tip label={copy.openSystem}>
            <Button
              aria-label={copy.openSystem}
              className="text-muted-foreground hover:text-foreground"
              onClick={openSystem}
              size="icon-xs"
              variant="ghost"
            >
              <LayoutDashboard />
            </Button>
          </Tip>
        </div>
      </div>

      {inferenceStatus?.reason && (
        <div className="border-t border-border/50 px-3 py-2 text-xs text-muted-foreground">
          <div className="line-clamp-3">{inferenceStatus.reason}</div>
        </div>
      )}

      {recentLogs.length > 0 && (
        <div className="px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <SectionLabel>{copy.recentActivity}</SectionLabel>
            <Button
              className="-mr-2 h-auto py-0 font-medium leading-none text-muted-foreground"
              onClick={openSystem}
              size="xs"
              type="button"
              variant="text"
            >
              {copy.viewAllLogs}
            </Button>
          </div>
          <LogView className="mt-1.5 max-h-40 border-0 px-0" ref={logScrollRef}>
            {recentLogs.map(trimLogLine).join('\n')}
          </LogView>
        </div>
      )}

      {platforms.length > 0 && (
        <div className="border-t border-border/50 px-3 py-2">
          <SectionLabel>{copy.messagingPlatforms}</SectionLabel>
          <ul className="mt-1.5 space-y-1">
            {platforms.map(([name, platform]) => (
              <li className="flex items-center justify-between gap-2 text-xs" key={name}>
                <span className="truncate capitalize">{name}</span>
                <span className="flex items-center gap-1.5 text-[0.66rem] text-muted-foreground">
                  <StatusDot tone={PLATFORM_TONE[platform.state] || 'muted'} />
                  {prettyState(platform.state)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80">{children}</div>
  )
}
