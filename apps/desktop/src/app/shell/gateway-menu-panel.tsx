import { IconLayoutDashboard } from '@tabler/icons-react'

import { StatusDot, type StatusTone } from '@/components/status-dot'
import { Button } from '@/components/ui/button'
import { Activity, AlertCircle } from '@/lib/icons'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { cn } from '@/lib/utils'
import type { StatusResponse } from '@/types/hermes'

interface GatewayMenuPanelProps {
  gatewayState: string
  inferenceStatus: RuntimeReadinessResult | null
  logLines: readonly string[]
  onOpenSystem: () => void
  statusSnapshot: StatusResponse | null
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
  logLines,
  onOpenSystem,
  statusSnapshot
}: GatewayMenuPanelProps) {
  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true

  const connectionLabel = gatewayOpen
    ? 'Connected'
    : gatewayConnecting
      ? 'Connecting'
      : prettyState(gatewayState || 'offline')

  const inferenceLabel = gatewayOpen
    ? inferenceStatus?.ready
      ? 'Inference ready'
      : inferenceStatus
        ? 'Inference not ready'
        : 'Checking inference'
    : 'Disconnected'

  const platforms = Object.entries(statusSnapshot?.gateway_platforms || {}).sort(([l], [r]) => l.localeCompare(r))
  const recentLogs = logLines.slice(-5)

  return (
    <div className="text-sm">
      <div className="flex items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          {inferenceReady ? (
            <Activity className="size-3.5 text-primary" />
          ) : (
            <AlertCircle className={cn('size-3.5', gatewayOpen ? 'text-amber-600' : 'text-destructive')} />
          )}
          <span className="font-medium">Gateway</span>
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <StatusDot tone={inferenceReady ? 'good' : gatewayOpen ? 'warn' : 'bad'} />
            {inferenceLabel}
          </span>
        </div>
        <div className="flex items-center">
          <Button
            aria-label="Open system panel"
            className="text-muted-foreground hover:text-foreground"
            onClick={onOpenSystem}
            size="icon-sm"
            title="Open system panel"
            variant="ghost"
          >
            <IconLayoutDashboard />
          </Button>
        </div>
      </div>

      <div className="border-t border-border/50 px-3 py-2 text-xs text-muted-foreground">
        <div>Connection: {connectionLabel}</div>
        {inferenceStatus?.reason && <div className="mt-1 line-clamp-3">{inferenceStatus.reason}</div>}
      </div>

      {recentLogs.length > 0 && (
        <div className="border-t border-border/50 px-3 py-2">
          <SectionLabel>Recent activity</SectionLabel>
          <ul className="mt-1.5 space-y-0.5">
            {recentLogs.map((line, index) => (
              <li
                className="truncate font-mono text-[0.68rem] text-muted-foreground/85"
                key={`${index}:${line}`}
                title={line.trim()}
              >
                {trimLogLine(line) || '\u00A0'}
              </li>
            ))}
          </ul>
          <button
            className="mt-1.5 text-[0.66rem] font-medium text-muted-foreground hover:text-foreground"
            onClick={onOpenSystem}
            type="button"
          >
            View all logs →
          </button>
        </div>
      )}

      {platforms.length > 0 && (
        <div className="border-t border-border/50 px-3 py-2">
          <SectionLabel>Messaging platforms</SectionLabel>
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
