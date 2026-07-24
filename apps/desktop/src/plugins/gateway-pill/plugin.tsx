/**
 * Gateway pill — the core statusbar gateway-health item implemented 1:1 as a
 * plugin: same trigger chrome (declarative `variant: 'menu'` StatusbarItem →
 * the app's own portal/popover plumbing, so nothing clips), same menu panel
 * (connection/inference rows, restart, reason, RECENT ACTIVITY tail,
 * messaging platforms), same copy (`useI18n`), same readiness logic
 * (`evaluateRuntimeReadiness` over `host.request`). The point: a plugin can
 * rebuild a REAL core feature through the SDK alone — only
 * `@hermes/plugin-sdk` + react (lint-fenced).
 *
 * Pattern notes:
 *  - a module-level `atom` shares the readiness poll between the live label
 *    elements and the menu panel (the same primitive `host.state` uses);
 *  - label/detail/icon of a DATA item are ReactNodes, so they can be tiny
 *    components that subscribe — a static item shape with live innards.
 */

import {
  atom,
  Button,
  cn,
  evaluateRuntimeReadiness,
  type HermesPlugin,
  host,
  icons,
  LogView,
  type RuntimeReadinessResult,
  type StatusbarItem,
  StatusDot,
  type StatusResponse,
  type StatusTone,
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useRef, useState } from 'react'

const READINESS_POLL_MS = 15_000
const LOG_TAIL = 120
const LOG_VISIBLE = 40
const LOG_POLL_MS = 3_000

// Per-connection WebSocket churn (accept/close/heartbeat) drowns out anything
// useful — strip it so the tail reads as real gateway activity at a glance.
const LOG_NOISE_RE = /\bws (?:accepted|closed|response sent|ping|pong)\b/i

// Strip leading "YYYY-MM-DD HH:MM:SS,mmm " and "[runtime_id] " prefixes.
const TIMESTAMP_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+/
const RUNTIME_BRACKET_RE = /^\[[^\]]+]\s+/
const trimLogLine = (raw: string) => raw.trim().replace(TIMESTAMP_RE, '').replace(RUNTIME_BRACKET_RE, '')

const PLATFORM_TONE: Record<string, StatusTone> = {
  connected: 'good',
  connecting: 'warn',
  retrying: 'warn',
  pending_restart: 'warn',
  startup_failed: 'bad',
  fatal: 'bad'
}

const prettyState = (state: string) => state.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

const SYSTEM_PANEL_ROUTE = '/command-center?section=system'

// ---------------------------------------------------------------------------
// Readiness poll — one loop at plugin scope, shared by label + panel.
// ---------------------------------------------------------------------------

const $readiness = atom<null | RuntimeReadinessResult>(null)

function startReadinessPoll() {
  let timer: null | number = null

  const stop = () => {
    if (timer !== null) {
      window.clearInterval(timer)
      timer = null
    }

    $readiness.set(null)
  }

  const refresh = () =>
    evaluateRuntimeReadiness(host.request)
      .then(next => $readiness.set(next))
      .catch(() => undefined)

  const sync = (gateway: string) => {
    if (gateway !== 'open') {
      stop()

      return
    }

    if (timer === null) {
      void refresh()
      timer = window.setInterval(() => void refresh(), READINESS_POLL_MS)
    }
  }

  sync(host.state.gateway.get())
  host.state.gateway.listen(sync)
}

// ---------------------------------------------------------------------------
// Live trigger innards (the item is static DATA; these subscribe).
// ---------------------------------------------------------------------------

function useHealth() {
  const gateway = useValue(host.state.gateway)
  const readiness = useValue($readiness)

  return {
    connecting: gateway === 'connecting',
    open: gateway === 'open',
    readiness,
    ready: gateway === 'open' && readiness?.ready === true
  }
}

function PillIcon() {
  const { connecting, open, ready } = useHealth()

  return (
    <span
      className={cn(
        'inline-flex',
        ready ? 'text-(--ui-text-tertiary)' : open || connecting ? 'text-amber-500' : 'text-red-400'
      )}
    >
      {ready ? <icons.Activity className="size-3" /> : <icons.AlertCircle className="size-3" />}
    </span>
  )
}

function PillDetail() {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const { connecting, open, readiness, ready } = useHealth()

  const detail = open
    ? ready
      ? copy.gatewayReady
      : readiness
        ? copy.gatewayNeedsSetup
        : copy.gatewayChecking
    : connecting
      ? copy.gatewayConnecting
      : copy.gatewayOffline

  return <>{detail}</>
}

// ---------------------------------------------------------------------------
// The menu panel — the real GatewayMenuPanel, rebuilt on SDK doors.
// ---------------------------------------------------------------------------

/** Live gui-log tail while the popover is mounted (i.e. open). */
function useGatewayLogTail(): string[] {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false

    const load = () =>
      host
        .logs({ file: 'gui', lines: LOG_TAIL })
        .then(res => {
          if (!cancelled) {
            setLines(
              res.lines
                .map(line => line.trim())
                .filter(line => line && !LOG_NOISE_RE.test(line))
                .slice(-LOG_VISIBLE)
            )
          }
        })
        .catch(() => undefined)

    void load()
    const timer = window.setInterval(load, LOG_POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  return lines
}

function Section({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('border-t border-border/50 px-3 py-2', className)}>{children}</div>
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80">{children}</div>
  )
}

function GatewayMenuPanel({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const copy = t.shell.gatewayMenu
  const gateway = useValue(host.state.gateway)
  const { readiness, ready } = useHealth()
  const [snapshot, setSnapshot] = useState<null | StatusResponse>(null)
  const recentLogs = useGatewayLogTail()

  useEffect(() => {
    void host
      .status()
      .then(setSnapshot)
      .catch(() => undefined)
  }, [])

  const openSystem = () => {
    onClose()
    host.navigate(SYSTEM_PANEL_ROUTE)
  }

  const restart = () => {
    onClose()
    void host.restartGateway().catch(() => undefined)
  }

  const gatewayOpen = gateway === 'open'
  const gatewayConnecting = gateway === 'connecting'

  const connectionLabel = gatewayOpen
    ? copy.connected
    : gatewayConnecting
      ? copy.connecting
      : prettyState(gateway || copy.offline)

  const inferenceLabel = gatewayOpen
    ? readiness?.ready
      ? copy.inferenceReady
      : readiness
        ? copy.inferenceNotReady
        : copy.checkingInference
    : copy.disconnected

  const platforms = Object.entries(snapshot?.gateway_platforms || {}).sort(([l], [r]) => l.localeCompare(r))

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
            <StatusDot tone={ready ? 'good' : gatewayOpen ? 'warn' : 'bad'} />
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
              <icons.RefreshCw />
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
              <icons.LayoutDashboard />
            </Button>
          </Tip>
        </div>
      </div>

      {readiness?.reason && (
        <Section className="text-xs text-muted-foreground">
          <div className="line-clamp-3">{readiness.reason}</div>
        </Section>
      )}

      {recentLogs.length > 0 && (
        <Section>
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
        </Section>
      )}

      {platforms.length > 0 && (
        <Section>
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
        </Section>
      )}
    </div>
  )
}

function PillLabel() {
  const { t } = useI18n()

  return <>{t.shell.statusbar.gateway}</>
}

// ---------------------------------------------------------------------------

const plugin: HermesPlugin = {
  id: 'gateway-pill',
  name: 'Gateway Pill',
  register(ctx) {
    startReadinessPoll()

    // Declarative menu item — the app's own trigger/popover chrome renders it
    // (portal, w-72, side=top), the plugin supplies live innards + the panel.
    ctx.register({
      id: 'pill',
      area: 'statusBar.right',
      order: 90,
      data: {
        icon: <PillIcon />,
        id: 'gateway-pill',
        label: <PillLabel />,
        detail: <PillDetail />,
        menuClassName: 'w-72',
        menuContent: (close: () => void) => <GatewayMenuPanel onClose={close} />,
        variant: 'menu'
      } satisfies StatusbarItem
    })
  }
}

export default plugin
