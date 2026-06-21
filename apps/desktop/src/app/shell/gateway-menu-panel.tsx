import { IconLayoutDashboard } from '@tabler/icons-react'

import { StatusDot, type StatusTone } from '@/components/status-dot'
import { Button } from '@/components/ui/button'
import { Activity, AlertCircle, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { DesktopLanguage } from '@/store/language'
import type { StatusResponse } from '@/types/hermes'

interface GatewayMenuPanelProps {
  logLines: readonly string[]
  onOpenSystem: () => void
  onRestart: () => void
  restarting: boolean
  statusSnapshot: StatusResponse | null
  language: DesktopLanguage
}

const PLATFORM_TONE: Record<string, StatusTone> = {
  connected: 'good',
  connecting: 'warn',
  retrying: 'warn',
  pending_restart: 'warn',
  startup_failed: 'bad',
  fatal: 'bad'
}

const STATE_ZH: Record<string, string> = {
  connected: '已连接',
  connecting: '连接中',
  fatal: '严重错误',
  offline: '离线',
  online: '在线',
  pending_restart: '等待重启',
  retrying: '重试中',
  startup_failed: '启动失败'
}

const prettyState = (state: string, language: DesktopLanguage) =>
  language === 'zh' ? (STATE_ZH[state] ?? state.replace(/_/g, ' ')) : state.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

// Strip leading "YYYY-MM-DD HH:MM:SS,mmm " and "[runtime_id] " prefixes from
// log lines so they don't dominate the display. Full text preserved on hover.
const TIMESTAMP_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+/
const RUNTIME_BRACKET_RE = /^\[[^\]]+]\s+/
const trimLogLine = (raw: string) => raw.trim().replace(TIMESTAMP_RE, '').replace(RUNTIME_BRACKET_RE, '')

export function GatewayMenuPanel({
  logLines,
  onOpenSystem,
  onRestart,
  restarting,
  statusSnapshot,
  language
}: GatewayMenuPanelProps) {
  const gatewayRunning = Boolean(statusSnapshot?.gateway_running)
  const platforms = Object.entries(statusSnapshot?.gateway_platforms || {}).sort(([l], [r]) => l.localeCompare(r))
  const stateLabel = gatewayRunning ? prettyState(statusSnapshot?.gateway_state || 'online', language) : language === 'zh' ? '离线' : 'Offline'
  const recentLogs = logLines.slice(-5)

  return (
    <div className="text-sm">
      <div className="flex items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          {gatewayRunning ? (
            <Activity className="size-3.5 text-primary" />
          ) : (
            <AlertCircle className="size-3.5 text-destructive" />
          )}
          <span className="font-medium">{language === 'zh' ? '网关' : 'Gateway'}</span>
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <StatusDot tone={gatewayRunning ? 'good' : 'bad'} />
            {stateLabel}
          </span>
        </div>
        <div className="flex items-center">
          <Button
            aria-label={language === 'zh' ? (restarting ? '正在重启网关' : '重启网关') : restarting ? 'Restarting gateway' : 'Restart gateway'}
            className="size-7 text-muted-foreground hover:text-foreground"
            disabled={restarting}
            onClick={onRestart}
            size="icon-sm"
            title={language === 'zh' ? (restarting ? '正在重启网关' : '重启网关') : restarting ? 'Restarting gateway' : 'Restart gateway'}
            variant="ghost"
          >
            <RefreshCw className={cn(restarting && 'animate-spin')} />
          </Button>
          <Button
            aria-label={language === 'zh' ? '打开系统面板' : 'Open system panel'}
            className="size-7 text-muted-foreground hover:text-foreground"
            onClick={onOpenSystem}
            size="icon-sm"
            title={language === 'zh' ? '打开系统面板' : 'Open system panel'}
            variant="ghost"
          >
            <IconLayoutDashboard />
          </Button>
        </div>
      </div>

      {recentLogs.length > 0 && (
        <div className="border-t border-border/50 px-3 py-2">
          <SectionLabel>{language === 'zh' ? '最近活动' : 'Recent activity'}</SectionLabel>
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
            {language === 'zh' ? '查看全部日志 →' : 'View all logs →'}
          </button>
        </div>
      )}

      {platforms.length > 0 && (
        <div className="border-t border-border/50 px-3 py-2">
          <SectionLabel>{language === 'zh' ? '平台' : 'Platforms'}</SectionLabel>
          <ul className="mt-1.5 space-y-1">
            {platforms.map(([name, platform]) => (
              <li className="flex items-center justify-between gap-2 text-xs" key={name}>
                <span className="truncate capitalize">{name}</span>
                <span className="flex items-center gap-1.5 text-[0.66rem] text-muted-foreground">
                  <StatusDot tone={PLATFORM_TONE[platform.state] || 'muted'} />
                  {prettyState(platform.state, language)}
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
