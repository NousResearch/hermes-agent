import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import type { CommandCenterSection } from '@/app/command-center'
import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import {
  Activity,
  AlertCircle,
  Clock,
  Command,
  Hash,
  Loader2,
  Sparkles,
  Terminal,
  Zap,
  ZapFilled
} from '@/lib/icons'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { contextBarLabel, LiveDuration, usageContextLabel } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { setGlobalYolo, setSessionYolo } from '@/lib/yolo-session'
import {
  $activeSessionId,
  $busy,
  $connection,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $yoloActive,
  setYoloActive
} from '@/store/session'
import { $subagentsBySession, activeSubagentCount, failedSubagentCount } from '@/store/subagents'
import { $gatewayRestarting } from '@/store/system-actions'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateStatus,
  openUpdateOverlayFor
} from '@/store/updates'
import type { StatusResponse } from '@/types/hermes'

import { CRON_ROUTE } from '../../routes'
import type { StatusbarItem, StatusbarSelectModifiers } from '../statusbar-controls'

interface StatusbarItemsOptions {
  agentsOpen: boolean
  chatOpen: boolean
  commandCenterOpen: boolean
  extraLeftItems: readonly StatusbarItem[]
  extraRightItems: readonly StatusbarItem[]
  gatewayLogLines: readonly string[]
  gatewayState: string
  inferenceStatus: RuntimeReadinessResult | null
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  freshDraftReady: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  statusSnapshot: StatusResponse | null
  toggleCommandCenter: () => void
}

export function useStatusbarItems({
  agentsOpen,
  chatOpen,
  commandCenterOpen,
  extraLeftItems,
  extraRightItems,
  gatewayLogLines,
  gatewayState,
  inferenceStatus,
  openAgents,
  openCommandCenterSection,
  freshDraftReady,
  requestGateway,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const activeSessionId = useStore($activeSessionId)
  const terminalTakeover = useStore($terminalTakeover)
  const yoloActive = useStore($yoloActive)
  const busy = useStore($busy)
  const currentUsage = useStore($currentUsage)
  const gatewayRestarting = useStore($gatewayRestarting)
  const sessionStartedAt = useStore($sessionStartedAt)
  const turnStartedAt = useStore($turnStartedAt)
  const subagentsBySession = useStore($subagentsBySession)
  const updateStatus = useStore($updateStatus)
  const updateApply = useStore($updateApply)
  const backendUpdateStatus = useStore($backendUpdateStatus)
  const backendUpdateApply = useStore($backendUpdateApply)
  const desktopVersion = useStore($desktopVersion)
  const connection = useStore($connection)

  const contextUsage = useMemo(() => usageContextLabel(currentUsage), [currentUsage])
  const contextBar = useMemo(() => contextBarLabel(currentUsage), [currentUsage])

  // Per-session approval bypass (same scope as the TUI's Shift+Tab). On a
  // new-chat draft (no runtime session yet) we arm locally; the session-create
  // path applies it once the backend session exists.
  //
  // Shift+click flips the GLOBAL approvals.mode instead — a persistent,
  // all-sessions/CLI/TUI/cron bypass that survives restarts.
  const toggleYolo = useCallback(
    async (modifiers?: StatusbarSelectModifiers) => {
      const next = !$yoloActive.get()

      setYoloActive(next)

      if (modifiers?.shiftKey) {
        try {
          await setGlobalYolo(requestGateway, next)
        } catch {
          setYoloActive(!next)
        }

        return
      }

      const sid = $activeSessionId.get()

      if (!sid) {
        return
      }

      try {
        await setSessionYolo(requestGateway, sid, next)
      } catch {
        setYoloActive(!next)
      }
    },
    [requestGateway]
  )

  const showYoloToggle = gatewayState === 'open' && (!!activeSessionId || freshDraftReady)

  const gatewayMenuContent = useMemo(
    () => (
      <GatewayMenuPanel
        gatewayState={gatewayState}
        inferenceStatus={inferenceStatus}
        logLines={gatewayLogLines}
        onOpenSystem={() => openCommandCenterSection('system')}
        statusSnapshot={statusSnapshot}
      />
    ),
    [gatewayLogLines, gatewayState, inferenceStatus, openCommandCenterSection, statusSnapshot]
  )

  // The indicator must speak the same scope as the Spawn-tree panel it opens:
  // every session's subagents, never background system actions (gateway
  // restarts, toolset installs) which surface in their own panels.
  const { subagentsFailed, subagentsRunning } = useMemo(() => {
    const lists = Object.values(subagentsBySession)

    return {
      subagentsFailed: lists.reduce((sum, items) => sum + failedSubagentCount(items), 0),
      subagentsRunning: lists.reduce((sum, items) => sum + activeSubagentCount(items), 0)
    }
  }, [subagentsBySession])

  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true
  const gatewayDegraded = gatewayOpen || gatewayConnecting

  const gatewayDetail = gatewayOpen
    ? inferenceStatus?.ready
      ? copy.gatewayReady
      : inferenceStatus
        ? copy.gatewayNeedsSetup
        : copy.gatewayChecking
    : gatewayConnecting
      ? copy.gatewayConnecting
      : copy.gatewayOffline

  const gatewayClassName = inferenceReady
    ? undefined
    : gatewayDegraded
      ? 'text-amber-600 hover:text-amber-600'
      : 'text-destructive hover:text-destructive'

  const clientVersionItem = useMemo<StatusbarItem>(() => {
    const appVersion = desktopVersion?.appVersion
    const sha = updateStatus?.currentSha?.slice(0, 7) ?? null
    const behind = updateStatus?.behind ?? 0
    const applying = updateApply.applying || updateApply.stage === 'restart'
    const remote = connection?.mode === 'remote'

    const version = appVersion ? `v${appVersion}` : (sha ?? copy.unknown)
    const base = remote ? copy.clientLabel(appVersion ?? sha ?? copy.unknown) : version
    const behindHint = !applying && behind > 0 ? ` (+${behind})` : ''

    const label = applying
      ? `${base} · ${updateApply.stage === 'restart' ? copy.restart : copy.update}`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? updateApply.message || copy.updateInProgress : null,
      !applying && behind > 0 && copy.commitsBehind(behind, updateStatus?.branch ?? '...'),
      appVersion && copy.desktopVersion(appVersion),
      sha && copy.commit(sha),
      updateStatus?.branch && copy.branch(updateStatus.branch)
    ]
      .filter(Boolean)
      .join(' · ')

    return {
      className: !applying && behind > 0 ? 'text-primary hover:text-primary' : undefined,
      detail: appVersion && sha && !applying && !remote ? sha : undefined,
      hidden: !appVersion && !sha,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-client',
      label,
      onSelect: () => openUpdateOverlayFor('client'),
      title: tooltip || undefined,
      variant: 'action'
    }
  }, [
    desktopVersion?.appVersion,
    connection?.mode,
    copy,
    updateApply.applying,
    updateApply.message,
    updateApply.stage,
    updateStatus?.behind,
    updateStatus?.branch,
    updateStatus?.currentSha
  ])

  const backendVersionItem = useMemo<StatusbarItem | null>(() => {
    if (connection?.mode !== 'remote') {
      return null
    }

    const backendVersion = statusSnapshot?.version
    const behind = backendUpdateStatus?.behind ?? 0
    const applying = backendUpdateApply.applying || backendUpdateApply.stage === 'restart'

    const base = copy.backendLabel(backendVersion ?? copy.unknown)
    const behindHint = !applying && behind > 0 ? ` (+${behind})` : ''

    const label = applying
      ? `${base} · ${backendUpdateApply.stage === 'restart' ? copy.restart : copy.update}`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? backendUpdateApply.message || copy.updateInProgress : null,
      !applying && behind > 0 && copy.commitsBehind(behind, 'main'),
      backendVersion && copy.backendVersion(backendVersion)
    ]
      .filter(Boolean)
      .join(' · ')

    return {
      className: !applying && behind > 0 ? 'text-primary hover:text-primary' : undefined,
      hidden: !backendVersion,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-backend',
      label,
      onSelect: () => openUpdateOverlayFor('backend'),
      title: tooltip || undefined,
      variant: 'action'
    }
  }, [
    connection?.mode,
    statusSnapshot?.version,
    backendUpdateStatus?.behind,
    backendUpdateApply.applying,
    backendUpdateApply.message,
    backendUpdateApply.stage,
    copy
  ])

  const coreLeftStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      {
        className: `w-7 justify-center px-0${commandCenterOpen ? ' bg-accent/55 text-foreground' : ''}`,
        icon: <Command className="size-3.5" />,
        id: 'command-center',
        onSelect: toggleCommandCenter,
        title: commandCenterOpen ? copy.closeCommandCenter : copy.openCommandCenter,
        variant: 'action'
      },
      {
        className: gatewayRestarting ? undefined : gatewayClassName,
        detail: gatewayRestarting ? copy.gatewayRestarting : gatewayDetail,
        icon: gatewayRestarting ? (
          <GlyphSpinner ariaLabel={copy.gatewayRestarting} className="size-3" />
        ) : inferenceReady ? (
          <Activity className="size-3" />
        ) : (
          <AlertCircle className="size-3" />
        ),
        id: 'gateway-health',
        label: copy.gateway,
        menuClassName: 'w-72',
        menuContent: gatewayMenuContent,
        title: inferenceStatus?.reason || copy.gatewayTitle,
        variant: 'menu'
      },
      {
        className: cn(
          agentsOpen && 'bg-accent/55 text-foreground',
          subagentsFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          subagentsRunning > 0
            ? copy.subagents(subagentsRunning)
            : subagentsFailed > 0
              ? copy.failed(subagentsFailed)
              : undefined,
        icon:
          subagentsFailed > 0 ? (
            <AlertCircle className="size-3" />
          ) : subagentsRunning > 0 ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Sparkles className="size-3" />
          ),
        id: 'agents',
        label: copy.agents,
        onSelect: openAgents,
        title: agentsOpen ? copy.closeAgents : copy.openAgents,
        variant: 'action'
      },
      {
        icon: <Clock className="size-3" />,
        id: 'cron',
        label: copy.cron,
        title: copy.openCron,
        to: CRON_ROUTE,
        variant: 'action'
      }
    ],
    [
      agentsOpen,
      commandCenterOpen,
      copy,
      gatewayMenuContent,
      gatewayClassName,
      gatewayDetail,
      gatewayRestarting,
      inferenceReady,
      inferenceStatus?.reason,
      openAgents,
      subagentsFailed,
      subagentsRunning,
      toggleCommandCenter
    ]
  )

  const coreRightStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      {
        detail: <LiveDuration since={turnStartedAt} />,
        hidden: !busy || !turnStartedAt,
        icon: <Loader2 className="size-3 animate-spin" />,
        id: 'running-timer',
        label: copy.turnRunning,
        title: copy.currentTurnElapsed,
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        hidden: !contextUsage,
        id: 'context-usage',
        label: contextUsage,
        title: copy.contextUsage,
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: copy.session,
        title: copy.runtimeSessionElapsed,
        variant: 'text'
      },
      {
        className: cn('px-1', yoloActive && 'bg-(--chrome-action-hover)'),
        hidden: !showYoloToggle,
        icon: yoloActive ? (
          <ZapFilled className="size-3.5 shrink-0" />
        ) : (
          <Zap className="size-3.5 shrink-0 opacity-70" />
        ),
        id: 'yolo',
        onSelect: modifiers => void toggleYolo(modifiers),
        title: yoloActive ? copy.yoloOn : copy.yoloOff,
        variant: 'action'
      },
      {
        className: `w-7 justify-center px-0${terminalTakeover ? ' bg-accent/55 text-foreground' : ''}`,
        hidden: !chatOpen,
        icon: <Terminal className="size-3.5" />,
        id: 'terminal',
        onSelect: () => setTerminalTakeover(!$terminalTakeover.get()),
        title: terminalTakeover ? copy.hideTerminal : copy.showTerminal,
        variant: 'action'
      },
      clientVersionItem,
      ...(backendVersionItem ? [backendVersionItem] : [])
    ],
    [
      busy,
      chatOpen,
      contextBar,
      contextUsage,
      copy,
      sessionStartedAt,
      showYoloToggle,
      terminalTakeover,
      toggleYolo,
      turnStartedAt,
      clientVersionItem,
      backendVersionItem,
      yoloActive
    ]
  )

  const leftStatusbarItems = useMemo(
    () => [...coreLeftStatusbarItems, ...extraLeftItems],
    [coreLeftStatusbarItems, extraLeftItems]
  )

  const statusbarItems = useMemo(
    () => [...extraRightItems, ...coreRightStatusbarItems],
    [coreRightStatusbarItems, extraRightItems]
  )

  return { leftStatusbarItems, statusbarItems }
}
