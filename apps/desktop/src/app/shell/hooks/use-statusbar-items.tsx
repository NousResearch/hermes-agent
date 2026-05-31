import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import type { CommandCenterSection } from '@/app/command-center'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import { Activity, AlertCircle, Clock, Command, Cpu, Hash, Loader2, Sparkles } from '@/lib/icons'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { contextBarLabel, LiveDuration, usageContextLabel } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { $desktopActionTasks } from '@/store/activity'
import { $previewServerRestartStatus } from '@/store/preview'
import {
  $busy,
  $currentModel,
  $currentProvider,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $workingSessionIds,
  setModelPickerOpen
} from '@/store/session'
import { $subagentsBySession, activeSubagentCount } from '@/store/subagents'
import { $desktopVersion, $updateApply, $updateStatus, setUpdateOverlayOpen } from '@/store/updates'
import type { StatusResponse } from '@/types/hermes'

import { CRON_ROUTE } from '../../routes'
import type { StatusbarItem } from '../statusbar-controls'

interface StatusbarItemsOptions {
  agentsOpen: boolean
  commandCenterOpen: boolean
  extraLeftItems: readonly StatusbarItem[]
  extraRightItems: readonly StatusbarItem[]
  gatewayLogLines: readonly string[]
  gatewayState: string
  inferenceStatus: RuntimeReadinessResult | null
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  statusSnapshot: StatusResponse | null
  toggleCommandCenter: () => void
}

export function useStatusbarItems({
  agentsOpen,
  commandCenterOpen,
  extraLeftItems,
  extraRightItems,
  gatewayLogLines,
  gatewayState,
  inferenceStatus,
  openAgents,
  openCommandCenterSection,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const busy = useStore($busy)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const currentUsage = useStore($currentUsage)
  const desktopActionTasks = useStore($desktopActionTasks)
  const previewServerRestartStatus = useStore($previewServerRestartStatus)
  const sessionStartedAt = useStore($sessionStartedAt)
  const turnStartedAt = useStore($turnStartedAt)
  const workingSessionIds = useStore($workingSessionIds)
  const subagentsBySession = useStore($subagentsBySession)
  const updateStatus = useStore($updateStatus)
  const updateApply = useStore($updateApply)
  const desktopVersion = useStore($desktopVersion)

  const contextUsage = useMemo(() => usageContextLabel(currentUsage), [currentUsage])
  const contextBar = useMemo(() => contextBarLabel(currentUsage), [currentUsage])

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

  const { bgFailed, bgRunning, subagentsRunning } = useMemo(() => {
    const actions = Object.values(desktopActionTasks)
    const running = actions.filter(t => t.status.running).length
    const failed = actions.filter(t => !t.status.running && (t.status.exit_code ?? 0) !== 0).length
    const previewRunning = previewServerRestartStatus === 'running' ? 1 : 0
    const previewFailed = previewServerRestartStatus === 'error' ? 1 : 0

    const subagentsRunning = Object.values(subagentsBySession).reduce(
      (sum, items) => sum + activeSubagentCount(items),
      0
    )

    return {
      bgFailed: failed + previewFailed,
      bgRunning: workingSessionIds.length + running + previewRunning,
      subagentsRunning
    }
  }, [desktopActionTasks, previewServerRestartStatus, subagentsBySession, workingSessionIds])

  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true
  const gatewayDegraded = gatewayOpen || gatewayConnecting

  const gatewayDetail = gatewayOpen
    ? inferenceStatus?.ready
      ? 'ready'
      : inferenceStatus
        ? 'needs setup'
        : 'checking'
    : gatewayConnecting
      ? 'connecting'
      : 'offline'

  const gatewayClassName = inferenceReady
    ? undefined
    : gatewayDegraded
      ? 'text-amber-600 hover:text-amber-600'
      : 'text-destructive hover:text-destructive'

  const versionItem = useMemo<StatusbarItem>(() => {
    const appVersion = desktopVersion?.appVersion
    const sha = updateStatus?.currentSha?.slice(0, 7) ?? null
    const behind = updateStatus?.behind ?? 0
    const applying = updateApply.applying || updateApply.stage === 'restart'
    const base = appVersion ? `v${appVersion}` : (sha ?? 'unknown')
    const behindHint = !applying && behind > 0 ? ` (+${behind})` : ''

    const label = applying
      ? updateApply.stage === 'restart'
        ? `${base} · restart`
        : `${base} · update`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? updateApply.message || 'Update in progress' : null,
      !applying && behind > 0 && `${behind} commit${behind === 1 ? '' : 's'} behind ${updateStatus?.branch ?? '…'}`,
      appVersion && `Hermes Desktop v${appVersion}`,
      sha && `commit ${sha}`,
      updateStatus?.branch && `branch ${updateStatus.branch}`
    ]
      .filter(Boolean)
      .join(' · ')

    return {
      className: !applying && behind > 0 ? 'text-primary hover:text-primary' : undefined,
      detail: appVersion && sha && !applying ? sha : undefined,
      hidden: !appVersion && !sha,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version',
      label,
      onSelect: () => setUpdateOverlayOpen(true),
      title: tooltip || undefined,
      variant: 'action'
    }
  }, [
    desktopVersion?.appVersion,
    updateApply.applying,
    updateApply.message,
    updateApply.stage,
    updateStatus?.behind,
    updateStatus?.branch,
    updateStatus?.currentSha
  ])

  const coreLeftStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      {
        className: `w-7 justify-center px-0${commandCenterOpen ? ' bg-accent/55 text-foreground' : ''}`,
        icon: <Command className="size-3.5" />,
        id: 'command-center',
        onSelect: toggleCommandCenter,
        title: commandCenterOpen ? 'Close Command Center' : 'Open Command Center',
        variant: 'action'
      },
      {
        className: gatewayClassName,
        detail: gatewayDetail,
        icon: inferenceReady ? <Activity className="size-3" /> : <AlertCircle className="size-3" />,
        id: 'gateway-health',
        label: 'Gateway',
        menuClassName: 'w-72',
        menuContent: gatewayMenuContent,
        title: inferenceStatus?.reason || 'Hermes inference gateway status',
        variant: 'menu'
      },
      {
        className: cn(
          agentsOpen && 'bg-accent/55 text-foreground',
          bgFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          subagentsRunning > 0
            ? `${subagentsRunning} subagent${subagentsRunning === 1 ? '' : 's'}`
            : bgFailed > 0
              ? `${bgFailed} failed`
              : bgRunning > 0
                ? `${bgRunning} running`
                : undefined,
        icon:
          bgFailed > 0 ? (
            <AlertCircle className="size-3" />
          ) : bgRunning > 0 || subagentsRunning > 0 ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Sparkles className="size-3" />
          ),
        id: 'agents',
        label: 'Agents',
        onSelect: openAgents,
        title: agentsOpen ? 'Close agents' : 'Open agents',
        variant: 'action'
      },
      {
        icon: <Clock className="size-3" />,
        id: 'cron',
        label: 'Cron',
        title: 'Open cron jobs',
        to: CRON_ROUTE,
        variant: 'action'
      }
    ],
    [
      agentsOpen,
      bgFailed,
      bgRunning,
      commandCenterOpen,
      gatewayMenuContent,
      gatewayClassName,
      gatewayDetail,
      inferenceReady,
      inferenceStatus?.reason,
      openAgents,
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
        label: 'Running',
        title: 'Current turn elapsed',
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        hidden: !contextUsage,
        id: 'context-usage',
        label: contextUsage,
        title: 'Context usage',
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: 'Session',
        title: 'Runtime session elapsed',
        variant: 'text'
      },
      {
        detail: currentProvider || '',
        icon: <Cpu className="size-3" />,
        id: 'model-summary',
        label: currentModel || 'No model selected',
        onSelect: () => setModelPickerOpen(true),
        title: currentProvider ? `Switch model · ${currentProvider}: ${currentModel || ''}` : 'Open model picker',
        variant: 'action'
      },
      versionItem
    ],
    [busy, contextBar, contextUsage, currentModel, currentProvider, sessionStartedAt, turnStartedAt, versionItem]
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
