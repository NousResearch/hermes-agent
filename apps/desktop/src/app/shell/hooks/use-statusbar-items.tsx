import { useStore } from '@nanostores/react'
import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

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
  const { t } = useTranslation()
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
      ? t('statusbar.ready')
      : inferenceStatus
        ? t('statusbar.needs_setup')
        : t('statusbar.checking')
    : gatewayConnecting
      ? t('statusbar.connecting')
      : t('statusbar.offline')

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
        ? `${base} · ${t('statusbar.restart')}`
        : `${base} · ${t('statusbar.update')}`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? updateApply.message || t('statusbar.update_in_progress') : null,
      !applying && behind > 0 && t('statusbar.commits_behind', { count: behind, plural: behind === 1 ? '' : 's', branch: updateStatus?.branch ?? '…' }),
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
        title: commandCenterOpen ? t('statusbar.close_command_center') : t('statusbar.open_command_center'),
        variant: 'action'
      },
      {
        className: gatewayClassName,
        detail: gatewayDetail,
        icon: inferenceReady ? <Activity className="size-3" /> : <AlertCircle className="size-3" />,
        id: 'gateway-health',
        label: t('statusbar.gateway'),
        menuClassName: 'w-72',
        menuContent: gatewayMenuContent,
        title: inferenceStatus?.reason || t('statusbar.gateway_status'),
        variant: 'menu'
      },
      {
        className: cn(
          agentsOpen && 'bg-accent/55 text-foreground',
          bgFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          subagentsRunning > 0
            ? `${subagentsRunning} ${t('statusbar.agents')}`
            : bgFailed > 0
              ? `${bgFailed} ${t('statusbar.failed')}`
              : bgRunning > 0
                ? `${bgRunning} ${t('statusbar.running')}`
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
        label: t('statusbar.agents'),
        onSelect: openAgents,
        title: agentsOpen ? t('statusbar.close_agents') : t('statusbar.open_agents'),
        variant: 'action'
      },
      {
        icon: <Clock className="size-3" />,
        id: 'cron',
        label: t('statusbar.cron'),
        title: t('statusbar.open_cron_jobs'),
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
        label: t('statusbar.running'),
        title: t('statusbar.current_turn_elapsed'),
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        hidden: !contextUsage,
        id: 'context-usage',
        label: contextUsage,
        title: t('statusbar.context_usage'),
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: t('statusbar.session'),
        title: t('statusbar.session_elapsed'),
        variant: 'text'
      },
      {
        detail: currentProvider || '',
        icon: <Cpu className="size-3" />,
        id: 'model-summary',
        label: currentModel || t('statusbar.no_model_selected'),
        onSelect: () => setModelPickerOpen(true),
        title: currentProvider ? `${t('statusbar.switch_model')} · ${currentProvider}: ${currentModel || ''}` : t('statusbar.open_model_picker'),
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
