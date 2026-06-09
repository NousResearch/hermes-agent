import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useRef } from 'react'

import type { CommandCenterSection } from '@/app/command-center'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import {
  Activity,
  AlertCircle,
  ChevronDown,
  Clock,
  Command,
  Hash,
  Loader2,
  Sparkles,
  Zap,
  ZapFilled
} from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { contextBarLabel, LiveDuration, usageContextLabel } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { getGlobalApprovalMode, setGlobalApprovalMode, setSessionYolo } from '@/lib/yolo-session'
import { $desktopActionTasks } from '@/store/activity'
import { $previewServerRestartStatus } from '@/store/preview'
import {
  $activeSessionId,
  $approvalMode,
  $busy,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $workingSessionIds,
  $yoloActive,
  setApprovalMode,
  setModelPickerOpen,
  setYoloActive
} from '@/store/session'
import { $subagentsBySession, activeSubagentCount } from '@/store/subagents'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateStatus,
  openUpdateOverlayFor
} from '@/store/updates'
import type { ApprovalMode, StatusResponse } from '@/types/hermes'

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
  modelMenuContent?: ReactNode
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  freshDraftReady: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
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
  modelMenuContent,
  openAgents,
  openCommandCenterSection,
  requestGateway,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const approvalMode = useStore($approvalMode)
  const yoloActive = useStore($yoloActive)
  const busy = useStore($busy)
  const currentFastMode = useStore($currentFastMode)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const currentReasoningEffort = useStore($currentReasoningEffort)
  const currentUsage = useStore($currentUsage)
  const desktopActionTasks = useStore($desktopActionTasks)
  const previewServerRestartStatus = useStore($previewServerRestartStatus)
  const sessionStartedAt = useStore($sessionStartedAt)
  const turnStartedAt = useStore($turnStartedAt)
  const workingSessionIds = useStore($workingSessionIds)
  const subagentsBySession = useStore($subagentsBySession)
  const updateStatus = useStore($updateStatus)
  const updateApply = useStore($updateApply)
  const backendUpdateStatus = useStore($backendUpdateStatus)
  const backendUpdateApply = useStore($backendUpdateApply)
  const desktopVersion = useStore($desktopVersion)
  const contextUsage = useMemo(() => usageContextLabel(currentUsage), [currentUsage])
  const contextBar = useMemo(() => contextBarLabel(currentUsage), [currentUsage])
  const approvalModeRevision = useRef(0)

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    const revision = ++approvalModeRevision.current

    void getGlobalApprovalMode(requestGateway)
      .then(mode => {
        if (revision === approvalModeRevision.current) {
          setApprovalMode(mode)
        }
      })
      .catch(() => undefined)
  }, [gatewayState, requestGateway])

  const applyApprovalMode = useCallback(
    async (next: ApprovalMode) => {
      const revision = ++approvalModeRevision.current
      const previousMode = $approvalMode.get()
      const previousYolo = $yoloActive.get()
      const sid = $activeSessionId.get()

      setApprovalMode(next)
      setYoloActive(next === 'off')

      try {
        const activeMode = await setGlobalApprovalMode(requestGateway, next)

        if (revision !== approvalModeRevision.current) {
          return
        }

        setApprovalMode(activeMode)
        setYoloActive(activeMode === 'off')

        if (activeMode !== 'off' && previousYolo && sid) {
          await setSessionYolo(requestGateway, sid, false).catch(() => undefined)
        }
      } catch {
        if (revision === approvalModeRevision.current) {
          setApprovalMode(previousMode)
          setYoloActive(previousYolo)
        }
      }
    },
    [requestGateway]
  )

  const showApprovalMenu = gatewayState === 'open'
  const fullAccessActive = yoloActive || approvalMode === 'off'
  const approvalMenuValue: ApprovalMode = fullAccessActive ? 'off' : approvalMode
  const approvalStatusLabel =
    approvalMenuValue === 'off'
      ? copy.approvalFull
      : approvalMenuValue === 'smart'
        ? copy.approvalSmart
        : copy.approvalAsk

  const approvalMenuContent = useMemo(
    () => (
      <div className="w-80 py-1">
        <DropdownMenuLabel className="px-2.5 py-1 text-xs font-medium text-foreground">
          {copy.approvalMenuTitle}
        </DropdownMenuLabel>
        <DropdownMenuSeparator className="mx-0" />
        <DropdownMenuRadioGroup
          value={approvalMenuValue}
          onValueChange={value => {
            void applyApprovalMode(value as ApprovalMode)
          }}
        >
          <ApprovalModeItem description={copy.approvalAskDescription} label={copy.approvalAsk} value="manual" />
          <ApprovalModeItem description={copy.approvalSmartDescription} label={copy.approvalSmart} value="smart" />
          <ApprovalModeItem description={copy.approvalFullDescription} label={copy.approvalFull} value="off" />
        </DropdownMenuRadioGroup>
      </div>
    ),
    [approvalMenuValue, applyApprovalMode, copy]
  )

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

  const { bgFailed, bgRunning } = useMemo(() => {
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
      bgRunning: workingSessionIds.length + running + previewRunning + subagentsRunning
    }
  }, [desktopActionTasks, previewServerRestartStatus, subagentsBySession, workingSessionIds])

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

    const version = appVersion ?? sha ?? copy.unknown
    const base = copy.clientLabel(version)
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
      detail: appVersion && sha && !applying ? sha : undefined,
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
    copy,
    updateApply.applying,
    updateApply.message,
    updateApply.stage,
    updateStatus?.behind,
    updateStatus?.branch,
    updateStatus?.currentSha
  ])

  const backendVersionItem = useMemo<StatusbarItem | null>(() => {
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
        className: gatewayClassName,
        detail: gatewayDetail,
        icon: inferenceReady ? <Activity className="size-3" /> : <AlertCircle className="size-3" />,
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
          bgFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          bgFailed > 0
            ? copy.failed(bgFailed)
            : bgRunning > 0
              ? copy.running(bgRunning)
              : undefined,
        icon:
          bgFailed > 0 ? (
            <AlertCircle className="size-3" />
          ) : bgRunning > 0 ? (
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
      bgFailed,
      bgRunning,
      commandCenterOpen,
      copy,
      gatewayMenuContent,
      gatewayClassName,
      gatewayDetail,
      inferenceReady,
      inferenceStatus?.reason,
      openAgents,
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
        id: 'model-summary',
        label: (
          <span className="inline-flex min-w-0 items-center gap-0.5">
            <span className="truncate">
              {formatModelStatusLabel(currentModel, {
                fastMode: currentFastMode,
                reasoningEffort: currentReasoningEffort
              })}
            </span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </span>
        ),
        ...(modelMenuContent
          ? {
              menuAlign: 'end' as const,
              menuClassName: 'w-64',
              menuContent: modelMenuContent,
              title: currentProvider
                ? copy.modelTitle(currentProvider, currentModel || copy.modelNone)
                : copy.switchModel,
              variant: 'menu' as const
            }
          : {
              onSelect: () => setModelPickerOpen(true),
              title: currentProvider
                ? copy.providerModelTitle(currentProvider, currentModel || copy.noModel)
                : copy.openModelPicker,
              variant: 'action' as const
            })
      },
      {
        className: cn(fullAccessActive && 'bg-(--chrome-action-hover)'),
        hidden: !showApprovalMenu,
        icon: fullAccessActive ? (
          <ZapFilled className="size-3.5 shrink-0" />
        ) : (
          <Zap className="size-3.5 shrink-0 opacity-70" />
        ),
        id: 'yolo',
        label: approvalStatusLabel,
        menuAlign: 'end',
        menuClassName: 'w-80',
        menuContent: approvalMenuContent,
        title: `${copy.approvalMenuTitle} ${approvalStatusLabel}`,
        variant: 'menu'
      },
      clientVersionItem,
      ...(backendVersionItem ? [backendVersionItem] : [])
    ],
    [
      busy,
      contextBar,
      contextUsage,
      copy,
      currentFastMode,
      currentModel,
      currentProvider,
      currentReasoningEffort,
      modelMenuContent,
      sessionStartedAt,
      showApprovalMenu,
      turnStartedAt,
      clientVersionItem,
      backendVersionItem,
      fullAccessActive,
      approvalStatusLabel,
      approvalMenuContent
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

function ApprovalModeItem({ description, label, value }: { description: string; label: string; value: string }) {
  return (
    <DropdownMenuRadioItem className="items-start gap-2 rounded-none px-2.5 py-1.5" value={value}>
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="truncate text-xs text-foreground">{label}</span>
        <span className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{description}</span>
      </span>
    </DropdownMenuRadioItem>
  )
}
