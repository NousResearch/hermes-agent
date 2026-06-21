import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState } from 'react'

import type { CommandCenterSection } from '@/app/command-center'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import { restartGateway } from '@/hermes'
import { Activity, AlertCircle, Command, Cpu, FolderOpen, GitBranch, Loader2, Sparkles } from '@/lib/icons'
import { compactPath, contextBarLabel, LiveDuration, usageContextLabel } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { $desktopActionTasks } from '@/store/activity'
import { $desktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'
import { $previewServerRestartStatus } from '@/store/preview'
import {
  $busy,
  $currentBranch,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $workingSessionIds,
  setModelPickerOpen
} from '@/store/session'
import type { StatusResponse } from '@/types/hermes'

import type { StatusbarItem } from '../statusbar-controls'

interface StatusbarItemsOptions {
  agentsOpen: boolean
  browseSessionCwd: () => Promise<void>
  commandCenterOpen: boolean
  extraLeftItems: readonly StatusbarItem[]
  extraRightItems: readonly StatusbarItem[]
  gatewayLogLines: readonly string[]
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  statusSnapshot: StatusResponse | null
  toggleCommandCenter: () => void
}

export function useStatusbarItems({
  agentsOpen,
  browseSessionCwd,
  commandCenterOpen,
  extraLeftItems,
  extraRightItems,
  gatewayLogLines,
  openAgents,
  openCommandCenterSection,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const busy = useStore($busy)
  const currentBranch = useStore($currentBranch)
  const currentCwd = useStore($currentCwd)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const currentUsage = useStore($currentUsage)
  const desktopActionTasks = useStore($desktopActionTasks)
  const language = useStore($desktopLanguage)
  const previewServerRestartStatus = useStore($previewServerRestartStatus)
  const sessionStartedAt = useStore($sessionStartedAt)
  const turnStartedAt = useStore($turnStartedAt)
  const workingSessionIds = useStore($workingSessionIds)

  const contextUsage = useMemo(() => usageContextLabel(currentUsage), [currentUsage])
  const contextBar = useMemo(() => contextBarLabel(currentUsage), [currentUsage])

  const [restartingGateway, setRestartingGateway] = useState(false)

  const handleRestartGateway = useCallback(async () => {
    if (restartingGateway) {
      return
    }

    setRestartingGateway(true)

    try {
      await restartGateway()
      notify({
        kind: 'success',
        title: language === 'zh' ? '已请求重启网关' : 'Gateway restart requested',
        message: language === 'zh' ? '网关重新连接后状态会自动更新。' : 'Status will update once the gateway reconnects.'
      })
    } catch (err) {
      notifyError(err, language === 'zh' ? '重启网关失败' : 'Failed to restart gateway')
    } finally {
      setRestartingGateway(false)
    }
  }, [language, restartingGateway])

  const gatewayMenuContent = useMemo(
    () => (
      <GatewayMenuPanel
        logLines={gatewayLogLines}
        onOpenSystem={() => openCommandCenterSection('system')}
        onRestart={() => void handleRestartGateway()}
        restarting={restartingGateway}
        statusSnapshot={statusSnapshot}
        language={language}
      />
    ),
    [gatewayLogLines, handleRestartGateway, language, openCommandCenterSection, restartingGateway, statusSnapshot]
  )

  const { bgFailed, bgRunning } = useMemo(() => {
    const actions = Object.values(desktopActionTasks)
    const running = actions.filter(t => t.status.running).length
    const failed = actions.filter(t => !t.status.running && (t.status.exit_code ?? 0) !== 0).length
    const previewRunning = previewServerRestartStatus === 'running' ? 1 : 0
    const previewFailed = previewServerRestartStatus === 'error' ? 1 : 0

    return { bgFailed: failed + previewFailed, bgRunning: workingSessionIds.length + running + previewRunning }
  }, [desktopActionTasks, previewServerRestartStatus, workingSessionIds])

  const gatewayUp = Boolean(statusSnapshot?.gateway_running)

  const coreLeftStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      {
        className: `h-6 w-6 justify-center px-0${commandCenterOpen ? ' bg-accent/55 text-foreground' : ''}`,
        icon: <Command className="size-3.5" />,
        id: 'command-center',
        onSelect: toggleCommandCenter,
        title: language === 'zh' ? (commandCenterOpen ? '关闭命令中心' : '打开命令中心') : commandCenterOpen ? 'Close Command Center' : 'Open Command Center',
        variant: 'action'
      },
      {
        className: gatewayUp ? undefined : 'text-destructive hover:text-destructive',
        detail: gatewayUp
          ? language === 'zh'
            ? statusSnapshot?.gateway_state === 'online'
              ? '在线'
              : statusSnapshot?.gateway_state || '在线'
            : statusSnapshot?.gateway_state || 'online'
          : language === 'zh'
            ? '离线'
            : 'offline',
        icon: gatewayUp ? <Activity className="size-3" /> : <AlertCircle className="size-3" />,
        id: 'gateway-health',
        label: language === 'zh' ? '网关' : 'Gateway',
        menuClassName: 'w-72',
        menuContent: gatewayMenuContent,
        title: language === 'zh' ? '网关与平台状态' : 'Gateway and platform health',
        variant: 'menu'
      },
      {
        className: cn(
          agentsOpen && 'bg-accent/55 text-foreground',
          bgFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          bgFailed > 0
            ? language === 'zh'
              ? `${bgFailed} 个失败`
              : `${bgFailed} failed`
            : bgRunning > 0
              ? language === 'zh'
                ? `${bgRunning} 个运行中`
                : `${bgRunning} running`
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
        label: language === 'zh' ? 'Agent' : 'Agents',
        onSelect: openAgents,
        title: language === 'zh' ? (agentsOpen ? '关闭 Agent 面板' : '打开 Agent 面板') : agentsOpen ? 'Close agents' : 'Open agents',
        variant: 'action'
      }
    ],
    [
      agentsOpen,
      bgFailed,
      bgRunning,
      commandCenterOpen,
      gatewayMenuContent,
      gatewayUp,
      language,
      openAgents,
      statusSnapshot?.gateway_state,
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
        label: language === 'zh' ? '运行中' : 'Running',
        title: language === 'zh' ? '当前轮次耗时' : 'Current turn elapsed',
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        hidden: !contextUsage,
        id: 'context-usage',
        label: contextUsage,
        title: language === 'zh' ? '上下文用量' : 'Context usage',
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: language === 'zh' ? '会话' : 'Session',
        title: language === 'zh' ? '运行会话耗时' : 'Runtime session elapsed',
        variant: 'text'
      },
      {
        detail: currentProvider || '',
        icon: <Cpu className="size-3" />,
        id: 'model-summary',
        label: currentModel || (language === 'zh' ? '未选择模型' : 'No model selected'),
        onSelect: () => setModelPickerOpen(true),
        title: currentProvider
          ? language === 'zh'
            ? `切换模型 · ${currentProvider}: ${currentModel || ''}`
            : `Switch model · ${currentProvider}: ${currentModel || ''}`
          : language === 'zh'
            ? '打开模型选择器'
            : 'Open model picker',
        variant: 'action'
      },
      {
        icon: <FolderOpen className="size-3" />,
        id: 'cwd',
        label: currentCwd ? compactPath(currentCwd) : language === 'zh' ? '未选择工作目录' : 'No project cwd',
        onSelect: () => void browseSessionCwd(),
        title: currentCwd
          ? language === 'zh'
            ? `更改工作目录 · ${currentCwd}`
            : `Change working directory · ${currentCwd}`
          : language === 'zh'
            ? '选择工作目录'
            : 'Choose working directory',
        variant: 'action'
      },
      {
        hidden: !currentBranch,
        icon: <GitBranch className="size-3" />,
        id: 'branch',
        label: currentBranch,
        title: currentBranch ? (language === 'zh' ? `当前分支：${currentBranch}` : `Current branch: ${currentBranch}`) : undefined,
        variant: 'text'
      }
    ],
    [
      browseSessionCwd,
      busy,
      contextBar,
      contextUsage,
      currentBranch,
      currentCwd,
      currentModel,
      currentProvider,
      language,
      sessionStartedAt,
      turnStartedAt
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
