import { useStore } from '@nanostores/react'
import { useMemo } from 'react'
import { useNavigate } from 'react-router'

import { ConnectionSwitcher } from '@/app/chat/sidebar/connection-switcher'
import type { CommandCenterSection } from '@/app/command-center'
import { useApprovalModeStatusbarItem } from '@/app/shell/approval-mode-menu'
import { ContextUsagePanel } from '@/app/shell/context-usage-panel'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import { useContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
import { useSystemResourcesStatusbarItem } from '@/app/shell/system-resources-statusbar'
import { $paneVisible, togglePaneVisible } from '@/components/pane-shell/tree/store'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { displayPath, pathLeaf } from '@/lib/display-path'
import {
  Activity,
  AlertCircle,
  Clock,
  Command,
  FolderOpen,
  Globe,
  Hash,
  Layers3,
  Loader2,
  Terminal,
  Zap
} from '@/lib/icons'
import { runtimeReadinessDisplay, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { cacheHitLabel, contextBarLabel, LiveDuration, tokensPerSecondLabel, usageContextLabel } from '@/lib/statusbar'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { resolveVersionStatus } from '@/lib/version-status'
import { copyFilePath, revealFile } from '@/store/file-actions'
import { revealFileInTree } from '@/store/layout'
import { $activeGatewayProfile } from '@/store/profile'
import { $projectTree, projectNameForCwd } from '@/store/projects'
import {
  $activeSessionId,
  $busy,
  $connection,
  $currentCwd,
  $currentUsage,
  $selectedStoredSessionId,
  $sessions,
  $sessionStartedAt,
  $turnStartedAt,
  idsShareLineage,
  sessionMatchesStoredId
} from '@/store/session'
import { $focusedRuntimeId, $focusedSessionState, $focusedStoredSessionId } from '@/store/session-states'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
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
import type { StatusResponse, UsageStats } from '@/types/hermes'

import { CRON_ROUTE, SETTINGS_ROUTE, WEBHOOKS_ROUTE } from '../../routes'
import type { StatusbarItem } from '../statusbar-controls'

const EMPTY_USAGE: UsageStats = { calls: 0, input: 0, output: 0, total: 0 }

interface StatusbarItemsOptions {
  agentsOpen: boolean
  chatOpen: boolean
  commandCenterOpen: boolean
  extraLeftItems: readonly StatusbarItem[]
  extraRightItems: readonly StatusbarItem[]
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
  gatewayState,
  inferenceStatus,
  openAgents,
  openCommandCenterSection,
  requestGateway,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const fileMenu = t.fileMenu
  const primaryActiveSessionId = useStore($activeSessionId)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  // What the button paints and flips is whether the terminal is ON SCREEN —
  // the takeover store alone stays true behind a stacked sibling tab or a
  // minimized zone, which lit the button for a pane the user couldn't see.
  const terminalShowing = useStore($paneVisible('terminal'))
  const sessionsShowing = useStore($paneVisible('sessions'))
  const botsShowing = useStore($paneVisible('hermes-bots:pane'))
  const primaryBusy = useStore($busy)
  // Draft / primary composer atom — used only while the focused surface is the
  // primary (or a draft with no runtime slice yet). A focused TILE keeps its
  // own cwd in `$sessionStates` and must not paint the primary's workspace.
  const primaryCwd = useStore($currentCwd)
  const primaryUsage = useStore($currentUsage)
  const gatewayRestarting = useStore($gatewayRestarting)
  const primarySessionStartedAt = useStore($sessionStartedAt)
  const primaryTurnStartedAt = useStore($turnStartedAt)

  // The indicator must speak the same scope as the Spawn-tree panel it opens:
  // every session's subagents, never background system actions. Only two
  // COUNTS are read, so select scalars — a whole-map `useStore` re-ran this
  // hook (rebuilding all ~9 statusbar items) on every subagent progress tick
  // in ANY session, including background ones.
  const subagentsRunning = useStoreSelector($subagentsBySession, bySession =>
    Object.values(bySession).reduce((sum, items) => sum + activeSubagentCount(items), 0)
  )

  const subagentsFailed = useStoreSelector($subagentsBySession, bySession =>
    Object.values(bySession).reduce((sum, items) => sum + failedSubagentCount(items), 0)
  )

  const updateStatus = useStore($updateStatus)
  const updateApply = useStore($updateApply)
  const backendUpdateStatus = useStore($backendUpdateStatus)
  const backendUpdateApply = useStore($backendUpdateApply)
  const desktopVersion = useStore($desktopVersion)
  const connection = useStore($connection)

  // The FOCUSED session (interacted tile, else the primary — the same
  // derivation the titlebar title follows): every session-scoped readout
  // below (workspace cwd, context count, timers, busy pulse) tracks it, so
  // clicking into a tile makes the statusbar describe THAT session.
  const focusedStoredSessionId = useStore($focusedStoredSessionId)
  const focusedRuntimeId = useStore($focusedRuntimeId)
  // `$focusedSessionState` is a projection of `$sessionStates`, which is
  // republished on EVERY message delta — tens of times a second during a turn.
  // Only the fields read here are selected, so an unchanged readout bails out
  // instead of rebuilding all ~9 statusbar items per token.
  const focusedBusy = useStoreSelector($focusedSessionState, state => Boolean(state?.busy))
  const focusedTurnStartedAt = useStoreSelector($focusedSessionState, state => state?.turnStartedAt ?? null)
  // `usage` is an object, so it can't be compared as a scalar. It IS however
  // replaced wholesale rather than mutated, and only changes when the backend
  // reports new usage — far rarer than a delta — so its reference is a valid
  // bail-out key on its own.
  const focusedUsage = useStoreSelector($focusedSessionState, state => state?.usage ?? null)
  const focusedStateCwd = useStoreSelector($focusedSessionState, state => state?.cwd?.trim() || '')

  // Runtime slices carry the stored id they were bound for. During a primary
  // tab switch the runtime id can lag a frame behind the new selection — the
  // slice still describes the PREVIOUS chat. Gate live cwd on ownership so we
  // never paint session A's workspace while the tab already shows session B.
  const focusedStateStoredId = useStoreSelector($focusedSessionState, state => state?.storedSessionId?.trim() || null)

  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const primaryFocused = !focusedStoredSessionId || focusedStoredSessionId === selectedStoredSessionId

  const activeSessionId = primaryFocused ? primaryActiveSessionId : (focusedRuntimeId ?? null)
  const busy = primaryFocused ? primaryBusy : focusedBusy

  // EMPTY_USAGE (module constant) keeps the fallback referentially stable —
  // a fresh `{...}` each render would bust the usage-label memos below.
  const currentUsage = primaryFocused ? primaryUsage : (focusedUsage ?? EMPTY_USAGE)

  const turnStartedAt = primaryFocused ? primaryTurnStartedAt : focusedTurnStartedAt

  // A tile's session-start + cold cwd come from its stored row (the cache only
  // knows runtime state). Only these scalars are read off `$sessions`, so
  // select them — a whole-list `useStore` re-ran the hook on every session-list
  // write (title updates, poll refreshes, archives).
  const focusedRowStartedAt = useStoreSelector($sessions, sessions =>
    focusedStoredSessionId
      ? (sessions.find(s => sessionMatchesStoredId(s, focusedStoredSessionId))?.started_at ?? null)
      : null
  )

  const focusedRowCwd = useStoreSelector($sessions, sessions => {
    if (!focusedStoredSessionId) {
      return ''
    }

    const row = sessions.find(s => sessionMatchesStoredId(s, focusedStoredSessionId))

    return row?.cwd?.trim() || ''
  })

  // Live runtime cwd is authoritative once it belongs to the focused chat
  // (agent can relocate mid-turn). Until then — cold tabs, mid-switch lag —
  // the stored session row is the selection's project. Primary drafts fall
  // through to `$currentCwd`. A focused TILE must never inherit the primary's
  // workspace — an empty tile cwd stays empty rather than lying about another
  // project's path.
  //
  // Lineage match is a pure derivation of ($sessions + the two ids). Select it
  // so a session-list write only re-renders when the answer actually flips —
  // not on every title/archive refresh of an unrelated row.
  const liveCwdSharesFocusLineage = useStoreSelector($sessions, sessions => {
    if (!focusedStoredSessionId || !focusedStateStoredId) {
      return false
    }

    return idsShareLineage(focusedStoredSessionId, focusedStateStoredId, sessions)
  })

  const liveCwdBelongsToFocus =
    Boolean(focusedStateCwd) &&
    (!focusedStoredSessionId ||
      !focusedStateStoredId ||
      focusedStateStoredId === focusedStoredSessionId ||
      liveCwdSharesFocusLineage)

  const currentCwd = (
    (liveCwdBelongsToFocus ? focusedStateCwd : '') ||
    focusedRowCwd ||
    (primaryFocused ? primaryCwd : '') ||
    ''
  ).trim()

  // Derive the workspace's project name from the already-cached project tree
  // (backend truth via projects.*), so the status item labels by project without
  // a second per-session copy of the same fact. Re-derives whenever the cwd or
  // the tree changes; null (no named project) falls back to the cwd leaf below.
  const projectTree = useStore($projectTree)
  const projectName = useMemo(() => projectNameForCwd(currentCwd), [currentCwd, projectTree])

  const sessionStartedAt = primaryFocused
    ? primarySessionStartedAt
    : focusedRowStartedAt
      ? focusedRowStartedAt * 1000
      : null

  // The backend only knows a session's MEASURED occupancy once a turn has run
  // in this process, so a resumed conversation reports none and the gauge had
  // nothing to paint — turning it on looked like it did nothing until you sent
  // a message. Estimate from the live prompt + transcript instead, on the same
  // read-only RPC the popover uses, so the readout is right the moment it's on
  // screen. Gated on the gauge being shown — the bar itself is unmounted while
  // toggled off, so this covers the rest.
  const contextItemHidden = useStore($statusbarHiddenIds).includes('context-usage')

  const { breakdown: contextBreakdown, loading: contextBreakdownLoading } = useContextBreakdown({
    busy,
    enabled: !contextItemHidden,
    requestGateway,
    sessionId: activeSessionId
  })

  // The breakdown wins whenever we have one, for two reasons: it reports the
  // MEASURED occupancy once the backend has it (falling back to the estimate
  // only before that), and it is keyed to the session it describes. The global
  // `$currentUsage` is neither — a resumed session reports no context fields,
  // and the store merges rather than replaces, so the PREVIOUS session's gauge
  // numbers survive the switch. Mid-turn there's no breakdown by design and
  // the streamed usage carries the gauge.
  const gaugeUsage = useMemo<UsageStats>(
    () =>
      contextBreakdown
        ? {
            ...currentUsage,
            context_max: contextBreakdown.context_max,
            context_percent: contextBreakdown.context_percent,
            context_used: contextBreakdown.context_used
          }
        : currentUsage,
    [contextBreakdown, currentUsage]
  )

  const contextUsage = useMemo(() => usageContextLabel(gaugeUsage), [gaugeUsage])
  const contextBar = useMemo(() => contextBarLabel(gaugeUsage), [gaugeUsage])
  // Both ride the same usage payload the context meter does (session.usage
  // ticks mid-turn, message.complete after) — no extra RPC, no polling.
  const cacheHit = cacheHitLabel(currentUsage)
  const tokensPerSecond = tokensPerSecondLabel(currentUsage)

  const approvalModeItem = useApprovalModeStatusbarItem(activeGatewayProfile, requestGateway)
  const systemResourcesItem = useSystemResourcesStatusbarItem()

  const gatewayMenuContent = useMemo(
    () => (close: () => void) => (
      <GatewayMenuPanel
        gatewayState={gatewayState}
        inferenceStatus={inferenceStatus}
        onClose={close}
        onOpenSystem={() => openCommandCenterSection('system')}
        statusSnapshot={statusSnapshot}
      />
    ),
    [gatewayState, inferenceStatus, openCommandCenterSection, statusSnapshot]
  )

  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true
  const gatewayDegraded = gatewayOpen || gatewayConnecting
  const readinessDisplay = runtimeReadinessDisplay(inferenceStatus)

  const gatewayDetail = gatewayOpen
    ? {
        checking: copy.gatewayChecking,
        needs_setup: copy.gatewayNeedsSetup,
        ready: copy.gatewayReady,
        unavailable: copy.gatewayUnavailable
      }[readinessDisplay]
    : gatewayConnecting
      ? copy.gatewayConnecting
      : copy.gatewayOffline

  const gatewayClassName = inferenceReady
    ? undefined
    : gatewayDegraded
      ? 'text-amber-600 hover:text-amber-600'
      : 'text-destructive hover:text-destructive'

  const clientVersionItem = useMemo<StatusbarItem>(() => {
    const applying = updateApply.applying || updateApply.stage === 'restart'

    const status = resolveVersionStatus({
      applying,
      applyMessage: updateApply.message,
      behind: updateStatus?.behind ?? 0,
      branch: updateStatus?.branch,
      copy,
      remote: connection?.mode === 'remote',
      restarting: updateApply.stage === 'restart',
      sha: updateStatus?.currentSha?.slice(0, 7) ?? null,
      target: 'client',
      updateAvailable: updateStatus?.updateAvailable,
      version: desktopVersion?.appVersion
    })

    return {
      className: status.hasUpdate ? 'text-primary hover:text-primary' : undefined,
      detail: status.detail,
      hidden: status.unknown,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-client',
      label: status.label,
      // Update state is not a preference: hiding it is how a user misses that
      // their client is behind. Listed in the menu, but locked on.
      lockedVisible: true,
      onSelect: () => openUpdateOverlayFor('client'),
      title: status.tooltip,
      toggleLabel: copy.toggleVersion,
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
    updateStatus?.currentSha,
    updateStatus?.updateAvailable
  ])

  const backendVersionItem = useMemo<StatusbarItem | null>(() => {
    if (connection?.mode !== 'remote') {
      return null
    }

    const applying = backendUpdateApply.applying || backendUpdateApply.stage === 'restart'

    const status = resolveVersionStatus({
      applying,
      applyMessage: backendUpdateApply.message,
      behind: backendUpdateStatus?.behind ?? 0,
      copy,
      remote: true,
      restarting: backendUpdateApply.stage === 'restart',
      target: 'backend',
      updateAvailable: backendUpdateStatus?.updateAvailable,
      version: statusSnapshot?.version
    })

    return {
      className: status.hasUpdate ? 'text-primary hover:text-primary' : undefined,
      hidden: status.unknown,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-backend',
      label: status.label,
      lockedVisible: true,
      onSelect: () => openUpdateOverlayFor('backend'),
      title: status.tooltip,
      toggleLabel: copy.toggleBackendVersion,
      variant: 'action'
    }
  }, [
    connection?.mode,
    statusSnapshot?.version,
    backendUpdateStatus?.behind,
    backendUpdateStatus?.updateAvailable,
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
        // The system icon: the way into every other surface, including the
        // settings that would bring a hidden item back. Never hideable.
        lockedVisible: true,
        onSelect: toggleCommandCenter,
        title: commandCenterOpen ? copy.closeCommandCenter : copy.openCommandCenter,
        toggleLabel: copy.toggleCommandCenter,
        variant: 'action'
      },
      {
        hidden: !sessionsShowing,
        id: 'gateway-switcher',
        lockedVisible: true,
        render: () => <StatusbarGatewaySwitcher />
      },
      {
        className: gatewayRestarting ? undefined : gatewayClassName,
        detail: gatewayRestarting ? copy.gatewayRestarting : gatewayDetail,
        hidden: botsShowing,
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
        // Tip only when there's a real status reason — not "gateway status" restating the label.
        title: inferenceStatus?.reason || undefined,
        toggleLabel: copy.gateway,
        variant: 'menu'
      },
      {
        hidden: !currentCwd,
        icon: <FolderOpen className="size-3" />,
        id: 'workspace-cwd',
        // Prefer the named project; fall back to the cwd leaf. Hover tip uses
        // the shared display formatter (home → ~) so statusbar and branch bar
        // agree on how a path looks.
        label: projectName || (currentCwd ? pathLeaf(currentCwd) : undefined),
        menuItems: currentCwd
          ? [
              {
                id: 'copy-workspace-path',
                label: fileMenu.copyPath,
                onSelect: () => void copyFilePath(currentCwd),
                title: displayPath(currentCwd)
              },
              {
                id: 'reveal-workspace-finder',
                label: fileMenu.revealFileManager,
                onSelect: () => void revealFile(currentCwd),
                title: displayPath(currentCwd)
              },
              {
                id: 'reveal-workspace-sidebar',
                label: fileMenu.revealInSidebar,
                onSelect: () => revealFileInTree(currentCwd),
                title: displayPath(currentCwd)
              }
            ]
          : undefined,
        title: currentCwd ? displayPath(currentCwd) : undefined,
        toggleLabel: copy.toggleWorkspace,
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
            <Codicon name="hubot" size="0.75rem" />
          ),
        id: 'agents',
        label: copy.agents,
        onSelect: openAgents,
        title: agentsOpen ? copy.closeAgents : copy.openAgents,
        toggleLabel: copy.agents,
        variant: 'action'
      },
      {
        icon: <Clock className="size-3" />,
        id: 'cron',
        label: copy.cron,
        to: CRON_ROUTE,
        toggleLabel: copy.cron,
        variant: 'action'
      },
      {
        icon: <Globe className="size-3" />,
        id: 'webhooks',
        label: copy.webhooks,
        to: WEBHOOKS_ROUTE,
        toggleLabel: copy.webhooks,
        variant: 'action'
      }
    ],
    [
      agentsOpen,
      botsShowing,
      commandCenterOpen,
      copy,
      currentCwd,
      fileMenu.copyPath,
      fileMenu.revealFileManager,
      fileMenu.revealInSidebar,
      gatewayMenuContent,
      gatewayClassName,
      gatewayDetail,
      gatewayRestarting,
      inferenceReady,
      inferenceStatus?.reason,
      openAgents,
      projectName,
      sessionsShowing,
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
        toggleLabel: copy.toggleRunningTimer,
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        // Never self-hide: the user opted this item in (it's hidden-by-
        // default), so an empty label must render as a waiting placeholder,
        // not a vanished item — an enabled-but-invisible toggle reads as
        // "another item took its spot".
        id: 'context-usage',
        label: contextUsage || '—',
        menuAlign: 'end',
        menuClassName: 'w-auto border-(--ui-stroke-secondary) p-0',
        menuContent: (
          <ContextUsagePanel breakdown={contextBreakdown} loading={contextBreakdownLoading} usage={gaugeUsage} />
        ),
        toggleLabel: copy.toggleContextUsage,
        variant: 'menu'
      },
      {
        icon: <Layers3 className="size-3" />,
        id: 'cache-hit-rate',
        // Same never-self-hide rule as the context meter: opted in means a
        // placeholder until the first cached turn reports, not a vanished item.
        label: cacheHit || '—',
        title: copy.cacheHitRateTitle,
        toggleLabel: copy.toggleCacheHitRate,
        variant: 'text'
      },
      {
        icon: <Zap className="size-3" />,
        id: 'tokens-per-second',
        label: tokensPerSecond || '—',
        title: copy.tokensPerSecondTitle,
        toggleLabel: copy.toggleTokensPerSecond,
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: copy.session,
        toggleLabel: copy.toggleSessionTimer,
        variant: 'text'
      },
      systemResourcesItem,
      {
        ...approvalModeItem,
        hidden: gatewayState !== 'open',
        toggleLabel: copy.toggleApprovalMode
      },
      {
        actionId: 'view.showTerminal',
        className: `w-7 justify-center px-0${terminalShowing ? ' bg-accent/55 text-foreground' : ''}`,
        hidden: !chatOpen,
        icon: <Terminal className="size-3.5" />,
        id: 'terminal',
        onSelect: () => togglePaneVisible('terminal'),
        title: terminalShowing ? copy.hideTerminal : copy.showTerminal,
        toggleLabel: copy.toggleTerminal,
        variant: 'action'
      },
      clientVersionItem,
      ...(backendVersionItem ? [backendVersionItem] : [])
    ],
    [
      approvalModeItem,
      backendVersionItem,
      busy,
      cacheHit,
      chatOpen,
      clientVersionItem,
      contextBar,
      contextBreakdown,
      contextBreakdownLoading,
      contextUsage,
      copy,
      gaugeUsage,
      sessionStartedAt,
      gatewayState,
      systemResourcesItem,
      terminalShowing,
      tokensPerSecond,
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

function StatusbarGatewaySwitcher() {
  const navigate = useNavigate()

  return <ConnectionSwitcher compact onConnect={() => navigate(`${SETTINGS_ROUTE}?tab=connections`)} />
}
