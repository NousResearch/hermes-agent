'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { ConnectorCard, type ConnectorCardCopy, ConnectorSummary } from '@/components/ui/connector-card'
import {
  addMcpServer,
  getActionStatus,
  getMcpCatalog,
  installMcpCatalogEntry,
  type McpCatalogEntry,
  removeMcpServer,
  setMcpServerEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { connectorText, type McpTarget, mcpTargets } from '@/lib/connector-tools'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import {
  type ConnectionRequest,
  type ConnectionTarget,
  type ConnectionTargetOutcome,
  continueConnectionRequest,
  respondToConnectionRequest,
  sessionConnectionRequest
} from '@/store/connection-request'
import { $gateway } from '@/store/gateway'
import { reconnectAction } from '@/store/gateway-reconnect'
import { notifyError } from '@/store/notifications'
import { invalidateMcpSuggestionIndex } from '@/store/suggestion-providers/mcp'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

type SetupAction = McpTarget['action']
type SetupCopy = ReturnType<typeof useI18n>['t']['assistant']['mcpSetup']

const CATALOG_INSTALL_POLL_MS = 1500

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

/** The card's strings, from this tool's own copy. The verb changes with the
 *  action (Install / Enable / Authorize); the rest is the shared consent
 *  vocabulary every connector card speaks. */
function cardCopy(
  copy: ReturnType<typeof useI18n>['t']['assistant']['mcpSetup'],
  action: SetupAction
): ConnectorCardCopy {
  return {
    connectAction:
      action === 'enable' ? copy.enableAction : action === 'authorize' ? copy.authorizeAction : copy.installAction,
    connectTitle:
      action === 'enable' ? copy.enableTitle : action === 'authorize' ? copy.authorizeTitle : copy.installTitle,
    decline: copy.decline,
    envRequired: copy.envRequired,
    grantAction: copy.authorizeAction,
    retryAction: copy.installAction,
    stateConnected: '',
    stateDeclined: copy.declined,
    stateDisabled: '',
    stateFailed: '',
    stateNeedsAuth: '',
    toolCount: copy.toolCount,
    trustCommunity: '',
    trustCommunityTip: () => '',
    trustVerified: () => '',
    trustVerifiedTip: () => ''
  }
}

export const McpSetupTool = (props: ToolCallMessagePartProps) => {
  if (props.result !== undefined) {
    return <McpSetupSettled {...props} />
  }

  return <McpSetupLive {...props} />
}

const McpSetupLive = (props: ToolCallMessagePartProps) => {
  const messageRunning = useAuiState(selectMessageRunning)

  if (!messageRunning) {
    return <ToolFallback {...props} />
  }

  return <McpSetupPending {...props} />
}

function McpSetupSettled({ args, result }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  const fromArgs = useMemo(() => readSetupArgs(args), [args])
  const fromResult = useMemo(() => readSetupResult(result), [result])

  const server = fromResult.server || fromArgs.server
  const status = fromResult.status ?? 'error'
  const displayName = prettyName(server)

  const line =
    status === 'installed'
      ? copy.installed(displayName)
      : status === 'enabled'
        ? copy.enabled(displayName)
        : status === 'authorized'
          ? copy.authorized(displayName)
          : status === 'declined'
            ? copy.declined
            : status === 'unanswered'
              ? copy.unanswered
              : copy.failed(displayName)

  const ok = status === 'installed' || status === 'enabled' || status === 'authorized'
  const neutral = status === 'declined' || status === 'unanswered'
  const toolCount = Array.isArray(fromResult.tools) ? fromResult.tools.length : 0

  // Settled is scaffolding, the same line a spent connector offer collapses
  // to: the name, then the verdict as meta. A failure keeps its reason.
  return (
    <ConnectorSummary
      connector={{ name: server, title: displayName }}
      meta={
        ok && toolCount > 0
          ? `${line} · ${copy.toolCount(toolCount)}`
          : !ok && !neutral && fromResult.detail
            ? `${line} — ${fromResult.detail}`
            : line
      }
      tone={ok ? 'ok' : neutral ? undefined : 'error'}
    />
  )
}

export function McpSetupPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  // Use the rendering transcript's session, not the globally active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionConnectionRequest(sessionId), [sessionId])
  const request = useStore($request)
  const action = useMemo(() => readSetupAction(args), [args])
  const title = TITLE[action](copy)

  // `tool.start` arrives before `connection.request`.
  if (!request) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 flex items-center gap-2')} data-slot="connector-card">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
        <span className="text-(--ui-text-tertiary)">{title}</span>
      </div>
    )
  }

  const open = request.targets.filter(target => !resolved(target))

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer>
      <ConnectorCard title={title}>
        {request.targets.map(target => (
          <McpSetupRow
            action={action}
            copy={copy}
            key={target.name}
            request={request}
            single={open.length === 1 && open[0] === target}
            target={target}
          />
        ))}
      </ConnectorCard>
      {open.length > 0 ? (
        <div className="px-3.5">
          <Button onClick={() => void continueConnectionRequest(request)} size="xs" variant="textStrong">
            {t.common.continue}
          </Button>
        </div>
      ) : null}
    </div>
  )
}

interface McpSetupRowProps {
  action: SetupAction
  copy: SetupCopy
  request: ConnectionRequest
  /** The only open row owns ⌘⏎; with several rows the buttons are the path. */
  single: boolean
  target: ConnectionTarget
}

function McpSetupRow({ action, copy, request, single, target }: McpSetupRowProps) {
  const { t } = useI18n()
  const gateway = useStore($gateway)
  const [working, setWorking] = useState(false)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [entry, setEntry] = useState<McpCatalogEntry | null | undefined>(undefined)
  const [envOpen, setEnvOpen] = useState(false)
  const server = target.name
  const displayName = prettyName(server)
  const done = resolved(target)

  const respond = async (outcome: ConnectionTargetOutcome) => {
    if (!gateway) {
      notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed, { action: reconnectAction() })

      return
    }

    if (outcome.status === 'connected') {
      invalidateMcpSuggestionIndex()
    }

    try {
      await respondToConnectionRequest(request, { targets: [outcome] })
    } catch (error) {
      notifyError(error, copy.sendFailed)
    }
  }

  const approve = async () => {
    const oauthScope = capabilityScoped()
    setWorking(true)

    try {
      if (action === 'enable') {
        await setMcpServerEnabled(server, true)
        triggerHaptic('submit')
        await respond({ name: server, status: 'connected' })

        return
      }

      if (action === 'authorize') {
        const flow = await completeMcpDesktopOAuth({ serverName: server, profile: oauthScope })

        triggerHaptic('submit')
        await respond({ name: server, status: 'connected', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      let catalogEntry = entry

      if (catalogEntry === undefined) {
        const catalog = await getMcpCatalog()
        catalogEntry = catalog.entries.find(candidate => candidate.name === server) ?? null
        setEntry(catalogEntry)
      }

      if (!catalogEntry) {
        await respond({ detail: copy.notInCatalog(server), name: server, status: 'failed' })

        return
      }

      const required = catalogEntry.required_env.filter(env => env.required)

      if (required.some(env => !envDraft[env.name]?.trim())) {
        setEnvOpen(true)

        return
      }

      const res = await installMcpCatalogEntry(server, envDraft)

      // Poll background installs so non-zero exits cannot report false success.
      if (res.background && res.action) {
        for (;;) {
          const status = await getActionStatus(res.action, 1)

          if (!status.running) {
            if (status.exit_code !== 0) {
              throw new Error(copy.failed(server))
            }

            break
          }

          await new Promise(resolve => setTimeout(resolve, CATALOG_INSTALL_POLL_MS))
        }
      }

      triggerHaptic('submit')
      await respond({ name: server, status: 'connected' })
    } catch (error) {
      // The user closed the sign-in window; the row simply offers again.
      if (error instanceof McpOAuthCancelled) {
        return
      }

      notifyError(error, copy.failed(displayName))
      await respond({
        detail: error instanceof Error ? error.message : String(error),
        name: server,
        status: 'failed'
      })
    } finally {
      setWorking(false)
    }
  }

  const displayName = prettyName(server)
  const card = cardCopy(copy, action)

  // What connecting actually means — the endpoint that will be contacted.
  // Catalog entries carry their transport URL in the API response; the
  // static directory remains a fallback rung for older backends.
  const known = directoryEntry(server)
  const sourceLine = action === 'install' ? (entry?.url ?? known?.url ?? copy.catalogSource) : null

  // ⌘/Ctrl+Enter → approve, Esc → decline/cancel. Same accelerators, same
  // guard shape as the approval bar (tool/approval.tsx). Unlike approve, Esc
  // stays live while a flow is in flight — that's the cancel path. Stands
  // down whenever a focusable control has focus (clarify's rule): a keystroke
  // meant for the composer, a popover, or the card's own credential fields
  // must never silently approve an install or throw away typed input.
  useEffect(() => {
    if (!single || done) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented || !isSubmitEnter(event) || !(event.metaKey || event.ctrlKey)) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (
        active &&
        (active.isContentEditable || active.matches('a[href], button, input, select, textarea, [role="button"]'))
      ) {
        return
      }

      if (!working) {
        event.preventDefault()
        void approve()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  })

  if (!ready) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 flex items-center gap-2')} data-slot="connector-card">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
        <span className="text-(--ui-text-tertiary)">{card.connectTitle?.(displayName)}</span>
      </div>
    )
  }

  // The same consent card the connector offer renders: one shape for every
  // "connect this?" in the transcript. `phase` is what flips the card into
  // its working state (spinner on the action, decline becomes cancel).
  return (
    <ConnectorCard
      accelerators
      connector={{
        description: reason || undefined,
        name: server,
        requiredEnv: entry?.required_env,
        title: displayName
      }}
      copy={{ ...card, decline: working ? t.common.cancel : card.decline }}
      envDraft={envDraft}
      envOpen={envOpen && !!entry && entry.required_env.length > 0}
      onConnect={() => void approve()}
      onDismiss={decline}
      onEnvChange={(key, value) => setEnvDraft(prev => ({ ...prev, [key]: value }))}
      phase={working ? '' : undefined}
      source={sourceLine ? { text: sourceLine } : undefined}
      state="not_configured"
      variant="avatar"
    />
  )
}
