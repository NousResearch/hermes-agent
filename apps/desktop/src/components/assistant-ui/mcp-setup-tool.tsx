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
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { directoryEntry } from '@/lib/mcp-directory'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { clearMcpSetupRequest, type McpSetupOutcome, sessionMcpSetupRequest } from '@/store/mcp-setup'
import { notifyError } from '@/store/notifications'
import { invalidateMcpSuggestionIndex } from '@/store/suggestion-providers/mcp'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

type SetupAction = 'authorize' | 'enable' | 'install'

interface SetupArgs {
  server: string
  action: SetupAction
  reason: string
}

const CATALOG_INSTALL_POLL_MS = 1500

// Thrown by the in-flight flow when the user cancels — the declined respond
// has already been sent, so the catch path must swallow this, not report it.
const CANCELLED = Symbol('mcp-setup-cancelled')

function readSetupArgs(args: unknown): SetupArgs {
  const row = parseMaybeObject(args)
  const rawAction = typeof row.action === 'string' ? row.action : 'install'

  return {
    action: rawAction === 'enable' || rawAction === 'authorize' ? rawAction : 'install',
    reason: typeof row.reason === 'string' ? row.reason : '',
    server: typeof row.server === 'string' ? row.server : ''
  }
}

/** The tool's settled JSON — the card's outcome plus the tool-only
 *  `unanswered` status (timeout, no user action). */
type SettledResult = Omit<Partial<McpSetupOutcome>, 'status'> & {
  status?: McpSetupOutcome['status'] | 'unanswered'
  note?: string
}

function readSetupResult(result: unknown): SettledResult {
  return parseMaybeObject(result) as SettledResult
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

/** The card's strings, from this tool's own copy. The verb changes with the
 *  action (Install / Enable / Authorize); the rest is the shared consent
 *  vocabulary every connector card speaks. */
function cardCopy(copy: ReturnType<typeof useI18n>['t']['assistant']['mcpSetup'], action: SetupAction): ConnectorCardCopy {
  return {
    connectAction: action === 'enable' ? copy.enableAction : action === 'authorize' ? copy.authorizeAction : copy.installAction,
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
  // Settled → static outcome line (the flow already ran or was declined).
  if (props.result !== undefined) {
    return <McpSetupSettled {...props} />
  }

  return <McpSetupLive {...props} />
}

const McpSetupLive = (props: ToolCallMessagePartProps) => {
  const messageRunning = useAuiState(selectMessageRunning)

  // Stopped mid-prompt with no result — don't leave a dead interactive panel.
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

function McpSetupPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's request (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionMcpSetupRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const fromArgs = useMemo(() => readSetupArgs(args), [args])

  const server = fromArgs.server || request?.server || ''
  const action: SetupAction = fromArgs.action ?? request?.action ?? 'install'
  const reason = fromArgs.reason || request?.reason || ''

  const [working, setWorking] = useState(false)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [entry, setEntry] = useState<McpCatalogEntry | null | undefined>(undefined)
  const [envOpen, setEnvOpen] = useState(false)
  // Set when the user cancels mid-flight (a stuck OAuth tab, a hung install).
  // The in-flight flow checks it at every poll boundary and aborts via the
  // CANCELLED sentinel; the declined respond has already been sent by then.
  const cancelRef = useRef(false)

  // Race: tool.start fires a tick before mcp.setup.request — hold the buttons
  // until the gateway request is wired (same spinner rule as clarify).
  const ready = Boolean(request?.requestId)

  const respond = useCallback(
    async (outcome: McpSetupOutcome) => {
      // Another path (cancel racing completion) may have already resolved this
      // request; the store is the single source of truth, so bail if this
      // session's entry is gone — same guard as the approval bar.
      if (!request || sessionMcpSetupRequest(request.sessionId).get()?.requestId !== request.requestId) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

        return
      }

      // Clear first: the answer is decided, and an in-flight RPC must not
      // leave a live card that can be answered a second time.
      clearMcpSetupRequest(request.requestId, request.sessionId)

      // A successful outcome changed mcp_servers — reload the live session
      // BEFORE unblocking the tool, or the agent resumes being told the
      // server is ready while its tool snapshot still lacks it (the same
      // write-through mcp-tab's silentReload does; consent was the card
      // click, so no confirm prompt). Reload failure isn't outcome failure:
      // the config landed, tools arrive next session — report it and move on.
      if (outcome.status === 'installed' || outcome.status === 'enabled' || outcome.status === 'authorized') {
        try {
          await gateway.request('reload.mcp', { confirm: true, session_id: request.sessionId ?? undefined })
        } catch (error) {
          notifyError(error, copy.reloadFailed)
        }

        // The just-set-up server must stop being suggested immediately.
        invalidateMcpSuggestionIndex()
      }

      try {
        await gateway.request<{ status?: string }>('mcp.setup.respond', {
          request_id: request.requestId,
          result: JSON.stringify(outcome)
        })
        // tool.complete lands next → McpSetupSettled.
      } catch (error) {
        notifyError(error, copy.sendFailed)
      }
    },
    [copy.gatewayDisconnected, copy.reloadFailed, copy.sendFailed, gateway, request]
  )

  const decline = useCallback(() => {
    // While a flow is in flight this is a CANCEL: answer declined right away
    // and let the abandoned work notice via cancelRef at its next poll.
    cancelRef.current = true
    triggerHaptic('cancel')
    void respond({ server, status: 'declined' })
  }, [respond, server])

  const approve = useCallback(async () => {
    cancelRef.current = false
    const oauthScope = capabilityScoped()
    setWorking(true)

    // Poll-boundary abort for the background-install loop; the OAuth flows
    // carry their own cancel via completeMcpDesktopOAuth's `cancelled`.
    const throwIfCancelled = <T,>(value: T): T => {
      if (cancelRef.current) {
        throw CANCELLED
      }

      return value
    }

    try {
      if (action === 'enable') {
        await setMcpServerEnabled(server, true)
        triggerHaptic('submit')
        await respond({ server, status: 'enabled' })

        return
      }

      if (action === 'authorize') {
        const flow = await completeMcpDesktopOAuth({
          serverName: server,
          profile: oauthScope,
          cancelled: () => cancelRef.current
        })

        triggerHaptic('submit')
        await respond({ server, status: 'authorized', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      // Install: prefer the reviewed catalog entry when one exists; otherwise
      // fall back to the desktop suggestion directory (official URL-only
      // remotes), written through the same validated POST the dashboard's add
      // form uses. Required catalog credentials get an inline prompt first
      // (never pre-filled, never echoed back).
      let resolved = entry

      if (resolved === undefined) {
        const catalog = await getMcpCatalog()
        resolved = catalog.entries.find(candidate => candidate.name === server) ?? null
        setEntry(resolved)
      }

      if (!resolved) {
        const known = directoryEntry(server)

        if (!known) {
          await respond({ detail: copy.notInCatalog(server), server, status: 'error' })

          return
        }

        // URL-only remote: add to config, then run the OAuth/probe flow so
        // "Install" lands the user on a working server, not a 401. If the
        // flow dies after the config write (cancel, closed OAuth tab), roll
        // the write back — decline means "no server", not an unauthorized
        // entry squatting in mcp_servers (authoritative-write rule).
        await addMcpServer({ name: known.name, url: known.url }, oauthScope)

        let flow

        try {
          flow = await completeMcpDesktopOAuth({
            serverName: known.name,
            profile: oauthScope,
            cancelled: () => cancelRef.current
          })
        } catch (error) {
          await removeMcpServer(known.name, oauthScope).catch(() => {
            // Rollback is best-effort; the primary error/cancel wins.
          })
          throw error
        }

        triggerHaptic('submit')
        await respond({ server, status: 'installed', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      const required = resolved.required_env.filter(env => env.required)

      if (required.some(env => !envDraft[env.name]?.trim())) {
        // Reveal the credential fields; the user approves again once filled.
        setEnvOpen(true)

        return
      }

      const res = await installMcpCatalogEntry(server, envDraft)

      // Git-backed entries clone in the background — poll to completion so a
      // non-zero exit surfaces as a real failure instead of a false success.
      if (res.background && res.action) {
        for (;;) {
          const status = throwIfCancelled(await getActionStatus(res.action, 1))

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
      await respond({ server, status: 'installed' })
    } catch (error) {
      // User cancel: the declined respond is already on the wire — the
      // abandoned flow just stops, nothing to report.
      if (error === CANCELLED || error instanceof McpOAuthCancelled) {
        return
      }

      notifyError(error, copy.failed(server))
      await respond({
        detail: error instanceof Error ? error.message : String(error),
        server,
        status: 'error'
      })
    } finally {
      setWorking(false)
    }
  }, [action, copy, entry, envDraft, respond, server])

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
    if (!ready) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (
        active &&
        (active.isContentEditable || active.matches('a[href], button, input, select, textarea, [role="button"]'))
      ) {
        return
      }

      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        if (!working) {
          event.preventDefault()
          void approve()
        }
      } else if (event.key === 'Escape') {
        event.preventDefault()
        decline()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [approve, decline, ready, working])

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
