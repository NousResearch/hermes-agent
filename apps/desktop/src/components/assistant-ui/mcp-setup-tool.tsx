'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorSummary, outcomeMeta } from '@/components/ui/connector-card'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AlertCircle, Loader2 } from '@/lib/icons'
import {
  connectConnector,
  type Connector,
  ConnectorCancelled,
  ConnectorNeedsAuth,
  type ConnectorResolution,
  type ConnectorState,
  connectorTitle,
  type ConnectPhase,
  invalidateConnectorCache,
  loadConnectorStates,
  resolveConnectors
} from '@/lib/mcp-connectors'
import { McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import {
  buildSetupOutcome,
  clearMcpSetupRequest,
  type McpConnectorOutcome,
  type McpSetupOutcome,
  sessionMcpSetupRequest
} from '@/store/mcp-setup'
import { notifyError } from '@/store/notifications'
import { invalidateMcpSuggestionIndex } from '@/store/suggestion-providers/mcp'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

/**
 * The inline connector cards.
 *
 * A task needing Jira *and* Figma offers both at once instead of blocking
 * twice — one card per connector, because a connector IS the unit of state
 * here: its own endpoint, its own sign-in, its own failure.
 *
 * Everything about *how* a connector connects — write config, flip enabled,
 * probe, sign in — belongs to `lib/mcp-connectors`. These components own the
 * consent surface and the phase presentation, nothing else. That's what lets
 * a no-auth server connect with one click and an OAuth one open a browser
 * tab from the same button, with no branching here.
 *
 * Each card also owns its own *recovery*. Three connectors is three OAuth
 * flows against three servers — under the MCP auth spec a token is bound to
 * one resource, so partial failure is a property of the protocol and the UI
 * only gets to choose whether it represents it honestly. So a card that
 * fails keeps its Retry while its siblings settle around it, and the tool
 * answers once every card has either connected or been waved off.
 *
 * Consent vocabulary follows the approval bar: primary-tinted action with
 * `⌘⏎`, ghost decline with `Esc`, clarify's focus stand-down so keystrokes
 * meant for the composer are never eaten.
 */

type SetupAction = 'authorize' | 'connect' | 'enable' | 'install'

type SetupCopy = ReturnType<typeof useI18n>['t']['assistant']['mcpSetup']

interface SetupArgs {
  servers: string[]
  action: SetupAction
  reason: string
}

function readSetupArgs(args: unknown): SetupArgs {
  const row = parseMaybeObject(args)
  const rawAction = typeof row.action === 'string' ? row.action : 'connect'

  const listed = Array.isArray(row.servers)
    ? row.servers.filter((name): name is string => typeof name === 'string' && name.trim().length > 0)
    : []

  const single = typeof row.server === 'string' && row.server.trim() ? [row.server] : []

  return {
    action: rawAction === 'enable' || rawAction === 'authorize' || rawAction === 'install' ? rawAction : 'connect',
    reason: typeof row.reason === 'string' ? row.reason : '',
    servers: [...new Set([...single, ...listed])]
  }
}

/** The tool's settled JSON — the card's outcome plus the tool-only
 *  `unanswered` status (timeout, no user action). */
type SettledResult = Omit<Partial<McpSetupOutcome>, 'status'> & {
  status?: 'unanswered' | McpSetupOutcome['status']
  note?: string
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

// Same platform sniff the approval bar uses for its accelerator hint.
const isMac = typeof navigator !== 'undefined' && /Mac|iP(hone|ad|od)/.test(navigator.platform)

const ICON_CLASS = 'mt-px size-4 shrink-0 text-(--ui-text-tertiary)'

function SetupLine({ children, trailing }: { children: ReactNode; trailing?: ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      <div className="min-w-0 flex-1">{children}</div>
      {trailing}
    </div>
  )
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
  const fromResult = useMemo(() => parseMaybeObject(result) as SettledResult, [result])

  const status = fromResult.status ?? 'error'

  // Older single-connector answers have no `connectors` array; synthesize one
  // so the settled view reads identically for both shapes.
  const connectors: McpConnectorOutcome[] = Array.isArray(fromResult.connectors)
    ? fromResult.connectors
    : [
        {
          server: fromResult.server || fromArgs.servers[0] || '',
          status: status === 'connected' ? 'connected' : status === 'declined' ? 'declined' : 'error'
        }
      ]

  return (
    <div className="my-1.5" data-slot="mcp-setup-inline">
      {connectors.map(connector => (
        <ConnectorSummary
          connector={{ name: connector.server, title: connectorTitle(connector.server) }}
          key={connector.server}
          meta={outcomeMeta(connector, copy)}
          tone={connector.status === 'error' ? 'error' : undefined}
        />
      ))}
    </div>
  )
}

/** One offered connector: what it is, and where it stands right now. */
interface RowModel {
  connector: Connector
  state: ConnectorState
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

  const names = useMemo(
    () => (fromArgs.servers.length > 0 ? fromArgs.servers : (request?.servers ?? [])),
    [fromArgs.servers, request?.servers]
  )

  const action: SetupAction = fromArgs.action ?? request?.action ?? 'connect'
  const reason = fromArgs.reason || request?.reason || ''
  const agentSteps = useMemo(() => request?.steps ?? [], [request?.steps])
  // Names are the identity of this card's offer; join so the resolve effect
  // doesn't re-run on every render just because the array is a new object.
  const namesKey = names.join(',')

  const [rows, setRows] = useState<null | RowModel[]>(null)
  // Per-connector results. A card is finished when it has a `connected` entry
  // or the user dismissed it; a failed entry keeps its card live and retryable.
  const [results, setResults] = useState<Record<string, McpConnectorOutcome>>({})
  const [dismissed, setDismissed] = useState<Record<string, boolean>>({})
  const [inFlight, setInFlight] = useState<null | { name: string; phase: ConnectPhase }>(null)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [envOpenFor, setEnvOpenFor] = useState<null | string>(null)
  const [unresolved, setUnresolved] = useState<string[]>([])
  // Set when the user cancels mid-flight (a stuck OAuth tab, a hung install).
  // The in-flight pass checks it at every boundary and stops there; the
  // respond carrying whatever had already landed has been sent by then.
  const cancelRef = useRef(false)
  // `results`/`dismissed` are what render; these are what's true *right now*.
  // Each card commits the moment it settles, so the "are we done" check and an
  // Esc mid-connect both read live values rather than a stale closure.
  const resultsRef = useRef<Record<string, McpConnectorOutcome>>({})
  const dismissedRef = useRef<Record<string, boolean>>({})

  const commitResult = useCallback((name: string, outcome: McpConnectorOutcome) => {
    resultsRef.current = { ...resultsRef.current, [name]: outcome }
    setResults(resultsRef.current)
  }, [])

  // Resolve the offered names down the connector ladder (catalog → curated
  // directory → public registry) and read their current state, once.
  useEffect(() => {
    if (names.length === 0) {
      return
    }

    let live = true

    void (async () => {
      const resolution = await resolveConnectors(names).catch((): ConnectorResolution => ({
        connectors: [],
        unresolved: names
      }))

      const states = await loadConnectorStates(resolution.connectors.map(entry => entry.name)).catch(
        (): Record<string, ConnectorState> => ({})
      )

      if (!live) {
        return
      }

      setUnresolved(resolution.unresolved)
      setRows(
        resolution.connectors.map(connector => ({
          // A reviewed connector's own steps always win — those were checked
          // by a human. The agent's only get a card when the connector came
          // with none, which is every registry entry.
          connector: connector.setup.length > 0 ? connector : { ...connector, setup: agentSteps },
          // An explicit `authorize` is the agent saying it already saw a 401,
          // which config alone can't tell us.
          state: action === 'authorize' ? 'needs_auth' : (states[connector.name] ?? 'not_configured')
        }))
      )
    })()

    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- namesKey is the stable identity of `names`
  }, [action, agentSteps, namesKey])

  // Race: tool.start fires a tick before mcp.setup.request — hold the buttons
  // until the gateway request is wired (same spinner rule as clarify).
  const ready = Boolean(request?.requestId) && rows !== null

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

      // Anything that landed changed mcp_servers — reload the live session
      // BEFORE unblocking the tool, or the agent resumes being told the
      // connector is ready while its tool snapshot still lacks it. Reload
      // failure isn't outcome failure: the config landed, tools arrive next
      // session — report it and move on.
      if (outcome.connectors.some(connector => connector.status === 'connected')) {
        try {
          await gateway.request('reload.mcp', { confirm: true, session_id: request.sessionId ?? undefined })
        } catch (error) {
          notifyError(error, copy.reloadFailed)
        }

        // The just-connected servers must stop being suggested immediately.
        invalidateMcpSuggestionIndex()
        invalidateConnectorCache()
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

  /** Answer the tool. Cards with no result were never connected — a card the
   *  user dismissed, or walked away from, is a decline of that connector. */
  const settle = useCallback(
    async (final: Record<string, McpConnectorOutcome>) => {
      await respond(
        buildSetupOutcome({
          names: (rows ?? []).map(row => row.connector.name),
          results: final,
          server: names[0] ?? ''
        })
      )
    },
    [names, respond, rows]
  )

  /** Every card is finished when it either connected or was dismissed; a
   *  failed card is NOT finished, because its Retry is still on offer. */
  const finished = useCallback(
    (name: string) => resultsRef.current[name]?.status === 'connected' || dismissedRef.current[name],
    []
  )

  const settleIfDone = useCallback(async () => {
    if ((rows ?? []).every(row => finished(row.connector.name))) {
      await settle(resultsRef.current)
    }
  }, [finished, rows, settle])

  /** Dismiss one card. The last one standing closes out the tool. */
  const dismiss = useCallback(
    (name: string) => {
      dismissedRef.current = { ...dismissedRef.current, [name]: true }
      setDismissed(dismissedRef.current)
      triggerHaptic('cancel')
      void settleIfDone()
    },
    [settleIfDone]
  )

  /** Esc — give up on everything still unfinished at once. Connectors that
   *  already succeeded stay reported as connected. */
  const dismissAll = useCallback(() => {
    cancelRef.current = true
    dismissedRef.current = Object.fromEntries((rows ?? []).map(row => [row.connector.name, true]))
    setDismissed(dismissedRef.current)
    triggerHaptic('cancel')
    void settle(resultsRef.current)
  }, [rows, settle])

  const connect = useCallback(
    async (row: RowModel) => {
      const name = row.connector.name

      // Credentials this connector declares but the user hasn't filled in yet
      // — reveal its fields and wait for a second click.
      if (
        row.state === 'not_configured' &&
        row.connector.requiredEnv.some(env => env.required && !envDraft[env.name]?.trim())
      ) {
        setEnvOpenFor(name)

        return
      }

      cancelRef.current = false
      setInFlight({ name, phase: 'adding' })

      try {
        const result = await connectConnector(row.connector, row.state, {
          cancelled: () => cancelRef.current,
          env: envDraft,
          onPhase: phase => setInFlight(current => (current?.name === name ? { name, phase } : current))
        })

        commitResult(name, { server: name, status: 'connected', tools: result.tools })
        triggerHaptic('submit')
      } catch (error) {
        // A closed sign-in tab is not a refusal — leave the card as it was so
        // its Connect is still there.
        if (!(error instanceof ConnectorCancelled) && !(error instanceof McpOAuthCancelled)) {
          commitResult(name, {
            detail: error instanceof Error ? error.message : String(error),
            needsAuth: error instanceof ConnectorNeedsAuth,
            server: name,
            status: 'error'
          })
        }

        // A refused credential is a wrong credential. Open the field that
        // holds it so the correction is right there, rather than behind
        // another click on a button that reads Grant access.
        if (error instanceof ConnectorNeedsAuth && row.connector.requiredEnv.length > 0) {
          setEnvOpenFor(name)
        }

        // The failure usually left the connector in config — an install that
        // wrote its stanza, a sign-in that got a token too narrow to use. The
        // row's state was read before any of that, so a retry against it would
        // re-run the wrong half of the flow: re-installing what is installed,
        // or adding what is already there instead of asking for access again.
        const states = await loadConnectorStates([name]).catch((): Record<string, ConnectorState> => ({}))

        setRows(
          current =>
            current?.map(entry =>
              entry.connector.name === name ? { ...entry, state: states[name] ?? entry.state } : entry
            ) ?? current
        )
      } finally {
        setInFlight(null)
      }

      if (!cancelRef.current) {
        await settleIfDone()
      }
    },
    [commitResult, envDraft, settleIfDone]
  )

  // ⌘/Ctrl+Enter acts on the first unfinished card, Esc gives up on all of
  // them. Same accelerators and the same guard shape as the approval bar
  // (tool/approval.tsx). Esc stays live while a connect is in flight — that's
  // the cancel path for a stuck OAuth tab. Stands down whenever a focusable
  // control has focus (clarify's rule): a keystroke meant for the composer, a
  // popover, or a card's credential field must never silently connect
  // something or throw away typed input.
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
        const next = (rows ?? []).find(row => !finished(row.connector.name))

        if (next && !inFlight) {
          event.preventDefault()
          void connect(next)
        }
      } else if (event.key === 'Escape') {
        event.preventDefault()
        dismissAll()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [connect, dismissAll, finished, inFlight, ready, rows])

  if (!ready) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 flex items-center gap-2')} data-slot="mcp-setup-inline">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
        <span className="text-(--ui-text-tertiary)">{copy.lookingUp(names.map(connectorTitle).join(', '))}</span>
      </div>
    )
  }

  // Every offered name failed to resolve — there is nothing to consent to,
  // so say which ones and let the agent move on rather than showing a
  // connect button that cannot work.
  if (rows.length === 0) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 grid gap-1.5')} data-slot="mcp-setup-inline">
        <SetupLine trailing={<AlertCircle aria-hidden className={cn(ICON_CLASS, 'text-destructive')} />}>
          <span className="font-medium">{copy.notFound(unresolved.map(connectorTitle).join(', '))}</span>
        </SetupLine>
        <div className="flex items-center gap-2.5">
          <Button
            className="h-6 gap-1.5 rounded-md px-1.5 text-xs font-normal text-(--ui-text-tertiary) hover:text-foreground"
            onClick={dismissAll}
            size="xs"
            variant="ghost"
          >
            {copy.dismiss}
            <span className="text-[0.625rem] opacity-55">Esc</span>
          </Button>
        </div>
      </div>
    )
  }

  // One card per connector. A connector IS the unit of state here — its own
  // auth, its own endpoint, its own failure — so it gets its own surface
  // rather than a row inside a shared verdict.
  return (
    <div className="my-1.5 grid gap-2" data-slot="mcp-setup-inline">
      {reason ? <p className="text-(--ui-text-secondary)">{reason}</p> : null}

      {rows.map(row => (
        <ConnectorCard
          connector={row.connector}
          copy={copy}
          dismissed={dismissed[row.connector.name] ?? false}
          envDraft={envDraft}
          envOpen={envOpenFor === row.connector.name}
          key={row.connector.name}
          onConnect={() => void connect(row)}
          onDismiss={() => dismiss(row.connector.name)}
          onEnvChange={(key, value) => setEnvDraft(prev => ({ ...prev, [key]: value }))}
          // One connect at a time: two OAuth tabs racing for focus is hostile.
          otherBusy={inFlight !== null && inFlight.name !== row.connector.name}
          outcome={results[row.connector.name]}
          phase={inFlight?.name === row.connector.name ? copy.phase[inFlight.phase] : undefined}
          state={row.state}
        />
      ))}

      {unresolved.length > 0 && (
        <p className="text-[0.6875rem] text-(--ui-text-tertiary)">
          {copy.notFound(unresolved.map(connectorTitle).join(', '))}
        </p>
      )}
    </div>
  )
}
