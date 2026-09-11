import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { isFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { FirstBuildConnectorOffer } from '@/components/assistant-ui/first-build-connectors'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { ConnectorCard, type ConnectorCardCopy } from '@/components/ui/connector-card'
import { Loader } from '@/components/ui/loader'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { connectionRows, connectorCalls, connectorTitle, connectorToolName, recordOf } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'
import { createConnectorFlow } from '@/store/connector-flow'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export function ConnectorTool(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const messages = useStore(view.$messages)
  const firstBuild = isFirstBuildSession(storedId)

  // One live card per offer. Every manage_connections call renders through
  // here, but only ONE is the card the user acts on; the rest are settled
  // tool rows. Which one: consecutive calls naming the same apps are one
  // exchange — connect, the wait the agent parks in while the user signs in,
  // the status it runs when the connection lands — and the FIRST of the last
  // exchange is the card. The newest would demote the card mid-authorization
  // into a row and mint a fresh one below it. A catalog listing (status with
  // nothing named) after a targeted ask never starts an exchange: it is a
  // read, not an offer.
  const offers = messages
    .flatMap(message => message.parts)
    .filter(
      part =>
        part.type === 'tool-call' &&
        (part.toolName === 'manage_connections' || connectorCalls(part.toolName, part.args).length > 0)
    )

  const keyOf = (part: (typeof offers)[number]) =>
    part.type === 'tool-call'
      ? connectionRows(part.args, part.result)
          .map(row => row.connector)
          .sort()
          .join('|')
      : ''

  const targeted = (part: (typeof offers)[number]) => {
    if (part.type !== 'tool-call') {
      return false
    }

    const asked = recordOf(part.args).connectors

    return Array.isArray(asked) && asked.length > 0
  }

  let liveId: string | undefined
  let liveKey: string | null = null
  let sawTargeted = false

  for (const part of offers) {
    if (part.type !== 'tool-call') {
      continue
    }

    const key = keyOf(part)

    if (sawTargeted && !targeted(part)) {
      continue
    }

    sawTargeted ||= targeted(part)

    if (key !== liveKey) {
      liveKey = key
      liveId = part.toolCallId
    }
  }

  const historical = liveId !== props.toolCallId

  const [owner, setOwner] = useState<{
    storedId: string
    runtimeId: string
    connectionId: null | string
    profile: string
  } | null>(null)

  const [ownerFailure, setOwnerFailure] = useState<string | null>(null)

  useEffect(() => {
    if (!storedId || !runtimeId || historical) {
      return
    }

    let cancelled = false
    const ambientProfile = $activeGatewayProfile.get()
    void resolveSessionOwner(storedId)
      .then(scope => {
        assertSessionOwnerResolved(scope, { method: 'connectors.list', sessionId: storedId })

        if (!cancelled) {
          setOwner({
            storedId,
            runtimeId,
            connectionId: isSessionOwnerRoute(scope) ? scope.connectionId : null,
            profile: isSessionOwnerRoute(scope) ? scope.profile : scope || ambientProfile
          })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setOwner(null)
          setOwnerFailure(`${storedId}:${runtimeId}`)
        }
      })

    return () => {
      cancelled = true
    }
  }, [storedId, runtimeId, historical])
  const rows = connectionRows(props.args, props.result)
  const signature = rows.map(row => row.connector).join('|')
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'

  // The TUI shape, stolen: the agent parks inside manage_connections
  // action="wait", which blocks the turn and polls the gateway, instead of
  // deciding what "not connected" means and building around the app. Each
  // card action sends one hidden line so the agent takes the right next call.
  // Read through a ref so the flow (memoised on identity) always nudges the
  // live composer target, never the one it was built with. Busy is the
  // composer's problem: a hidden request mid-turn steers or queues there.
  const nudgeRef = useRef((_text: string) => {})

  nudgeRef.current = (text: string) => {
    requestComposerSubmit(`[connectors] ${text}`, { displayKind: 'hidden', target })
  }

  const flow = useMemo(() => {
    if (
      firstBuild ||
      historical ||
      !runtimeId ||
      !owner ||
      owner.storedId !== storedId ||
      owner.runtimeId !== runtimeId
    ) {
      return null
    }

    const seeds = signature ? signature.split('|').map(connector => ({ connector })) : []

    return createConnectorFlow(runtimeId, seeds, {
      request: (method, params) => requestGatewayForAgent(owner.connectionId, owner.profile, method, params, 45000),
      open: async url => {
        if (!window.hermesDesktop?.openExternal) {
          throw new Error('System browser unavailable')
        }

        await window.hermesDesktop.openExternal(url)
      },
      onWaiting: slug =>
        nudgeRef.current(
          `The user clicked Connect for ${connectorTitle(slug)} and the sign-in is open in their browser. Call manage_connections action="wait" connectors=["${slug}"] now and hold there until it reports connected. Do NOT call connect again — a second link cancels the one they are signing in with. Say nothing until wait returns.`
        )
    })
  }, [runtimeId, owner, storedId, signature, historical, firstBuild])

  const { t } = useI18n()
  // Ordinary sessions require a click to begin authorization.
  useEffect(() => {
    if (!flow) {
      return
    }

    return () => flow.dispose()
  }, [flow])
  useEffect(() => {
    if (flow) {
      void flow.refresh()
    }
  }, [flow, props.result])

  if (historical) {
    return <ToolFallback {...props} />
  }

  if (firstBuild && storedId && owner?.storedId === storedId && owner.runtimeId === runtimeId) {
    return (
      <FirstBuildConnectorOffer
        connectionId={owner.connectionId}
        part={props}
        profile={owner.profile}
        runtimeId={owner.runtimeId}
        storedId={storedId}
        target={view.kind === 'tile' ? `tile:${storedId}` : 'main'}
      />
    )
  }

  if (!flow) {
    return (
      <p className="text-xs text-muted-foreground">
        {ownerFailure === `${storedId}:${runtimeId}` ? t.connectors.ownerMissing : t.connectors.checking}
      </p>
    )
  }

  return (
    <ConnectorOffer
      flow={flow}
      key={`${runtimeId}:${signature}`}
      onSkipped={slug =>
        nudgeRef.current(
          `The user chose Not now for ${connectorTitle(slug)}. Do not connect it, do not route around it with another client, credential or CLI for the same app. Continue the task without it, or ask what they want to do.`
        )
      }
    />
  )
}

interface ConnectorOfferProps {
  flow: ReturnType<typeof createConnectorFlow>
  /** The user waved the app off. */
  onSkipped: (slug: string) => void
}

export function ConnectorOffer({ flow, onSkipped }: ConnectorOfferProps) {
  const state = useStore(flow.state)
  const { t } = useI18n()
  const copy = t.connectors
  const [query, setQuery] = useState('')
  const active = state.rows.some(row => row.phase === 'opening' || row.phase === 'waiting')

  const cardCopy: ConnectorCardCopy = {
    connectAction: copy.connect,
    decline: copy.skip,
    envRequired: '',
    grantAction: copy.grant,
    retryAction: copy.retry,
    stateConnected: copy.connected,
    stateDeclined: copy.skipped,
    stateDisabled: copy.disabled,
    stateFailed: copy.failed,
    stateNeedsAuth: copy.needsAuth,
    toolCount: count => String(count),
    trustCommunity: '',
    trustCommunityTip: () => '',
    trustVerified: () => '',
    trustVerifiedTip: () => ''
  }

  if (state.loading) {
    return <Loader />
  }

  const rows = state.rows.filter(row => connectorTitle(row.connector).toLowerCase().includes(query.toLowerCase()))
  // A targeted ask ("connect Gmail") is one or two cards, each already a
  // complete question. A heading, a disclaimer and a refresh control over
  // them is a settings panel dropped into the chat. Only a real catalog — the
  // model asked for status with nothing named — earns the chrome.
  const catalog = state.rows.length > 4

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer>
      {catalog ? (
        <div className="grid gap-0.5 px-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">{copy.title}</span>
            <Button onClick={() => void flow.refresh()} size="xs" variant="text">
              {copy.refresh}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">{copy.disclaimer}</p>
        </div>
      ) : null}
      {state.error ? (
        <p className="flex flex-wrap items-center gap-2 px-1 text-xs text-destructive" role="alert">
          {copy.statusError}
          <Button onClick={() => void flow.refresh()} size="xs" variant="text">
            {copy.retry}
          </Button>
        </p>
      ) : null}
      {!state.available && !state.error ? <p className="px-1 text-xs text-muted-foreground">{copy.unavailable}</p> : null}
      {catalog ? <SearchField onChange={setQuery} placeholder={copy.search} value={query} /> : null}
      <div className={cn('grid min-w-0', catalog && 'max-h-96 overflow-y-auto')}>
        {rows.map(row => (
          <div className="grid" key={row.connector}>
            <ConnectorCard
              actionDisabled={!state.available || row.enabled === false || !!state.error}
              collapseWhenSettled={false}
              connector={{
                name: row.connector,
                title: row.name || connectorTitle(row.connector),
                description: row.description || copy.describe(row.name || connectorTitle(row.connector))
              }}
              copy={{
                ...cardCopy,
                connectTitle: copy.connectTitle,
                decline: row.phase === 'opening' || row.phase === 'waiting' ? copy.cancel : copy.skip,
                connectAction: ['expired', 'revoked'].includes(row.connectionStatus ?? '') ? copy.grant : copy.connect
              }}
              dismissed={row.phase === 'skipped'}
              onConnect={() => void flow.connect(row.connector)}
              onDismiss={() => {
                const wasPending = ['opening', 'waiting'].includes(row.phase)
                flow.skip(row.connector)

                // A cancel mid-authorization is not a skip: the agent may be
                // parked in wait and will hear the timeout itself.
                if (!wasPending) {
                  onSkipped(row.connector)
                }
              }}
              otherBusy={active && !['opening', 'waiting'].includes(row.phase)}
              outcome={
                row.phase === 'connected'
                  ? { status: 'connected' }
                  : row.phase === 'error'
                    ? {
                        status: 'error',
                        detail:
                          row.error === 'connect'
                            ? copy.connectError
                            : row.error === 'unavailable'
                              ? copy.unavailable
                              : copy.statusError
                      }
                    : undefined
              }
              phase={row.phase === 'opening' ? copy.opening : row.phase === 'waiting' ? copy.waiting : undefined}
              state={
                row.enabled === false
                  ? 'disabled'
                  : ['expired', 'revoked'].includes(row.connectionStatus ?? '')
                    ? 'needs_auth'
                    : 'not_configured'
              }
              variant="avatar"
            />
            {row.phase === 'timeout' ? (
              <div className="flex flex-wrap items-center gap-2 px-3.5 text-xs text-muted-foreground">
                <span>{copy.timeout}</span>
                <Button onClick={() => void flow.keepWaiting(row.connector)} size="xs" variant="textStrong">
                  {copy.keepWaiting}
                </Button>
              </div>
            ) : null}
          </div>
        ))}
        {!rows.length && state.available ? <p className="px-1 text-xs text-muted-foreground">{copy.empty}</p> : null}
      </div>
    </div>
  )
}

/** Keep execution output in the standard disclosure, with one row per app call. */
export function ConnectorExecution(props: ToolCallMessagePartProps) {
  const calls = connectorCalls(props.toolName, props.args)
  const input = recordOf(props.args)
  const batch = Array.isArray(input.calls) ? input.calls : [input]

  // Mixed remote batches keep their complete disclosure and original result order.
  if (props.toolName === 'tool_call' && calls.length !== batch.length) {
    return <ToolFallback {...props} />
  }

  const output = recordOf(props.result)
  const results = Array.isArray(output.results) ? output.results : []

  const repair = calls
    .filter((_call, index) => {
      const item = recordOf(props.toolName === 'tool_call' ? results[index] : props.result)

      return ['CONNECTION_REQUIRED', 'CONNECTION_EXPIRED', 'AUTH_REQUIRED'].includes(
        String(recordOf(item.error).code ?? '')
      )
    })
    .map(call => {
      // SAFETY: connectorCalls includes only names accepted by connectorToolName.
      return connectorToolName(call.name)!.connector
    })

  return (
    <>
      {calls.map((call, index) => {
        const item =
          props.toolName === 'tool_call' ? (results[index] ?? (output.error ? output : undefined)) : props.result

        const result = recordOf(item)
        // SAFETY: connectorCalls includes only names accepted by connectorToolName.
        const identity = connectorToolName(call.name)!

        return (
          <ToolFallback
            {...props}
            args={recordOf(call.arguments)}
            isError={Boolean(result.error) || props.isError === true}
            key={`${props.toolCallId}:${index}`}
            result={props.result === undefined ? undefined : (item ?? { error: 'Missing connector result' })}
            toolCallId={`${props.toolCallId}:${index}`}
            toolName={`${connectorTitle(identity.connector)}: ${identity.action}`}
          />
        )
      })}
      {repair.length ? (
        <ConnectorTool {...props} args={{ action: 'status', connectors: repair }} result={undefined} />
      ) : null}
    </>
  )
}
