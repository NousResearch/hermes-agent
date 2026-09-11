import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { ConnectorCard, type ConnectorCardCopy } from '@/components/ui/connector-card'
import { Loader } from '@/components/ui/loader'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { connectionRows, connectorCalls, connectorTitle, connectorToolName, recordOf } from '@/lib/connector-tools'
import { createConnectorFlow } from '@/store/connector-flow'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export function ConnectorTool(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const busy = useStore(view.$busy)
  const messages = useStore(view.$messages)

  const latest = messages
    .flatMap(message => message.parts)
    .filter(
      part =>
        part.type === 'tool-call' &&
        (part.toolName === 'manage_connections' || connectorCalls(part.toolName, part.args).length > 0)
    )
    .at(-1)

  const historical = latest?.type === 'tool-call' && latest.toolCallId !== props.toolCallId

  const [owner, setOwner] = useState<{
    storedId: string
    runtimeId: string
    connectionId: null | string
    profile: string
  } | null>(null)

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
        }
      })

    return () => {
      cancelled = true
    }
  }, [storedId, runtimeId, historical])
  const rows = connectionRows(props.args, props.result)
  const signature = rows.map(row => row.connector).join('|')

  const flow = useMemo(() => {
    if (historical || !runtimeId || !owner || owner.storedId !== storedId || owner.runtimeId !== runtimeId) {
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
      }
    })
  }, [runtimeId, owner, storedId, signature, historical])

  const { t } = useI18n()
  // A result is a snapshot. Reopening a transcript only refreshes status; it
  // cannot mint links, open tabs or restart an abandoned authorization.
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

  if (!flow) {
    return <p className="text-xs text-muted-foreground">{t.connectors.ownerMissing}</p>
  }

  return (
    <ConnectorOffer
      busy={busy}
      flow={flow}
      key={`${runtimeId}:${signature}`}
      onContinue={async text => {
        if (!owner || !runtimeId) {
          throw new Error('Session unavailable')
        }

        const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'

        if (!requestComposerSubmit(text, { target })) {
          throw new Error('Composer unavailable')
        }
      }}
    />
  )
}

interface ConnectorOfferProps {
  flow: ReturnType<typeof createConnectorFlow>
  busy: boolean
  onContinue: (text: string) => Promise<void>
}

export function ConnectorOffer({ flow, busy, onContinue }: ConnectorOfferProps) {
  const state = useStore(flow.state)
  const { t } = useI18n()
  const copy = t.connectors
  const [query, setQuery] = useState('')
  const [continuing, setContinuing] = useState(false)
  const [continued, setContinued] = useState(false)
  const [continueError, setContinueError] = useState(false)
  const active = state.rows.some(row => row.phase === 'opening' || row.phase === 'waiting')
  const decided = state.rows.some(row => row.phase === 'connected' || row.phase === 'skipped')

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

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-3" data-connector-offer>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">{copy.title}</span>
        <Button onClick={() => void flow.refresh()} size="xs" variant="text">
          {copy.refresh}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">{copy.disclaimer}</p>
      {state.error ? (
        <p className="text-xs text-destructive" role="alert">
          {copy.statusError}
        </p>
      ) : null}
      {!state.available && !state.error ? <p className="text-xs text-muted-foreground">{copy.unavailable}</p> : null}
      {state.rows.length > 6 ? <SearchField onChange={setQuery} placeholder={copy.search} value={query} /> : null}
      <div className="grid max-h-96 min-w-0 gap-3 overflow-y-auto">
        {rows.map(row => (
          <div className="grid gap-1" key={row.connector}>
            <ConnectorCard
              actionDisabled={!state.available || row.enabled === false || !!state.error}
              connector={{
                name: row.connector,
                title: row.name || connectorTitle(row.connector),
                description: row.description
              }}
              copy={{
                ...cardCopy,
                decline: row.phase === 'opening' || row.phase === 'waiting' ? copy.cancel : copy.skip,
                connectAction: ['expired', 'revoked'].includes(row.connectionStatus ?? '') ? copy.grant : copy.connect
              }}
              dismissed={row.phase === 'skipped'}
              onConnect={() => void flow.connect(row.connector)}
              onDismiss={() => flow.skip(row.connector)}
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
            />
            {row.phase === 'timeout' ? (
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span>{copy.timeout}</span>
                <Button onClick={() => void flow.keepWaiting(row.connector)} size="xs" variant="textStrong">
                  {copy.keepWaiting}
                </Button>
              </div>
            ) : null}
          </div>
        ))}
        {!rows.length && state.available ? <p className="text-xs text-muted-foreground">{copy.empty}</p> : null}
      </div>
      {decided && !continued ? (
        <div>
          <Button
            disabled={busy || active || continuing}
            onClick={() => {
              setContinuing(true)
              setContinueError(false)

              const connected = state.rows
                .filter(row => row.phase === 'connected')
                .map(row => connectorTitle(row.connector))

              const skipped = state.rows
                .filter(row => row.phase === 'skipped')
                .map(row => connectorTitle(row.connector))

              void onContinue(
                `Continue the task. Connected apps: ${connected.join(', ') || 'none'}. Continue without: ${skipped.join(', ') || 'none'}. Use current connector status before accessing anything.`
              )
                .then(() => setContinued(true))
                .catch(() => setContinueError(true))
                .finally(() => setContinuing(false))
            }}
            size="sm"
          >
            {busy ? copy.continueBusy : copy.continue}
          </Button>
          {continueError ? (
            <p className="text-xs text-destructive" role="alert">
              {copy.continueFailed}
            </p>
          ) : null}
        </div>
      ) : null}
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
