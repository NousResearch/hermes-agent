import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { ConnectionTargetState } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorRow, type ConnectorRowMark, ConnectorSummary } from '@/components/ui/connector-card'
import { useI18n } from '@/i18n'
import {
  connectorAuthorizationUrl,
  connectorCalls,
  connectorText,
  connectorTitle,
  connectorToolName,
  recordOf
} from '@/lib/connector-tools'
import {
  type ConnectionRequest,
  type ConnectionTarget,
  continueConnectionRequest,
  sessionConnectionRequest
} from '@/store/connection-request'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

interface ConnectorOwner {
  connectionId: null | string
  profile: string
}

/** Names requested by a manage_connections part, including an event-projected row. */
function requestedConnectorNames(args: ToolCallMessagePartProps['args']): string[] {
  const connectors = recordOf(args).connectors
  const entries = Array.isArray(connectors) ? connectors : [connectors]

  return entries.flatMap(entry => {
    const row = recordOf(entry)
    const name = connectorText(entry) ?? connectorText(row.name) ?? connectorText(row.connector)
    const trimmed = name?.trim()

    return trimmed ? [trimmed] : []
  })
}

function matchingTargetNames(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false
  }

  const leftSorted = [...left].sort()
  const rightSorted = [...right].sort()

  return leftSorted.every((name, index) => name === rightSorted[index])
}

/** The card lives on the tool row whose id opened the operation and on no other. */
export function connectionRequestOwnsPart(props: ToolCallMessagePartProps, request: ConnectionRequest | null): boolean {
  return Boolean(request && props.toolCallId === request.toolCallId)
}

export function ConnectorTool(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const $request = useMemo(() => sessionConnectionRequest(runtimeId), [runtimeId])
  const request = useStore($request)
  const targetNames = requestedConnectorNames(props.args)

  const untargetedStatus =
    props.toolName === 'manage_connections' &&
    (recordOf(props.args).action ?? 'status') === 'status' &&
    targetNames.length === 0

  const live = !untargetedStatus && connectionRequestOwnsPart(props, request)
  // Owner routes and hints are keyed by the stored id, not the runtime id the events carry.
  const ownerSessionId = storedId
  const [owner, setOwner] = useState<ConnectorOwner | null>(null)

  useEffect(() => {
    if (!ownerSessionId || !live) {
      setOwner(null)

      return
    }

    let cancelled = false
    const ambientProfile = $activeGatewayProfile.get()

    void resolveSessionOwner(ownerSessionId)
      .then(scope => {
        assertSessionOwnerResolved(scope, { method: 'connectors.connect', sessionId: ownerSessionId })

        if (!cancelled) {
          setOwner({
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
  }, [live, ownerSessionId])

  if (!live || !request) {
    return <ToolFallback {...props} />
  }

  return owner ? <ConnectorOffer owner={owner} request={request} /> : null
}

type ConnectorCopy = ReturnType<typeof useI18n>['t']['connectors']
type ConnectorVerb = 'none' | 'open' | 'reissue'

/** The settled row's word; the card never says why. */
interface SettledWord {
  meta: string
  tone?: 'ok'
}

interface ConnectorCardPhase {
  mark: ConnectorRowMark
  resolved: boolean
  settled: (copy: ConnectorCopy) => SettledWord
  verb: ConnectorVerb
}

const connected = (copy: ConnectorCopy): SettledWord => ({ meta: copy.connected, tone: 'ok' })
const notConnected = (copy: ConnectorCopy): SettledWord => ({ meta: copy.notConnected })
const skipped = (copy: ConnectorCopy): SettledWord => ({ meta: copy.skipped })

const CONNECTOR_CARD_PHASES = {
  connected: { mark: 'connected', resolved: true, settled: connected, verb: 'none' },
  expired: { mark: 'idle', resolved: false, settled: notConnected, verb: 'reissue' },
  failed: { mark: 'idle', resolved: false, settled: notConnected, verb: 'reissue' },
  initiated: { mark: 'waiting', resolved: false, settled: notConnected, verb: 'open' },
  not_connected: { mark: 'idle', resolved: false, settled: notConnected, verb: 'none' },
  pending: { mark: 'idle', resolved: false, settled: notConnected, verb: 'open' },
  skipped: { mark: 'idle', resolved: true, settled: skipped, verb: 'none' },
  unavailable: { mark: 'idle', resolved: true, settled: notConnected, verb: 'none' }
} satisfies Record<ConnectionTargetState, ConnectorCardPhase>

const MARK_LABEL = {
  connected: (copy: ConnectorCopy) => copy.connected,
  idle: (copy: ConnectorCopy) => copy.notConnected,
  waiting: (copy: ConnectorCopy) => copy.waiting
} satisfies Record<ConnectorRowMark, (copy: ConnectorCopy) => string>

interface ConnectorOfferProps {
  owner: ConnectorOwner
  request: ConnectionRequest
}

export function ConnectorOffer({ owner, request }: ConnectorOfferProps) {
  const { t } = useI18n()
  const copy = t.connectors
  const [reissuing, setReissuing] = useState<ReadonlySet<string>>(new Set())
  const unresolved = request.targets.some(target => !CONNECTOR_CARD_PHASES[target.state].resolved)

  // Try again is one RPC on the open operation; the backend re-mints only a dead link. The fresh link
  // opens at once, and the update frame then paints the row as waiting. A refused re-mint is a click
  // that changed nothing, so it gets a toast; the row stays as it was.
  const reissue = async (target: ConnectionTarget): Promise<void> => {
    setReissuing(current => new Set(current).add(target.name))

    try {
      const reply = await requestGatewayForAgent<ToolCallMessagePartProps['result']>(
        owner.connectionId,
        owner.profile,
        'connectors.connect',
        {
          connectors: [target.name],
          reconnect: true,
          session_id: request.sessionId
        },
        45000
      )

      const rows = recordOf(reply).targets
      const minted = Array.isArray(rows) ? rows.map(recordOf).find(row => connectorText(row.name) === target.name) : undefined
      const url = connectorAuthorizationUrl(minted?.connect_url)

      if (url) {
        void window.hermesDesktop?.openExternal?.(url)
      }
    } catch (error) {
      notifyError(error, copy.connectErrorFor(connectorTitle(target.name)))
    } finally {
      setReissuing(current => {
        const next = new Set(current)
        next.delete(target.name)

        return next
      })
    }
  }

  // A settled operation is a static per-target summary: no controls, no polling, nothing live.
  if (request.settled) {
    return (
      <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer>
        {request.targets.map(target => {
          const { meta, tone } = CONNECTOR_CARD_PHASES[target.state].settled(copy)

          return (
            <ConnectorSummary
              connector={{ name: target.name, title: connectorTitle(target.name) }}
              key={target.name}
              meta={meta}
              tone={tone}
            />
          )
        })}
      </div>
    )
  }

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer>
      <ConnectorCard title={copy.title}>
        {request.targets.map(target => {
          const phase = CONNECTOR_CARD_PHASES[target.state]
          const busy = reissuing.has(target.name)

          const action =
            phase.verb === 'none'
              ? undefined
              : {
                  busy,
                  // Prevent concurrent sign-in tabs; a waiting row without a link has nothing to open yet.
                  disabled: (reissuing.size > 0 && !busy) || (phase.verb === 'open' && target.connectUrl === null),
                  label: phase.verb === 'open' ? copy.connect : copy.retry,
                  onClick: () => {
                    if (phase.verb === 'open' && target.connectUrl && window.hermesDesktop?.openExternal) {
                      void window.hermesDesktop.openExternal(target.connectUrl)
                    }

                    if (phase.verb === 'reissue') {
                      void reissue(target)
                    }
                  }
                }

          return (
            <ConnectorRow
              action={action}
              connector={{ name: target.name, title: connectorTitle(target.name) }}
              cue={phase.mark === 'waiting' ? copy.waiting : undefined}
              key={target.name}
              mark={phase.mark}
              markLabel={MARK_LABEL[phase.mark](copy)}
            />
          )
        })}
      </ConnectorCard>
      {unresolved ? (
        <div className="px-3.5">
          <Button onClick={() => void continueConnectionRequest(request)} size="xs" variant="textStrong">
            {t.common.continue}
          </Button>
        </div>
      ) : null}
    </div>
  )
}

/** Keep execution output in the standard disclosure, with one row per app call. */
export function ConnectorExecution(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const $request = useMemo(() => sessionConnectionRequest(sessionId), [sessionId])
  const request = useStore($request)
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

      return recordOf(item.error).connect_card_available === true
    })
    .map(call => {
      // SAFETY: connectorCalls includes only names accepted by connectorToolName.
      return connectorToolName(call.name)!.connector
    })

  const openRepair =
    request &&
    !request.settled &&
    matchingTargetNames(
      repair,
      request.targets.map(target => target.name)
    )

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
      {openRepair ? (
        <ConnectorTool {...props} args={{ action: 'status', connectors: repair }} result={undefined} />
      ) : null}
    </>
  )
}
