import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { ConnectionTargetState } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { sessionRoute } from '@/app/routes'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorRow, type ConnectorRowMark, ConnectorSummary } from '@/components/ui/connector-card'
import { useI18n } from '@/i18n'
import {
  connectorAuthorizationUrl,
  connectorCalls,
  connectorIconUrl,
  connectorText,
  connectorTitle,
  connectorToolName,
  recordOf
} from '@/lib/connector-tools'
import {
  $connectionRequests,
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

/** Which backend owns the session whose operation this card drives. */
export interface ConnectorOwner {
  connectionId: null | string
  profile: string
}

/** Resolve that owner for one stored session, or null when it cannot be resolved. */
async function connectionOwnerFor(sessionId: string, method: string): Promise<ConnectorOwner | null> {
  const ambientProfile = $activeGatewayProfile.get()

  try {
    const scope = await resolveSessionOwner(sessionId)
    assertSessionOwnerResolved(scope, { method, sessionId })

    return {
      connectionId: isSessionOwnerRoute(scope) ? scope.connectionId : null,
      profile: isSessionOwnerRoute(scope) ? scope.profile : scope || ambientProfile
    }
  } catch {
    return null
  }
}

/** Resolve that owner. Null until it resolves and null when it cannot: a card RPC must reach the
 *  gateway that holds the operation, never whichever one the window happens to have in front. */
export function useConnectionOwner(sessionId: null | string, active: boolean): ConnectorOwner | null {
  const [owner, setOwner] = useState<ConnectorOwner | null>(null)

  useEffect(() => {
    if (!sessionId || !active) {
      setOwner(null)

      return
    }

    let cancelled = false

    void connectionOwnerFor(sessionId, 'connectors.connect').then(resolved => {
      if (!cancelled) {
        setOwner(resolved)
      }
    })

    return () => {
      cancelled = true
    }
  }, [active, sessionId])

  return owner
}

/** The browser leg of a connection came back through `hermes://connections/done`. Show the session
 *  that opened the operation and tell its backend to read the account now instead of at its next
 *  tick. Nothing in the link is trusted to move a row: the op id only names which card to show, and
 *  the backend reads the account itself. An operation this window holds no card for, or one that
 *  already settled, is ignored: the tab can come back long after Continue, and a stale link must
 *  not pull the user away from where they are. */
export async function openConnectionDoneLink(
  op: string,
  navigate: (to: string) => void,
  storedSessionIdFor: (runtimeSessionId: string) => string
): Promise<void> {
  const request = Object.values($connectionRequests.get()).find(entry => entry.opId === op)

  if (!request?.sessionId || request.settled) {
    return
  }

  const storedId = storedSessionIdFor(request.sessionId)
  navigate(sessionRoute(storedId))

  const owner = await connectionOwnerFor(storedId, 'connectors.operation.wake')

  if (!owner) {
    return
  }

  try {
    await requestGatewayForAgent(owner.connectionId, owner.profile, 'connectors.operation.wake', {
      op_id: op,
      session_id: request.sessionId
    })
  } catch {
    // The wake only shortens the wait. The operation can settle and leave the live registry between
    // the link and this RPC (4004); the watcher reads the account at its next tick regardless.
  }
}

/** Try again for one target of the open operation: one RPC, and the fresh link when the backend
 *  minted one. The backend re-mints only what is actually dead. */
export async function reissueConnectionTarget(
  owner: ConnectorOwner,
  request: ConnectionRequest,
  name: string
): Promise<null | string> {
  const reply = await requestGatewayForAgent<ToolCallMessagePartProps['result']>(
    owner.connectionId,
    owner.profile,
    'connectors.connect',
    {
      connectors: [name],
      reconnect: true,
      session_id: request.sessionId
    },
    45000
  )

  const rows = recordOf(reply).targets
  const minted = Array.isArray(rows) ? rows.map(recordOf).find(row => connectorText(row.name) === name) : undefined

  return connectorAuthorizationUrl(minted?.connect_url)
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
  const owner = useConnectionOwner(storedId, live)

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

export const CONNECTOR_CARD_PHASES = {
  connected: { mark: 'connected', resolved: true, settled: connected, verb: 'none' },
  expired: { mark: 'idle', resolved: false, settled: notConnected, verb: 'reissue' },
  failed: { mark: 'idle', resolved: false, settled: notConnected, verb: 'reissue' },
  initiated: { mark: 'waiting', resolved: false, settled: notConnected, verb: 'open' },
  not_connected: { mark: 'idle', resolved: false, settled: notConnected, verb: 'none' },
  pending: { mark: 'idle', resolved: false, settled: notConnected, verb: 'open' },
  skipped: { mark: 'idle', resolved: true, settled: skipped, verb: 'none' },
  unavailable: { mark: 'idle', resolved: true, settled: notConnected, verb: 'none' }
} satisfies Record<ConnectionTargetState, ConnectorCardPhase>

// A disabled verb (a working row, a waiting row with no link yet) refuses focus, and the keyboard
// would land on the document body; so the first control that can take it, else the row itself.
const FOCUSABLE_IN_ROW = 'button:not([disabled]), [href], input:not([disabled])'
// The user is typing a credential; a row moving elsewhere on the card must not take the keyboard.
const EDITABLE = 'input, textarea, select, [contenteditable]:not([contenteditable="false"])'

function focusChangedRow(card: HTMLElement, name: string): void {
  const row = [...card.querySelectorAll<HTMLElement>('[data-connector-row]')].find(
    node => node.dataset.connectorRow === name
  )

  ;(row?.querySelector<HTMLElement>(FOCUSABLE_IN_ROW) ?? row)?.focus()
}

/** Move focus to the row the backend changed. Only while the card already holds focus, and never
 *  out of a field the user is typing in — a transition the user is not looking at must not take
 *  the keyboard away from wherever they are. */
export function useConnectorFocusHandoff(
  targets: readonly ConnectionTarget[],
  cardRef: RefObject<HTMLDivElement | null>
): void {
  const seen = useRef<Map<string, ConnectionTargetState> | null>(null)
  const states = targets.map(target => `${target.name}=${target.state}`).join('|')

  // The ref holds what the last frame said, for comparison only: nothing renders from it, so it
  // cannot lag a render the way a mirrored atom would.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    const previous = seen.current
    seen.current = new Map(targets.map(target => [target.name, target.state]))

    const card = cardRef.current

    const moved = targets.find(target => {
      const before = previous?.get(target.name)

      return before !== undefined && before !== target.state
    })

    const active = document.activeElement

    if (!previous || !moved || !card?.contains(active) || active?.matches(EDITABLE)) {
      return
    }

    focusChangedRow(card, moved.name)
    // The target states are the whole input; `states` changes exactly when one of them moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [states])
}

export const MARK_LABEL = {
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
  // A DOM handle for the focus handoff, never rendered state.
  const cardRef = useRef<HTMLDivElement | null>(null)

  useConnectorFocusHandoff(request.targets, cardRef)

  // The update frame paints the row as waiting with the fresh link; the user opens it from the row.
  // A refused re-mint is a click that changed nothing, so it gets a toast; the row stays as it was.
  const reissue = async (target: ConnectionTarget): Promise<void> => {
    setReissuing(current => new Set(current).add(target.name))

    try {
      await reissueConnectionTarget(owner, request, target.name)
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
              connector={{
                iconUrl: connectorIconUrl(target.name),
                name: target.name,
                title: connectorTitle(target.name)
              }}
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
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer ref={cardRef}>
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
              connector={{
                iconUrl: connectorIconUrl(target.name),
                name: target.name,
                title: connectorTitle(target.name)
              }}
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
