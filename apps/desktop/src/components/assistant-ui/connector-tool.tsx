import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { ConnectionTargetState, ConnectorsConnectResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'
import { useInRouterContext, useNavigate } from 'react-router'

import { accountRenameRoute } from '@/app/capabilities/connectors/data/deep-link'
import { useSessionView } from '@/app/chat/session-view'
import { sessionRoute } from '@/app/routes'
import { ConnectorDirectoryCard } from '@/components/assistant-ui/connector-directory-card'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { TypedAnswer } from '@/components/assistant-ui/typed-answer'
import { useConnectionOwner } from '@/components/assistant-ui/use-connection-owner'
import { SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorRow, type ConnectorRowMark, ConnectorSummary } from '@/components/ui/connector-card'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { useI18n } from '@/i18n'
import { presentAuthorizationUrl } from '@/lib/authorization-url'
import {
  connectorAuthorizationUrl,
  connectorIconUrl,
  connectorText,
  connectorTitle,
  recordOf,
  toolLabels,
  toolLabelTitle
} from '@/lib/connector-tools'
import {
  $connectionRequests,
  connectionOpOf,
  type ConnectionOwner,
  connectionOwnerFor,
  type ConnectionRequest,
  connectionRequestOpen,
  type ConnectionTarget,
  continueConnectionRequest,
  toolConnectionRequest
} from '@/store/connection-request'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

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
      owner: { session_id: request.sessionId, type: 'session' }
    })
  } catch {
    // The wake only shortens the wait. The operation can settle and leave the live registry between
    // the link and this RPC (4004); the watcher reads the account at its next tick regardless.
  }
}

/** Try again for one target of the open operation: one RPC, and the fresh link when the backend
 *  minted one. The backend re-mints only what is actually dead. A settled operation is dead: the
 *  RPC would open a second one that no card on this row can answer. */
export async function reissueConnectionTarget(
  owner: ConnectionOwner,
  request: ConnectionRequest,
  name: string
): Promise<null | string> {
  if (!connectionRequestOpen(request)) {
    return null
  }

  const reply = await requestGatewayForAgent<ConnectorsConnectResult>(
    owner.connectionId,
    owner.profile,
    'connectors.connect',
    {
      connectors: [name],
      owner: { session_id: request.sessionId, type: 'session' },
      reconnect: true
    },
    45000
  )

  const minted = reply.targets.find(target => target.name === name)

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

/** The card lives on the tool row whose id opened the operation and on no other. */
export function connectionRequestOwnsPart(props: ToolCallMessagePartProps, request: ConnectionRequest | null): boolean {
  return Boolean(request && props.toolCallId === request.toolCallId)
}

export function ConnectorTool(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const opId = connectionOpOf(props.args)
  const running = props.result === undefined

  const $request = useMemo(
    () => toolConnectionRequest(runtimeId, props.toolCallId, opId, running),
    [opId, props.toolCallId, running, runtimeId]
  )

  const request = useStore($request)
  const targetNames = requestedConnectorNames(props.args)

  const untargetedStatus =
    props.toolName === 'manage_connections' &&
    (recordOf(props.args).action ?? 'status') === 'status' &&
    targetNames.length === 0

  const live = !untargetedStatus && connectionRequestOwnsPart(props, request)
  // Owner routes and hints are keyed by the stored id, not the runtime id the events carry.
  const owner = useConnectionOwner(storedId, live)

  if (recordOf(props.args).action === 'display') {
    return <ConnectorDirectoryTool live={live} part={props} request={request} />
  }

  if (!live || !request) {
    return <ToolFallback {...props} />
  }

  return owner ? <ConnectorOffer owner={owner} request={request} /> : null
}

interface ConnectorDirectoryToolProps {
  live: boolean
  part: ToolCallMessagePartProps
  request: ConnectionRequest | null
}

/** The display card while its operation is open; afterwards the one line the result says, or the generic row
 *  for a call that failed. */
function ConnectorDirectoryTool({ live, part, request }: ConnectorDirectoryToolProps) {
  if (live && request && !request.settled) {
    return (
      <ConnectorDirectoryCard
        onContinue={() => void continueConnectionRequest(request)}
        suggested={request.display?.suggested}
      />
    )
  }

  if (part.result === undefined) {
    return null
  }

  return part.isError || recordOf(part.result).error !== undefined ? (
    <ToolFallback {...part} />
  ) : (
    <ConnectorDirectoryReceipt result={part.result} />
  )
}

// A timed-out card still names what the user connected before walking away.
function receiptText(copy: ConnectorCopy['directory'], outcome: ReturnType<typeof recordOf>, added: string[]): string {
  if (added.length > 0) {
    return copy.receiptConnected(added.map(connectorTitle).join(', '))
  }

  return outcome.timed_out === true ? copy.receiptTimedOut : copy.receiptNone
}

export function ConnectorDirectoryReceipt({ result }: { result: unknown }) {
  const { t } = useI18n()
  const copy = t.connectors.directory
  const outcome = recordOf(result)

  const added = Array.isArray(outcome.newly_connected)
    ? outcome.newly_connected.flatMap(name => connectorText(name) ?? [])
    : []

  const said = connectorText(outcome.said)?.trim()

  return (
    <>
      {added.length > 0 || !said ? (
        <div data-conversation-scaffold="" data-slot="connector-directory-receipt">
          <ScaffoldRow>
            {added.map(name => (
              <ConnectorLogo
                className="size-4 rounded-[0.25rem]"
                connector={{ iconUrl: connectorIconUrl(name), name, title: connectorTitle(name) }}
                key={name}
              />
            ))}
            <span className={SCAFFOLD_LABEL_CLASS}>{receiptText(copy, outcome, added)}</span>
          </ScaffoldRow>
        </div>
      ) : null}
      {said ? <TypedAnswer text={said} /> : null}
    </>
  )
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
  skipped: { mark: 'idle', resolved: true, settled: skipped, verb: 'none' }
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
  owner: ConnectionOwner
  request: ConnectionRequest
}

export function ConnectorOffer({ owner, request }: ConnectorOfferProps) {
  const { t } = useI18n()
  const copy = t.connectors
  const [reissuing, setReissuing] = useState<ReadonlySet<string>>(new Set())
  // Rows whose link the user opened from this card. `initiated` only means a link was minted: the
  // connect-first handoff (D85) mints on the watcher's first pass, before anyone clicks, so the
  // "Waiting for your browser…" cue belongs to a row the user actually opened.
  const [opened, setOpened] = useState<ReadonlySet<string>>(new Set())
  const unresolved = request.targets.some(target => !CONNECTOR_CARD_PHASES[target.state].resolved)
  // A DOM handle for the focus handoff, never rendered state.
  const cardRef = useRef<HTMLDivElement | null>(null)

  useConnectorFocusHandoff(request.targets, cardRef)

  // The update frame paints the row as waiting with the fresh link; the user opens it from the row.
  // A refused re-mint is a click that changed nothing, so it gets a toast; the row stays as it was.
  const reissue = async (target: ConnectionTarget): Promise<void> => {
    setReissuing(current => new Set(current).add(target.name))
    setOpened(current => {
      const next = new Set(current)
      next.delete(target.name)

      return next
    })

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
        <NamedAccountNotices owner={owner} targets={request.targets} />
      </div>
    )
  }

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer ref={cardRef}>
      <ConnectorCard title={copy.title}>
        {request.targets.map(target => {
          const phase = CONNECTOR_CARD_PHASES[target.state]
          const busy = reissuing.has(target.name)
          const mark = phase.mark === 'waiting' && !opened.has(target.name) ? 'idle' : phase.mark

          const action =
            phase.verb === 'none'
              ? undefined
              : {
                  busy,
                  // Prevent concurrent sign-in tabs; a waiting row without a link has nothing to open yet.
                  disabled: (reissuing.size > 0 && !busy) || (phase.verb === 'open' && target.connectUrl === null),
                  label: phase.verb === 'open' ? copy.connect : copy.retry,
                  onClick: () => {
                    if (phase.verb === 'open' && target.connectUrl) {
                      presentAuthorizationUrl(target.connectUrl)
                      setOpened(current => new Set(current).add(target.name))
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
              cue={mark === 'waiting' ? copy.waiting : undefined}
              key={target.name}
              mark={mark}
              markLabel={MARK_LABEL[mark](copy)}
            />
          )
        })}
      </ConnectorCard>
      <NamedAccountNotices owner={owner} targets={request.targets} />
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

/** One line per account Hermes named and connected; Rename opens that account's editor on the Connectors tab. */
function NamedAccountNotices({ owner, targets }: { owner: ConnectionOwner; targets: readonly ConnectionTarget[] }) {
  const named = targets.filter(target => target.state === 'connected' && target.alias)

  if (named.length === 0) {
    return null
  }

  return (
    <div className="grid gap-1 px-3.5" data-slot="connector-named-account">
      {named.map(target => (
        <NamedAccountNotice
          alias={target.alias ?? ''}
          connectionId={target.connectionId}
          key={target.name}
          name={target.name}
          owner={owner}
        />
      ))}
    </div>
  )
}

interface NamedAccountProps {
  alias: string
  /** The account the line is about; the Rename link goes by it, since the name can change hands. */
  connectionId: string
  name: string
  owner: ConnectionOwner
}

function NamedAccountNotice({ alias, connectionId, name, owner }: NamedAccountProps) {
  const { t } = useI18n()
  const inRouter = useInRouterContext()

  return (
    <p className="flex flex-wrap items-center gap-x-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
      {t.connectors.namedAccount(connectorTitle(name), alias)}
      {inRouter && connectionId ? (
        <RenameAccountLink alias={alias} connectionId={connectionId} name={name} owner={owner} />
      ) : null}
    </p>
  )
}

// useNavigate() throws outside a Router; the parent mounts this only inside one.
function RenameAccountLink({ connectionId, name, owner }: NamedAccountProps) {
  const { t } = useI18n()
  const navigate = useNavigate()

  return (
    <Button onClick={() => navigate(accountRenameRoute(name, connectionId, owner))} size="inline" variant="textStrong">
      {t.connectors.renameAccount}
    </Button>
  )
}

const MISSING_CALL_RESULT = { error: 'No result for this call.' }

/** Keep execution output in the standard disclosure, with one row per inner call.
 *  The gateway labels every call the tool_search bridge runs — hosted, MCP or local —
 *  so hosted-only, MCP-only and mixed batches all render the same way. */
export function ConnectorExecution(props: ToolCallMessagePartProps) {
  const labels = toolLabels(props.args)
  const output = recordOf(props.result)
  const results = Array.isArray(output.results) ? output.results : []

  if (labels.length === 0) {
    return <ToolFallback {...props} />
  }

  const input = recordOf(props.args)
  const batch = Array.isArray(input.calls) ? input.calls : [input]

  return (
    <>
      {labels.map((label, index) => {
        // A hosted batch answers one result per call; anything else answers once for the
        // whole call, and every row shows that same outcome (a rejected batch, an error).
        // A batch that answered short says so on the rows it left out.
        const item = results[index] ?? (results.length > 0 ? MISSING_CALL_RESULT : props.result)
        const result = recordOf(item)

        return (
          <ToolFallback
            {...props}
            args={recordOf(recordOf(batch[index]).arguments ?? props.args)}
            innerToolName={props.toolName === 'tool_call' ? label.name : props.toolName}
            isError={Boolean(result.error) || props.isError === true}
            key={`${props.toolCallId}:${index}`}
            result={item}
            toolCallId={`${props.toolCallId}:${index}`}
            toolName={toolLabelTitle(label)}
          />
        )
      })}
    </>
  )
}
