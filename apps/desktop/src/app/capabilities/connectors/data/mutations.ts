// The page's writes.
//
// Every one of them answers a typed outcome instead of throwing: the container
// decides what a failure looks like, because the words belong to i18n and this
// folder holds none. Nothing here paints before the backend answers except the
// tool editor, which owns its own optimism (`useToolsEditor`) — Connect never
// does, and neither does the connector switch.

import type { ConnectionAnswer, ConnectorPolicyGetResult } from '@hermes/shared'
import { useCallback, useRef, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'

import type { SaveResult } from '../use-tools-editor'

import { type AccountOperation, clearAccountOperation, startAccountOperation } from './account-operations'
import { memberDisabledTools, memberRevision, readPolicy } from './join'
import { connectorsAccountsQueryKey, connectorsListQueryKey, connectorsPolicyQueryKey } from './keys'
import {
  asConnectorError,
  connectAccountConnectors,
  connectorPolicy,
  type ConnectorRpcError,
  isConnectorReason,
  removeConnectorAccount,
  respondToAccountOperation,
  setConnectorPolicy,
  wakeAccountOperation
} from './rpc'

/** A write either landed or it did not. The error is typed so the caller can
 *  tell "signed out" from "the portal is down" without reading a message. */
export type WriteOutcome = { error: ConnectorRpcError; ok: false } | { ok: true }

const failed = (error: unknown) => ({ error: asConnectorError(error), ok: false as const })

/** The policy is the source of every "off" on this page; a write to it makes the
 *  hosted list's own flags stale too. */
function refetchPolicy(scope: ProfileScope): void {
  void queryClient.invalidateQueries({ queryKey: connectorsPolicyQueryKey(scope) })
  void queryClient.invalidateQueries({ queryKey: connectorsListQueryKey(scope) })
}

// ── the compare-and-set token ──────────────────────────────────────────────
//
// `connectors.policy.set` answers the revision it produced, and the refetch that
// would put it on screen is a slow call. So the token for the next write is that
// answer, not the revision the screen still shows — sending the old one makes
// the backend report the person's own previous write as somebody else's change.

/** What a writer last got back, remembered against the revision it superseded.
 *  It is the token until a fresh read moves `seen` past it. */
interface WrittenRevision {
  after: string | undefined
  revision: string
}

const tokenFor = (written: null | WrittenRevision, seen: string | undefined): string | undefined =>
  written && written.after === seen ? written.revision : seen

const remember = (seen: string | undefined, revision: string | undefined): null | WrittenRevision =>
  revision === undefined ? null : { after: seen, revision }

// ── the tool editor's compare-and-set write ────────────────────────────────

export interface ConnectorToolsSaver {
  /** Feed straight into `useToolsEditor({ onSave })`. The editor's `overwrite`
   *  option is not read: the token is always the freshest revision this window
   *  knows, which after a conflict is the one that would win. */
  onSave: (disabled: string[]) => Promise<SaveResult>
  /** The conflict view's Reload: drop this editor's work and take their version. */
  reload: () => void
  /** The rule the other editor saved, set only while a conflict is unresolved.
   *  `conflictDifference(theirs, mine)` turns it into the sentence. */
  theirs: string[] | null
}

/**
 * Save one connector's tool rule.
 *
 * `seenRevision` is the member layer's revision as it was when the list was
 * painted. It is the compare-and-set token only until this editor writes: from
 * then on the token is what the backend answered (`tokenFor`), because the read
 * that would refresh the screen lands long after the next Save. A
 * losing write comes back `POLICY_CONFLICT`; this hook then reads the policy
 * once more to learn both what the other editor saved and the revision that
 * would win, and answers `conflict`. "Save over their version" re-sends the
 * same list against that fresher revision.
 */
export function useConnectorToolsSave(
  scope: ProfileScope,
  connector: string,
  seenRevision: string | undefined
): ConnectorToolsSaver {
  const [theirs, setTheirs] = useState<string[] | null>(null)
  // Written and read inside one callback and never rendered, so it is a ref: as
  // state it would re-render the editor mid-save and it is not a fact the screen
  // shows.
  const written = useRef<null | WrittenRevision>(null)

  const reload = useCallback(() => {
    setTheirs(null)
    // The remembered revision is left alone: it is the one that would win, and
    // the refetch that replaces the baseline supersedes it on its own.
    void queryClient.invalidateQueries({ queryKey: connectorsPolicyQueryKey(scope) })
  }, [scope])

  const onSave = useCallback(
    // `overwrite` no longer picks the token: "Save over their version" and an
    // ordinary Save both send the freshest revision this window knows, which
    // after a conflict IS the one the other editor produced. What overwrite
    // changes is that the editor asked for the write at all.
    async (disabled: string[]): Promise<SaveResult> => {
      const expected = tokenFor(written.current, seenRevision)

      try {
        const result = await setConnectorPolicy(scope, { connector, disabled_tools: disabled, type: 'tools' }, expected)

        written.current = remember(seenRevision, result.revision)
      } catch (error) {
        if (!isConnectorReason(error, 'POLICY_CONFLICT')) {
          return 'failed'
        }

        try {
          // Read WITHOUT writing the cache. The cached member rule is the
          // editor's baseline, and replacing it would reset the editor out of
          // the very conflict view it is about to show.
          const policy = readPolicy((await connectorPolicy(scope)).layers)

          written.current = remember(seenRevision, memberRevision(policy))
          setTheirs([...memberDisabledTools(policy, connector)])

          return 'conflict'
        } catch {
          // Their version could not be read, so there is no conflict to show —
          // only a save that did not happen, with the work still on screen.
          return 'failed'
        }
      }

      setTheirs(null)
      refetchPolicy(scope)

      return 'saved'
    },
    [connector, scope, seenRevision]
  )

  return { onSave, reload, theirs }
}

// ── the connector's own switch ─────────────────────────────────────────────

export interface ConnectorSwitch {
  /** The connector whose write is in flight, for the disabled switch. */
  pending: null | string
  setEnabled: (connector: string, enabled: boolean) => Promise<WriteOutcome>
}

/** "Off for you" / "Turn back on", and the dialog's own switch. */
export function useConnectorSwitch(scope: ProfileScope): ConnectorSwitch {
  const [pending, setPending] = useState<null | string>(null)
  // Two quick flips of the same connector are one write after another, and the
  // first one's revision is the only token the second can win with.
  const written = useRef<null | WrittenRevision>(null)

  const setEnabled = useCallback(
    async (connector: string, enabled: boolean): Promise<WriteOutcome> => {
      setPending(connector)

      try {
        // One fact, no editor to conflict against, so the token is simply the
        // freshest revision the page holds: what this window last wrote, or the
        // cached read before it has written. Sending none would make the write
        // unconditional, which is not what a compare-and-set API is for.
        const cached = queryClient.getQueryData<ConnectorPolicyGetResult>(connectorsPolicyQueryKey(scope))
        const seen = cached ? memberRevision(readPolicy(cached.layers)) : undefined

        const result = await setConnectorPolicy(
          scope,
          { connector, enabled, type: 'connector' },
          tokenFor(written.current, seen)
        )

        written.current = remember(seen, result.revision)
        refetchPolicy(scope)

        return { ok: true }
      } catch (error) {
        return failed(error)
      } finally {
        setPending(null)
      }
    },
    [scope]
  )

  return { pending, setEnabled }
}

// ── disconnect ─────────────────────────────────────────────────────────────

export interface AccountDisconnect {
  disconnect: (connectionId: string) => Promise<WriteOutcome>
  pending: boolean
}

/** Forget one account. The caller asks through `ConfirmDialog` first — this hook
 *  only performs it. */
export function useDisconnectAccount(scope: ProfileScope): AccountDisconnect {
  const [pending, setPending] = useState(false)

  const disconnect = useCallback(
    async (connectionId: string): Promise<WriteOutcome> => {
      setPending(true)

      try {
        await removeConnectorAccount(scope, connectionId)
        void queryClient.invalidateQueries({ queryKey: connectorsAccountsQueryKey(scope) })
        void queryClient.invalidateQueries({ queryKey: connectorsListQueryKey(scope) })

        return { ok: true }
      } catch (error) {
        return failed(error)
      } finally {
        setPending(false)
      }
    },
    [scope]
  )

  return { disconnect, pending }
}

// ── connect ────────────────────────────────────────────────────────────────

export type ConnectOutcome = { error: ConnectorRpcError; ok: false } | { ok: true; operation: AccountOperation }

export interface ConnectorConnect {
  /** Connect, or Try again / Reconnect with `reconnect`. Answers the operation
   *  the connect element renders; later facts arrive as `connection.update`
   *  frames, never from here. */
  connect: (slug: string, options?: { reconnect?: boolean }) => Promise<ConnectOutcome>
  /** Stop waiting: end the operation now and forget it. */
  giveUp: (opId: string) => Promise<WriteOutcome>
  /** The connector whose connect is in flight. */
  pending: null | string
  /** The connect element's per-target answer. */
  respond: (opId: string, answer: ConnectionAnswer) => Promise<WriteOutcome>
  /** The return from the browser: look at the operation now. */
  wake: (opId: string) => Promise<WriteOutcome>
}

export function useConnectConnector(scope: ProfileScope): ConnectorConnect {
  const [pending, setPending] = useState<null | string>(null)

  const connect = useCallback(
    async (slug: string, options?: { reconnect?: boolean }): Promise<ConnectOutcome> => {
      setPending(slug)

      try {
        const snapshot = await connectAccountConnectors(scope, [slug], options?.reconnect ?? false)

        return { ok: true, operation: startAccountOperation(scope, [slug], snapshot) }
      } catch (error) {
        return failed(error)
      } finally {
        setPending(null)
      }
    },
    [scope]
  )

  const respond = useCallback(
    async (opId: string, answer: ConnectionAnswer): Promise<WriteOutcome> => {
      try {
        await respondToAccountOperation(scope, opId, answer)

        return { ok: true }
      } catch (error) {
        return failed(error)
      }
    },
    [scope]
  )

  const giveUp = useCallback(
    async (opId: string): Promise<WriteOutcome> => {
      const outcome = await respond(opId, { settled_by: 'continue' })

      // The card is gone either way: a refused Continue still means the person
      // asked to stop looking at it, and the operation settles at its deadline.
      clearAccountOperation(opId)

      return outcome
    },
    [respond]
  )

  const wake = useCallback(
    async (opId: string): Promise<WriteOutcome> => {
      try {
        await wakeAccountOperation(scope, opId)

        return { ok: true }
      } catch (error) {
        return failed(error)
      }
    },
    [scope]
  )

  return { connect, giveUp, pending, respond, wake }
}
