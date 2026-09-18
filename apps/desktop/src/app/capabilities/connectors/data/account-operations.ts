// The connect operations this window started or joined, keyed by `op_id`.
//
// A chat's connector card lives in `store/connection-request.ts`, keyed by
// SESSION. This page has no session: the backend tags every `connection.update`
// frame with its owner, and account-owned frames arrive on the session-less
// broadcast path, so they are matched by operation id alone. The two stores
// share the frame vocabulary (`parseConnectionTarget`) and nothing else — the
// chat store also owns a tool call, a transcript row and a needs-input flag,
// none of which exist here.
//
// Connect is never optimistic. Nothing enters this store until the backend has
// answered `connectors.connect` with a real operation, and every later fact
// comes from a frame the backend sent.

import type {
  ConnectionOperationStatus,
  ConnectionSettleReason,
  ConnectionUpdatePayload,
  ConnectorsConnectResult
} from '@hermes/shared'
import { atom } from 'nanostores'

import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { type ConnectionTarget, parseConnectionTarget } from '@/store/connection-request'

import { connectorsAccountsQueryKey, connectorsListQueryKey } from './keys'
import { accountOperationStatus } from './rpc'

export interface AccountOperation {
  /** The slugs this window asked for, in the order it asked. */
  connectors: string[]
  /** Unix seconds; backend-owned. */
  deadlineAt: number
  opId: string
  /** The scope the operation was started under. A settle refetches THAT scope's
   *  reads — the person may have moved the selector since they pressed Connect. */
  scope: ProfileScope
  /** The newest frame applied. An older frame for the same operation is dropped:
   *  the transport can reorder them and an older one would regress a target. */
  seq: number
  settled: boolean
  settledBy: ConnectionSettleReason | null
  targets: ConnectionTarget[]
}

export const $accountOperations = atom<Readonly<Record<string, AccountOperation>>>({})

const parseTargets = (targets: ConnectionUpdatePayload['targets']): ConnectionTarget[] =>
  targets.map(parseConnectionTarget).filter((target): target is ConnectionTarget => target !== null)

/** The operation open for one app, if this window has one. A plain selector
 *  rather than a `computed` per slug: the container already subscribes to the
 *  atom once, and a store per card would churn on every frame. */
export function accountOperationFor(
  operations: Readonly<Record<string, AccountOperation>>,
  slug: string
): AccountOperation | null {
  const mine = Object.values(operations).filter(operation => operation.connectors.includes(slug))

  // An unsettled operation is the live one; among settled ones the newest is the
  // result the person is still looking at.
  return mine.find(operation => !operation.settled) ?? mine[mine.length - 1] ?? null
}

/** Record the operation `connectors.connect` just opened. */
export function startAccountOperation(
  scope: ProfileScope,
  connectors: readonly string[],
  snapshot: ConnectorsConnectResult
): AccountOperation {
  const operation: AccountOperation = {
    connectors: [...connectors],
    deadlineAt: snapshot.deadline_at,
    opId: snapshot.op_id,
    scope,
    seq: snapshot.seq,
    settled: snapshot.settled,
    settledBy: snapshot.settled_by ?? null,
    targets: parseTargets(snapshot.targets)
  }

  // A finished attempt on the same app is history the moment a new one opens;
  // leaving it would let `accountOperationFor` answer with the old result once
  // this one settles too.
  const kept = Object.entries($accountOperations.get()).filter(
    ([, previous]) => !previous.settled || !previous.connectors.some(slug => operation.connectors.includes(slug))
  )

  $accountOperations.set({ ...Object.fromEntries(kept), [operation.opId]: operation })
  refetchOnSettle(operation)

  return operation
}

/** Carry the authorizing link and the vendor account across one frame.
 *
 *  An ACCOUNT operation's broadcast reaches every connected client, this
 *  person's other profiles included, and the link authorizes an ACCOUNT — so the
 *  backend strips `connect_url` and `connection_id` from it. Those travel only
 *  in the reply to the caller that asked (`connectors.connect`,
 *  `connectors.operation.status`). Replacing the target wholesale would throw
 *  away the one link the person still has to open.
 *
 *  The exception is a target that has just re-entered `initiated`: that is a new
 *  attempt, the held link belonged to the previous one, and keeping it would
 *  open a dead page. It is dropped, and `syncAccountOperation` fetches the
 *  replacement. */
function carryLink(held: ConnectionTarget | undefined, target: ConnectionTarget): ConnectionTarget {
  if (!held) {
    return target
  }

  const reissued = target.state === 'initiated' && held.state !== 'initiated'

  return {
    ...target,
    connectUrl: target.connectUrl ?? (reissued ? null : held.connectUrl),
    connectionId: target.connectionId || held.connectionId
  }
}

/** Where a snapshot came from, because the two need different staleness rules.
 *  A `broadcast` frame can arrive behind an older one, so a seq that did not
 *  move is a reorder and is dropped. A `reply` to `connectors.operation.status`
 *  is the operation as it stands: the backend bumps seq only when something
 *  changed, so the reply normally carries the seq of the last frame already
 *  applied — and it is the ONLY message that carries the authorizing link. */
type SnapshotSource = 'broadcast' | 'reply'

/** Apply one authoritative snapshot — a `connection.update` frame or a
 *  `connectors.operation.status` reply. Every one carries the operation's whole
 *  target list, so the store takes it as given and only carries the link
 *  across. */
function applySnapshot(
  opId: string,
  snapshot: Pick<ConnectionOperationStatus, 'deadline_at' | 'seq' | 'settled' | 'settled_by' | 'targets'>,
  source: SnapshotSource
): void {
  const current = $accountOperations.get()[opId]

  if (!current || (source === 'broadcast' ? snapshot.seq <= current.seq : snapshot.seq < current.seq)) {
    return
  }

  const held = new Map(current.targets.map(target => [target.name, target] as const))

  const next: AccountOperation = {
    ...current,
    deadlineAt: snapshot.deadline_at,
    seq: snapshot.seq,
    settled: snapshot.settled,
    settledBy: snapshot.settled_by ?? null,
    targets: parseTargets(snapshot.targets).map(target => carryLink(held.get(target.name), target))
  }

  $accountOperations.set({ ...$accountOperations.get(), [next.opId]: next })
  refetchOnSettle(next)
}

/** Apply one account-owned `connection.update`. A frame for an operation this
 *  window never started is not ours to hold. */
export function applyAccountConnectionUpdate(payload: ConnectionUpdatePayload): void {
  if (payload.owner.type !== 'account') {
    return
  }

  applySnapshot(payload.op_id, payload, 'broadcast')
}

/** Ask the backend for the operation again, which is the only way to get a link
 *  this window does not hold: after a Try again another window started, or after
 *  a target went back to `initiated`. Silent on failure — the card keeps saying
 *  what it last knew, and the person can press the verb again. */
export async function syncAccountOperation(opId: string): Promise<void> {
  const operation = $accountOperations.get()[opId]

  if (!operation) {
    return
  }

  try {
    const status = await accountOperationStatus(operation.scope, opId)
    applySnapshot(opId, status, 'reply')
  } catch {
    // A settled operation leaves the live registry; the list refetch that
    // settling already scheduled is the answer, not an error here.
  }
}

/** Drop one operation — the dialog closed on a finished connect, or the person
 *  stopped waiting. */
export function clearAccountOperation(opId: string): void {
  const operations = $accountOperations.get()

  if (!(opId in operations)) {
    return
  }

  const next = { ...operations }
  delete next[opId]
  $accountOperations.set(next)
}

/** A settled operation makes the hosted list and the accounts list wrong
 *  whatever it settled as: it connected an account, or it did not and the row
 *  that says "Connecting…" has to stop saying it. Refetch both; the tool lists
 *  and the catalog did not change, so they are left alone. */
function refetchOnSettle(operation: AccountOperation): void {
  if (!operation.settled) {
    return
  }

  void queryClient.invalidateQueries({ queryKey: connectorsListQueryKey(operation.scope) })
  void queryClient.invalidateQueries({ queryKey: connectorsAccountsQueryKey(operation.scope) })
}
