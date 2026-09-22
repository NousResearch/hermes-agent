import type {
  ConnectionOperationStatus,
  ConnectionSettleReason,
  ConnectionUpdatePayload,
  ConnectorsConnectResult
} from '@hermes/shared'
import { atom } from 'nanostores'

import type { ProfileScope } from '@/hermes'
import { type ConnectionTarget, parseConnectionTarget } from '@/store/connection-request'

import { invalidateConnectors } from './keys'
import { accountOperationStatus } from './rpc'

export interface AccountOperation {
  connectors: string[]
  deadlineAt: number
  opId: string
  scope: ProfileScope
  seq: number
  settled: boolean
  settledBy: ConnectionSettleReason | null
  targets: ConnectionTarget[]
}

export const $accountOperations = atom<Readonly<Record<string, AccountOperation>>>({})

export const $abandonedConnects = atom<readonly string[]>([])

export function abandonConnect(slugs: readonly string[]): void {
  const kept = $abandonedConnects.get().filter(slug => !slugs.includes(slug))

  $abandonedConnects.set([...kept, ...slugs])
}

function resumeConnect(slugs: readonly string[]): void {
  const next = $abandonedConnects.get().filter(slug => !slugs.includes(slug))

  if (next.length !== $abandonedConnects.get().length) {
    $abandonedConnects.set(next)
  }
}

const parseTargets = (targets: ConnectionUpdatePayload['targets']): ConnectionTarget[] =>
  targets.map(parseConnectionTarget).filter((target): target is ConnectionTarget => target !== null)

export function accountOperationFor(
  operations: Readonly<Record<string, AccountOperation>>,
  slug: string
): AccountOperation | null {
  const mine = Object.values(operations).filter(operation => operation.connectors.includes(slug))

  return mine.find(operation => !operation.settled) ?? mine[mine.length - 1] ?? null
}

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

  resumeConnect(operation.connectors)

  const kept = Object.entries($accountOperations.get()).filter(
    ([, previous]) => !previous.settled || !previous.connectors.some(slug => operation.connectors.includes(slug))
  )

  $accountOperations.set({ ...Object.fromEntries(kept), [operation.opId]: operation })
  refetchOnSettle(operation)

  return operation
}

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

type SnapshotSource = 'broadcast' | 'reply'

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

export function applyAccountConnectionUpdate(payload: ConnectionUpdatePayload): void {
  if (payload.owner.type !== 'account') {
    return
  }

  applySnapshot(payload.op_id, payload, 'broadcast')
}

export async function syncAccountOperation(opId: string): Promise<void> {
  const operation = $accountOperations.get()[opId]

  if (!operation) {
    return
  }

  try {
    const status = await accountOperationStatus(operation.scope, opId)
    applySnapshot(opId, status, 'reply')
    // eslint-disable-next-line no-empty
  } catch {
  }
}

export function clearAccountOperation(opId: string): void {
  const operations = $accountOperations.get()

  if (!(opId in operations)) {
    return
  }

  const next = { ...operations }
  delete next[opId]
  $accountOperations.set(next)
}

function refetchOnSettle(operation: AccountOperation): void {
  if (!operation.settled) {
    return
  }

  invalidateConnectors(operation.scope, 'list', 'accounts')
}
