/** Durable compensation journal for hosted Group Chat setup and reconnect. */

import { atom, host } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'

import { botsText } from './i18n'
import type { ProfileRoute } from './types'

export const HOSTED_ROOM_CLEANUP_KEY = 'hosted-room-cleanup-v1'
const HOSTED_ROOM_CLEANUP_LIMIT = 64
const HOSTED_ROOM_CLEANUP_LOCK = 'hermes-bots-hosted-room-cleanup'
const HOSTED_ROOM_OWNER_LOCK_PREFIX = 'hermes-bots-hosted-room-owner:'
const HOSTED_ROOM_OWNER_LEASE_MS = 60_000

export interface HostedRoomCleanupOperation {
  armed: boolean
  cancelId?: null | string
  catalog?: null | Record<string, unknown>
  connectionId: string
  expectedGrantSha256?: null | string
  grant?: null | string
  grantSha256?: null | string
  homeConnectionId?: null | string
  homeProfile?: null | string
  kind: 'home-disband' | 'peer-reconnect' | 'peer-revoke' | 'peer-revoke-exact'
  memberId?: null | string
  operationId: string
  ownerId: string
  ownerLeaseUntil: number
  profile?: null | string
  roomId?: null | string
  setupId: string
  targetUrl?: null | string
}

export interface HostedRoomCleanup {
  operations: HostedRoomCleanupOperation[]
  version: 1
}

export const $hostedRoomCleanup = atom<HostedRoomCleanup>({ version: 1, operations: [] })

let cleanupOwnerId = ''
let cleanupStorage: null | PluginContext['storage'] = null
let cleanupDispatching = false
let cleanupDisposed = true
let cleanupGeneration = 0
let cleanupMutationTail: Promise<void> = Promise.resolve()
let cleanupOwnerLockRelease: null | (() => void) = null

interface CleanupLockManager {
  request<T>(
    name: string,
    options: { ifAvailable?: boolean; mode: 'exclusive' },
    callback: (lock: null | object) => Promise<T> | T
  ): Promise<T>
}

function newCleanupOwnerId() {
  return globalThis.crypto?.randomUUID?.() || `desktop-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function record(value: unknown): null | Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

async function routeForReference(connectionId: string, profile = 'default') {
  if (typeof host.profileRoutes !== 'function') {
    return null
  }

  const routes = await host.profileRoutes()

  return ((Array.isArray(routes) ? routes : []).find(route => {
    const routeProfile = String(route?.targetProfile || route?.profile || '')

    return String(route?.connectionId || '') === connectionId && routeProfile === profile
  }) || null) as ProfileRoute | null
}

async function request<T>(route: ProfileRoute, method: string, params: Record<string, unknown>) {
  if (!route?.connectionId || typeof host.requestProfile !== 'function') {
    throw new Error(botsText().group.hostRouteMissing)
  }

  return host.requestProfile(route, method, params) as Promise<T>
}

export function normalizeHostedRoomCleanup(value: unknown): HostedRoomCleanup {
  const candidate = record(value)
  const operations: HostedRoomCleanupOperation[] = []

  for (const raw of Array.isArray(candidate?.operations) ? candidate.operations : []) {
    const operation = record(raw)
    const operationId = String(operation?.operationId || '')
    const setupId = String(operation?.setupId || '')
    const kind = String(operation?.kind || '')
    const connectionId = String(operation?.connectionId || '')

    if (
      !operationId ||
      !setupId ||
      !connectionId ||
      !['home-disband', 'peer-reconnect', 'peer-revoke', 'peer-revoke-exact'].includes(kind)
    ) {
      continue
    }

    if (kind === 'home-disband' && !String(operation?.roomId || '')) {
      continue
    }

    if (
      ['peer-revoke', 'peer-revoke-exact'].includes(kind) &&
      (!String(operation?.grant || '') || !String(operation?.profile || ''))
    ) {
      continue
    }

    if (
      kind === 'peer-reconnect' &&
      (!String(operation?.grant || '') ||
        !/^[0-9a-f]{64}$/.test(String(operation?.grantSha256 || '')) ||
        !String(operation?.profile || '') ||
        !String(operation?.homeConnectionId || '') ||
        !String(operation?.homeProfile || '') ||
        !String(operation?.roomId || '') ||
        !String(operation?.memberId || '') ||
        !String(operation?.targetUrl || '') ||
        !record(operation?.catalog))
    ) {
      continue
    }

    const ownerId = String(operation?.ownerId || '')
    const ownerLeaseUntil = Number(operation?.ownerLeaseUntil || 0)

    operations.push({
      armed: operation?.armed === true || !ownerId,
      operationId,
      setupId,
      kind: kind as HostedRoomCleanupOperation['kind'],
      connectionId,
      ownerId,
      ownerLeaseUntil: Number.isFinite(ownerLeaseUntil) && ownerLeaseUntil > 0 ? ownerLeaseUntil : 0,
      roomId: ['home-disband', 'peer-reconnect'].includes(kind) ? String(operation?.roomId || '') : null,
      cancelId:
        kind === 'home-disband' ? String(operation?.cancelId || `rollback-${String(operation?.roomId || '')}`) : null,
      profile: kind === 'home-disband' ? null : String(operation?.profile || ''),
      grant: kind === 'home-disband' ? null : String(operation?.grant || ''),
      grantSha256: kind === 'peer-reconnect' ? String(operation?.grantSha256 || '') : null,
      expectedGrantSha256:
        kind === 'peer-reconnect' && /^[0-9a-f]{64}$/.test(String(operation?.expectedGrantSha256 || ''))
          ? String(operation?.expectedGrantSha256)
          : null,
      homeConnectionId: kind === 'peer-reconnect' ? String(operation?.homeConnectionId || '') : null,
      homeProfile: kind === 'peer-reconnect' ? String(operation?.homeProfile || '') : null,
      memberId: kind === 'peer-reconnect' ? String(operation?.memberId || '') : null,
      targetUrl: kind === 'peer-reconnect' ? String(operation?.targetUrl || '') : null,
      catalog: kind === 'peer-reconnect' ? record(operation?.catalog) : null
    })
  }

  return {
    version: 1,
    operations: operations.slice(-HOSTED_ROOM_CLEANUP_LIMIT)
  }
}

function processCleanupLock<T>(callback: () => Promise<T>) {
  const result = cleanupMutationTail.then(callback, callback)

  cleanupMutationTail = result.then(
    () => undefined,
    () => undefined
  )

  return result
}

function cleanupLockManager() {
  return (globalThis.navigator as (Navigator & { locks?: CleanupLockManager }) | undefined)?.locks
}

async function withCleanupLock<T>(callback: () => Promise<T>) {
  const locks = cleanupLockManager()

  return locks?.request
    ? locks.request(HOSTED_ROOM_CLEANUP_LOCK, { mode: 'exclusive' }, callback)
    : processCleanupLock(callback)
}

async function holdCleanupOwnerLock(ownerId: string) {
  cleanupOwnerLockRelease?.()
  cleanupOwnerLockRelease = null
  const locks = cleanupLockManager()

  if (!locks?.request) {
    return
  }

  let entered: () => void = () => undefined
  let release: () => void = () => undefined
  const acquired = new Promise<void>(resolve => {
    entered = resolve
  })
  const held = new Promise<void>(resolve => {
    release = resolve
  })

  cleanupOwnerLockRelease = release
  void locks
    .request(`${HOSTED_ROOM_OWNER_LOCK_PREFIX}${ownerId}`, { mode: 'exclusive' }, async () => {
      entered()
      await held
    })
    .catch(() => entered())
  await acquired
}

async function cleanupOwnerIsLive(operation: HostedRoomCleanupOperation) {
  if (!operation.ownerId) {
    return false
  }

  if (operation.ownerId === cleanupOwnerId) {
    return !operation.armed
  }

  const locks = cleanupLockManager()

  if (!locks?.request) {
    return operation.ownerLeaseUntil > Date.now()
  }

  try {
    let live = true

    await locks.request(
      `${HOSTED_ROOM_OWNER_LOCK_PREFIX}${operation.ownerId}`,
      { ifAvailable: true, mode: 'exclusive' },
      lock => {
        live = lock === null
      }
    )

    return live
  } catch {
    return operation.ownerLeaseUntil > Date.now()
  }
}

async function readPersistedCleanup() {
  if (!cleanupStorage?.get) {
    throw new Error('Desktop storage is unavailable, so Group Chat setup cannot be secured.')
  }

  return normalizeHostedRoomCleanup(await cleanupStorage.get(HOSTED_ROOM_CLEANUP_KEY, null))
}

async function replaceCleanup(previous: HostedRoomCleanup, next: HostedRoomCleanup) {
  if (!cleanupStorage?.set || !cleanupStorage?.get) {
    throw new Error('Desktop storage is unavailable, so Group Chat setup cannot be secured.')
  }

  $hostedRoomCleanup.set(next)

  try {
    await cleanupStorage.set(HOSTED_ROOM_CLEANUP_KEY, next)
    const persisted = normalizeHostedRoomCleanup(await cleanupStorage.get(HOSTED_ROOM_CLEANUP_KEY, null))

    if (JSON.stringify(persisted) !== JSON.stringify(next)) {
      throw new Error('Desktop storage did not persist Group Chat cleanup.')
    }
  } catch (error) {
    $hostedRoomCleanup.set(previous)
    throw error
  }
}

async function mutateCleanup(update: (current: HostedRoomCleanup) => HostedRoomCleanup) {
  return withCleanupLock(async () => {
    const current = await readPersistedCleanup()
    const next = normalizeHostedRoomCleanup(update(current))

    if (JSON.stringify(current) === JSON.stringify(next)) {
      $hostedRoomCleanup.set(current)
      return current
    }

    await replaceCleanup(current, next)

    return next
  })
}

export async function addHostedRoomCleanup(
  operation: Omit<HostedRoomCleanupOperation, 'armed' | 'ownerId' | 'ownerLeaseUntil'>
) {
  await mutateCleanup(current => {
    const next = normalizeHostedRoomCleanup({
      version: 1,
      operations: [
        ...current.operations.filter(entry => entry.operationId !== operation.operationId),
        {
          ...operation,
          armed: false,
          ownerId: cleanupOwnerId,
          ownerLeaseUntil: Date.now() + HOSTED_ROOM_OWNER_LEASE_MS
        }
      ]
    })

    if (next.operations.length >= HOSTED_ROOM_CLEANUP_LIMIT && current.operations.length >= HOSTED_ROOM_CLEANUP_LIMIT) {
      throw new Error('Group Chat cleanup is pending. Reconnect the affected devices before creating another.')
    }

    return next
  })
}

export async function releaseHostedRoomCleanup(setupId: string) {
  await mutateCleanup(current => ({
    version: 1,
    operations: current.operations.filter(operation => operation.setupId !== setupId)
  }))
}

export async function armHostedRoomCleanup(setupId: string) {
  await mutateCleanup(current => ({
    version: 1,
    operations: current.operations.map(operation =>
      operation.setupId === setupId
        ? {
            ...operation,
            armed: true,
            ownerId: '',
            ownerLeaseUntil: 0
          }
        : operation
    )
  }))
}

export function hostedRoomCleanupPending(setupId: string) {
  return normalizeHostedRoomCleanup($hostedRoomCleanup.get()).operations.some(
    operation => operation.setupId === setupId
  )
}

function homeDisbandAlreadySettled(operation: HostedRoomCleanupOperation, error: unknown) {
  const candidate = record(error)
  const inner = record(candidate?.error)
  const code = Number(candidate?.code ?? inner?.code)
  const message = String(candidate?.message || inner?.message || error || '')

  return operation.kind === 'home-disband' && code === 4113 && /hosted room not found|already disbanded/i.test(message)
}

async function peerRouteStatus(operation: HostedRoomCleanupOperation, homeRoute: ProfileRoute) {
  const state = record(
    await request<Record<string, unknown>>(homeRoute, 'groups.state', {
      room_id: operation.roomId
    })
  )
  const driver = record(state?.driver_status)
  if (!driver || !Array.isArray(driver.peer_routes)) {
    return 'unknown' as const
  }
  const route = driver.peer_routes
    .map(record)
    .find(candidate => String(candidate?.member_id || '') === String(operation.memberId || ''))

  const status = String(route?.status || '')
  const grantSha256 = String(route?.grant_sha256 || '')
  const sameGrant = grantSha256 && grantSha256 === String(operation.grantSha256 || '')
  const expectedGrant = grantSha256 && grantSha256 === String(operation.expectedGrantSha256 || '')

  if (status === 'needs_reauthorization' && sameGrant) {
    return 'nonready' as const
  }

  if (sameGrant) {
    return 'matching' as const
  }

  if (expectedGrant || (!grantSha256 && !operation.expectedGrantSha256)) {
    return 'expected' as const
  }

  if (grantSha256) {
    return 'conflict' as const
  }

  if (status === 'needs_reauthorization') {
    return 'nonready' as const
  }

  return 'unknown' as const
}

async function settlePeerReconnect(operation: HostedRoomCleanupOperation) {
  const homeRoute = await routeForReference(
    String(operation.homeConnectionId || ''),
    String(operation.homeProfile || 'default')
  )

  if (!homeRoute) {
    return 'pending' as const
  }

  try {
    if (['conflict', 'nonready'].includes(await peerRouteStatus(operation, homeRoute))) {
      return 'revoke' as const
    }
  } catch {
    /* registration remains the only safe settlement proof */
  }

  try {
    await request(homeRoute, 'groups.peer.register', {
      room_id: operation.roomId,
      member_id: operation.memberId,
      target_url: operation.targetUrl,
      target_profile: operation.profile,
      grant: operation.grant,
      catalog: operation.catalog,
      expected_grant_sha256: operation.expectedGrantSha256 || ''
    })

    return 'settled' as const
  } catch {
    try {
      return ['conflict', 'nonready'].includes(await peerRouteStatus(operation, homeRoute))
        ? ('revoke' as const)
        : ('pending' as const)
    } catch {
      return 'pending' as const
    }
  }
}

async function runCleanup(operation: HostedRoomCleanupOperation) {
  if (operation.kind === 'peer-reconnect') {
    const outcome = await settlePeerReconnect(operation)

    if (outcome === 'settled') {
      return true
    }

    if (outcome === 'pending') {
      return false
    }
  }

  const profile = operation.kind === 'home-disband' ? 'default' : String(operation.profile || '')
  const route = await routeForReference(operation.connectionId, profile)

  if (!route) {
    return false
  }

  try {
    if (operation.kind === 'home-disband') {
      await request(route, 'groups.disband', {
        room_id: operation.roomId,
        cancel_id: operation.cancelId
      })
    } else {
      await request(route, operation.kind === 'peer-revoke' ? 'groups.peer.revoke' : 'groups.peer.revoke_exact', {
        grant: operation.grant,
        profile: operation.profile
      })
    }

    return true
  } catch (error) {
    return homeDisbandAlreadySettled(operation, error)
  }
}

export async function dispatchHostedRoomCleanup() {
  if (cleanupDispatching || cleanupDisposed) {
    return
  }

  cleanupDispatching = true

  try {
    const snapshot = await mutateCleanup(current => ({
      version: 1,
      operations: current.operations.map(operation =>
        operation.ownerId === cleanupOwnerId && !operation.armed
          ? {
              ...operation,
              ownerLeaseUntil: Date.now() + HOSTED_ROOM_OWNER_LEASE_MS
            }
          : operation
      )
    }))

    for (const operation of snapshot.operations) {
      if (await cleanupOwnerIsLive(operation)) {
        continue
      }

      let claimed: HostedRoomCleanupOperation | null = null

      await mutateCleanup(current => ({
        version: 1,
        operations: current.operations.map(entry => {
          if (JSON.stringify(entry) !== JSON.stringify(operation)) {
            return entry
          }

          claimed = {
            ...entry,
            armed: true,
            ownerId: cleanupOwnerId,
            ownerLeaseUntil: Date.now() + HOSTED_ROOM_OWNER_LEASE_MS
          }

          return claimed
        })
      }))

      if (!claimed || !(await runCleanup(claimed))) {
        continue
      }

      await mutateCleanup(latest => ({
        version: 1,
        operations: latest.operations.filter(
          entry => entry.operationId !== claimed?.operationId || JSON.stringify(entry) !== JSON.stringify(claimed)
        )
      }))
    }
  } finally {
    cleanupDispatching = false
  }
}

export async function startHostedRoomCleanup(storage: PluginContext['storage']) {
  const generation = ++cleanupGeneration
  const previousOwnerId = cleanupOwnerId
  cleanupOwnerId = newCleanupOwnerId()
  cleanupStorage = storage
  cleanupDisposed = false
  await holdCleanupOwnerLock(cleanupOwnerId)

  await withCleanupLock(async () => {
    let persisted: unknown = null

    try {
      persisted = await storage?.get?.(HOSTED_ROOM_CLEANUP_KEY, null)
    } catch {
      /* empty cleanup is the safe fallback */
    }

    if (!cleanupDisposed && generation === cleanupGeneration) {
      const current = normalizeHostedRoomCleanup(persisted)
      const next = normalizeHostedRoomCleanup({
        version: 1,
        operations: current.operations.map(operation =>
          previousOwnerId && operation.ownerId === previousOwnerId
            ? {
                ...operation,
                armed: true,
                ownerId: '',
                ownerLeaseUntil: 0
              }
            : operation
        )
      })

      await replaceCleanup(current, next)
    }
  })

  if (cleanupDisposed || generation !== cleanupGeneration) {
    return
  }

  await dispatchHostedRoomCleanup().catch(() => undefined)
}

export function stopHostedRoomCleanup() {
  cleanupGeneration += 1
  cleanupDisposed = true
  cleanupOwnerLockRelease?.()
  cleanupOwnerLockRelease = null
}

export function resetHostedRoomCleanupForTests() {
  stopHostedRoomCleanup()
  cleanupDispatching = false
  cleanupOwnerId = ''
  cleanupStorage = null
  $hostedRoomCleanup.set({ version: 1, operations: [] })
}
