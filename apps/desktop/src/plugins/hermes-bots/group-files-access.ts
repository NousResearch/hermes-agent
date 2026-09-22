/** Window-local observed-denial fence. Not authorization and never persisted. */

import { assertGroupFileIntent, GroupFileError, groupFileFailure } from './group-file-errors'
import type { GroupChat, ProfileRoute } from './types'

interface AccessState {
  generation: number
  blocked: boolean
  deliveries: Set<AbortController>
  listeners: Set<() => void>
}
export interface GroupFileAccessToken {
  readonly state: AccessState
  readonly generation: number
}
type Request = <T>(route: ProfileRoute, method: string, params?: Record<string, unknown>) => Promise<T>
const rooms = new Map<string, AccessState>()

export function captureGroupFileAccess(room: GroupChat | undefined): GroupFileAccessToken | null {
  if (!room?.roomId || !room.hosted) {
    return null
  }

  const key = JSON.stringify([room.roomId, room.hosted, room.hostedEpoch || null])
  let state = rooms.get(key)

  if (!state) {
    state = { generation: 0, blocked: false, deliveries: new Set(), listeners: new Set() }
    rooms.set(key, state)
  }

  return { state, generation: state.generation }
}

export function groupFileAccessCurrent(token: GroupFileAccessToken | null): boolean {
  return !token || (!token.state.blocked && token.generation === token.state.generation)
}

export function invalidateGroupFileAccess(token: GroupFileAccessToken | null): void {
  if (!token || token.generation !== token.state.generation) {
    return
  }

  const state = token.state
  state.blocked = true
  state.generation += 1

  // Abort every lease before notifying React. Success promises queued in the
  // same event-loop tick must see this fence before they can create an anchor.
  for (const controller of state.deliveries) {
    controller.abort()
  }

  state.deliveries.clear()

  for (const listener of state.listeners) {
    listener()
  }
}

export function subscribeGroupFileAccess(token: GroupFileAccessToken | null, listener: () => void): () => void {
  token?.state.listeners.add(listener)

  return () => {
    token?.state.listeners.delete(listener)
  }
}

export function confirmGroupFileCatalog(token: GroupFileAccessToken | null): void {
  if (!token) {
    return
  }

  if (token.generation !== token.state.generation) {
    throw new GroupFileError('access')
  }

  token.state.blocked = false
}

export function guardGroupFileRequest(
  request: Request,
  token: GroupFileAccessToken | null,
  signal?: AbortSignal,
  catalog = false
): Request {
  return async <T>(route: ProfileRoute, method: string, params?: Record<string, unknown>): Promise<T> => {
    assertGroupFileIntent(signal)

    if (!catalog && !groupFileAccessCurrent(token)) {
      throw new GroupFileError('access')
    }

    try {
      const result = await request<T>(route, method, params)
      assertGroupFileIntent(signal)

      if (!catalog && !groupFileAccessCurrent(token)) {
        throw new GroupFileError('access')
      }

      return result
    } catch (error) {
      if (groupFileFailure(error) === 'access') {
        invalidateGroupFileAccess(token)
      }

      throw error
    }
  }
}

export function beginGroupFileDelivery(room: GroupChat, signal?: AbortSignal) {
  assertGroupFileIntent(signal)
  const token = captureGroupFileAccess(room)

  if (!groupFileAccessCurrent(token)) {
    throw new GroupFileError('access')
  }

  const controller = new AbortController()
  const abort = () => controller.abort()
  signal?.addEventListener('abort', abort, { once: true })
  token?.state.deliveries.add(controller)

  return {
    token,
    signal: controller.signal,
    cancel: abort,
    current: () => !controller.signal.aborted && groupFileAccessCurrent(token),
    release: () => {
      signal?.removeEventListener('abort', abort)
      token?.state.deliveries.delete(controller)
    }
  }
}
