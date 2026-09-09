import { desktopRoomIdentity } from './desktop-room-command-client'
import type { GroupChat } from './types'

export interface DesktopCommandResult {
  room_name: string
  thread_id?: string
  stopped?: boolean
  stale?: true
}

export interface DesktopCommandReceipt {
  at: number
  action: 'send' | 'stop'
  roomId: string
  authorityHash: string
  result: DesktopCommandResult
}

// Numeric entries are pre-release settlements without recoverable results.
// Keep them as fail-closed markers, never as permission to execute again.
export type DesktopCommandSettled = Record<string, DesktopCommandReceipt | number>
export const DESKTOP_COMMAND_RECEIPT_LIMIT = 128

function receipt(value: unknown): DesktopCommandReceipt | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const raw = value as DesktopCommandReceipt
  const result = raw.result

  if (
    !Number.isFinite(raw.at) ||
    raw.at < 0 ||
    !['send', 'stop'].includes(raw.action) ||
    typeof raw.roomId !== 'string' ||
    !raw.roomId ||
    raw.roomId.length > 200 ||
    typeof raw.authorityHash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(raw.authorityHash) ||
    !result ||
    typeof result !== 'object' ||
    Array.isArray(result) ||
    typeof result.room_name !== 'string' ||
    !result.room_name ||
    result.room_name.length > 512 ||
    (result.thread_id !== undefined &&
      (typeof result.thread_id !== 'string' || !result.thread_id || result.thread_id.length > 200)) ||
    (result.stopped !== undefined && typeof result.stopped !== 'boolean') ||
    (result.stale !== undefined && result.stale !== true) ||
    (raw.action === 'send' &&
      (result.stale !== undefined || (result.thread_id ? result.stopped !== undefined : result.stopped !== true))) ||
    (raw.action === 'stop' &&
      (typeof result.stopped !== 'boolean' || result.thread_id !== undefined || (!result.stopped && !result.stale)))
  ) {
    return null
  }

  return {
    at: raw.at,
    action: raw.action,
    roomId: raw.roomId,
    authorityHash: raw.authorityHash,
    result: {
      room_name: result.room_name,
      ...(result.thread_id !== undefined ? { thread_id: result.thread_id } : {}),
      ...(result.stopped !== undefined ? { stopped: result.stopped } : {}),
      ...(result.stale === true ? { stale: true as const } : {})
    }
  }
}

export function boundedDesktopCommandSettled(
  value: unknown,
  limit = DESKTOP_COMMAND_RECEIPT_LIMIT
): DesktopCommandSettled {
  return Object.fromEntries(
    Object.entries(value && typeof value === 'object' && !Array.isArray(value) ? value : {})
      .filter(([id]) => Boolean(id) && id.length <= 160)
      .map(
        ([id, raw]) =>
          [id, receipt(raw) ?? (typeof raw === 'number' && Number.isFinite(raw) ? Math.max(0, raw) : 0)] as const
      )
      .sort(
        ([, left], [, right]) =>
          (typeof right === 'number' ? right : right.at) - (typeof left === 'number' ? left : left.at)
      )
      .slice(0, Math.max(0, Math.min(DESKTOP_COMMAND_RECEIPT_LIMIT, limit)))
  )
}

export function desktopCommandResult(name: string, room: GroupChat, id: string, action: 'send' | 'stop') {
  if (!Object.hasOwn(room.desktopCommandSettled || {}, id)) {
    return null
  }

  const saved = receipt(room.desktopCommandSettled?.[id])

  if (
    !saved ||
    saved.action !== action ||
    saved.roomId !== desktopRoomIdentity(name, room) ||
    saved.authorityHash !== room.desktopAuthorityHash
  ) {
    throw new Error(
      'This Group Chat command already settled, but its saved result cannot be recovered. Send a new message explicitly.'
    )
  }

  return saved.result
}

export function settleDesktopCommand(
  name: string,
  room: GroupChat,
  id: string,
  action: 'send' | 'stop',
  result: DesktopCommandResult
) {
  if (Object.hasOwn(room.desktopCommandSettled || {}, id)) {
    return room.desktopCommandSettled!
  }

  const saved = receipt({
    at: Date.now(),
    action,
    roomId: desktopRoomIdentity(name, room),
    authorityHash: room.desktopAuthorityHash,
    result
  })

  if (!saved || !id || id.length > 160) {
    throw new Error('The Group Chat command result could not be saved.')
  }

  return boundedDesktopCommandSettled({ [id]: saved, ...room.desktopCommandSettled })
}
