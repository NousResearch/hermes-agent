/** Durable command-failure state projected onto the matching Group Chat. */

import { $groupChats, updateGroupChat } from './group-chat'
import type { HostedRoomCommand, HostedRoomOutbox } from './hosted-room-client'
import { botsText } from './i18n'

const MAX_ATTEMPTS = 5
const TERMINAL_CODES = new Set([4110, 4111, 4113, 4117])
const UNBOUNDED_COMMANDS = new Set<HostedRoomCommand['kind']>(['disband', 'stop'])

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

export function hostedRoomCommandFailureCode(error: unknown, command: HostedRoomCommand) {
  const candidate = record(error)
  const nested = record(candidate?.error)
  const code = Number(candidate?.code ?? nested?.code)
  const message = String(candidate?.message || nested?.message || error || '')

  if (Number.isInteger(code) && TERMINAL_CODES.has(code)) {
    return String(code)
  }

  if (
    command.kind === 'stop' &&
    code === 5116 &&
    /already disbanded|authority|does not exist|managed by another|not found|retired/i.test(message)
  ) {
    return String(code)
  }

  return !UNBOUNDED_COMMANDS.has(command.kind) && command.attempts >= MAX_ATTEMPTS ? 'retry-exhausted' : ''
}

export function failedHostedRoomCommand(outbox: HostedRoomOutbox, roomId: string) {
  return outbox.commands.find(command => command.roomId === roomId && command.status === 'failed')
}

export function pendingHostedRoomSafetyCommand(outbox: HostedRoomOutbox, roomId: string) {
  return outbox.commands.find(
    command =>
      command.roomId === roomId &&
      command.status === 'pending' &&
      (command.kind === 'disband' || command.kind === 'stop')
  )
}

export function safetyCommandsBlockedByFailure(outbox: HostedRoomOutbox) {
  return outbox.commands.filter(
    command =>
      command.status === 'pending' &&
      (command.kind === 'disband' || command.kind === 'stop') &&
      Boolean(failedHostedRoomCommand(outbox, command.roomId))
  )
}

export function surfaceHostedRoomCommandFailure(command: HostedRoomCommand) {
  const roomName = Object.entries($groupChats.get()).find(([, room]) => room.roomId === command.roomId)?.[0]

  if (!roomName) {
    return
  }

  updateGroupChat(
    roomName,
    room => ({
      ...room,
      hostedStatus: {
        canRetry: true,
        canStop: room.hostedStatus?.canStop,
        label: botsText().group.hostedNeedsAttention,
        retryCommandId: command.commandId,
        state: 'failed'
      },
      continuityIssue: botsText().group.hostRejectedCommand
    }),
    { sync: false }
  )
}
