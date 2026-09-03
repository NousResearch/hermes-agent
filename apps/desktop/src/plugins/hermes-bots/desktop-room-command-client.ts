import type { GroupChat, ProfileRoute } from './types'

const MAX_COMMANDS_PER_CLAIM = 8
const MAX_COMMANDS_PER_WAKE = 64
const MAX_ROOM_IDS_PER_CLAIM = 128
const LEASE_RENEW_INTERVAL_MS = 15_000

export interface DesktopRoomDescriptor {
  authorityToken: string
  name: string
  roomId: string
}

export interface DesktopRoomCommand {
  action?: string
  attempts?: number
  command_id?: string
  lease_token?: string
  payload?: Record<string, unknown>
  room_id?: string
  target_command_state?: string
  target_result_code?: string
}

interface CommandExecutionContext {
  consumerId: string
  request: (method: string, params: Record<string, unknown>) => Promise<unknown>
  route: ProfileRoute
  signal: AbortSignal | null
}

interface RunDesktopRoomCommandCycleInput {
  actions?: string[] | null
  consumerId: string
  execute: (
    command: DesktopRoomCommand,
    rooms: DesktopRoomDescriptor[],
    context: CommandExecutionContext
  ) => Promise<unknown>
  request: (route: ProfileRoute, method: string, params: Record<string, unknown>) => Promise<unknown>
  rooms: Record<string, GroupChat>
  routes: ProfileRoute[]
  shouldContinue?: () => boolean
}

export interface DesktopRoomCommandOutcome {
  commandId: string
  connectionId: string
  leaseLost?: boolean
  retryable?: boolean
  success: boolean
}

/** Stable identity shared by the bounded gateway projection and command queue. */
export function desktopRoomIdentity(name: string, room: GroupChat) {
  const roomId = String(room?.roomId || '').trim()

  return roomId || `name:${String(name || '').trim()}`
}

/** Classic rooms this Desktop can coordinate. Hosted rooms run on a gateway. */
export function desktopRoomDescriptors(rooms: Record<string, GroupChat>): DesktopRoomDescriptor[] {
  return Object.entries(rooms || {})
    .filter(([, room]) => {
      const hosted = typeof room?.hosted === 'string' ? room.hosted.trim() : ''

      return !hosted && !room?.tombstone && Array.isArray(room?.log)
    })
    .map(([name, room]) => ({
      name,
      roomId: desktopRoomIdentity(name, room),
      authorityToken: String(room?.desktopAuthorityToken || '').trim()
    }))
    .filter(room => room.roomId && room.name && room.authorityToken)
}

export function createDesktopRoomConsumerId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return `desktop:${globalThis.crypto.randomUUID()}`
  }

  return `desktop:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function boundedError(error: unknown) {
  const text = String(error instanceof Error ? error.message : 'Desktop could not apply the Group Chat command')
    .replace(/\s+/g, ' ')
    .trim()

  return text.slice(0, 240)
}

/** Claim and apply classic-room commands from every reachable default gateway.
 * Missing methods identify an older backend and leave its queue untouched. */
export async function runDesktopRoomCommandCycle({
  routes,
  consumerId,
  rooms,
  request,
  execute,
  actions = null,
  shouldContinue = () => true
}: RunDesktopRoomCommandCycleInput): Promise<DesktopRoomCommandOutcome[]> {
  const descriptors = desktopRoomDescriptors(rooms)

  if (!descriptors.length) {
    return []
  }

  const roomBatches: DesktopRoomDescriptor[][] = []

  for (let index = 0; index < descriptors.length; index += MAX_ROOM_IDS_PER_CLAIM) {
    roomBatches.push(descriptors.slice(index, index + MAX_ROOM_IDS_PER_CLAIM))
  }

  const seenConnections = new Set<string>()
  const outcomes: DesktopRoomCommandOutcome[] = []

  for (const route of Array.isArray(routes) ? routes : []) {
    const connectionId = String(route?.connectionId || '')
    const routeKey = connectionId || '__active__'

    if (seenConnections.has(routeKey)) {
      continue
    }

    seenConnections.add(routeKey)
    let remaining = MAX_COMMANDS_PER_WAKE

    for (const batch of roomBatches) {
      if (remaining <= 0) {
        return outcomes
      }

      const roomAuthorities = batch.map(room => ({
        room_id: room.roomId,
        authority_token: room.authorityToken
      }))

      while (remaining > 0) {
        const claimLimit = Math.min(MAX_COMMANDS_PER_CLAIM, remaining)
        let claimed: unknown

        try {
          claimed = await request(route, 'groups.desktop.claim', {
            consumer_id: consumerId,
            room_authorities: roomAuthorities,
            ...(Array.isArray(actions) && actions.length
              ? {
                  actions
                }
              : {}),
            limit: claimLimit
          })
        } catch {
          break
        }

        const commands = Array.isArray((claimed as { commands?: unknown[] } | null)?.commands)
          ? (claimed as { commands: DesktopRoomCommand[] }).commands || []
          : []

        remaining -= commands.length

        for (const command of commands) {
          if (!shouldContinue()) {
            return outcomes
          }

          let success = false
          let result: unknown
          let renewTimer: ReturnType<typeof setInterval> | null = null
          let leaseLost = false
          const abortController = typeof AbortController === 'function' ? new AbortController() : null
          const leaseToken = String(command?.lease_token || '')

          if (leaseToken && typeof setInterval === 'function') {
            renewTimer = setInterval(() => {
              if (!shouldContinue()) {
                return
              }

              void request(route, 'groups.desktop.renew', {
                consumer_id: consumerId,
                command_id: command.command_id,
                lease_token: leaseToken
              }).catch(() => {
                leaseLost = true
                abortController?.abort('lease-lost')
              })
            }, LEASE_RENEW_INTERVAL_MS)
          }

          try {
            result = await execute(command, descriptors, {
              signal: abortController?.signal || null,
              route,
              consumerId,
              request: (method, params) => request(route, method, params)
            })

            if (leaseLost) {
              outcomes.push({
                commandId: String(command.command_id || ''),
                connectionId: routeKey,
                success: false,
                retryable: true,
                leaseLost: true
              })

              continue
            }

            success = true
          } catch (error) {
            if (leaseLost || (error as { retryable?: boolean } | null)?.retryable === true) {
              outcomes.push({
                commandId: String(command.command_id || ''),
                connectionId: routeKey,
                success: false,
                retryable: true,
                ...(leaseLost
                  ? {
                      leaseLost: true
                    }
                  : {})
              })

              continue
            }

            result = {
              message: boundedError(error)
            }
          } finally {
            if (renewTimer !== null && typeof clearInterval === 'function') {
              clearInterval(renewTimer)
            }
          }

          try {
            await request(route, 'groups.desktop.complete', {
              consumer_id: consumerId,
              command_id: command.command_id,
              lease_token: leaseToken,
              success,
              result
            })
          } catch {
            // The lease expires and retries the same command id. Room effects
            // are idempotent, so a lost completion ACK cannot duplicate text.
          }

          outcomes.push({
            commandId: String(command.command_id || ''),
            connectionId: routeKey,
            success
          })
        }

        if (commands.length < claimLimit) {
          break
        }
      }
    }
  }

  return outcomes
}
