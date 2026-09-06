import { classicAuthorityClaim } from './group-desktop-authority'
import type { GroupChat, ProfileRoute } from './types'

const MAX_COMMANDS_PER_CLAIM = 1
const MAX_COMMANDS_PER_WAKE = 64
const MAX_ROOM_IDS_PER_CLAIM = 128
const LEASE_RENEW_INTERVAL_MS = 15_000
const LEASE_TTL_MS = 45_000

export interface DesktopRoomDescriptor {
  authorityHash: string
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
  leaseValid?: () => boolean
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
  const candidates = Object.entries(rooms || {})
    .filter(([, room]) => {
      const hosted = typeof room?.hosted === 'string' ? room.hosted.trim() : ''

      return !hosted && !room?.tombstone && Array.isArray(room?.log)
    })
    .flatMap(([name, room]) => {
      const authority = classicAuthorityClaim(room)
      const roomId = desktopRoomIdentity(name, room)

      return authority && roomId && roomId.length <= 200 && !/[\0\r\n]/.test(roomId)
        ? [{ name, roomId, authorityHash: authority.authorityHash, authorityToken: authority.authorityToken }]
        : []
    })

  const counts = new Map<string, number>()

  for (const room of candidates) {
    counts.set(room.roomId, Number(counts.get(room.roomId) || 0) + 1)
  }

  return candidates.filter(room => counts.get(room.roomId) === 1)
}

export function createDesktopRoomConsumerId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return `desktop:${globalThis.crypto.randomUUID()}`
  }

  throw new Error('Secure Group Chat control is unavailable in this Desktop build.')
}

function authorityParams(rooms: DesktopRoomDescriptor[]) {
  return rooms.map(room => ({ room_id: room.roomId, authority_token: room.authorityToken }))
}

function roomBatches(rooms: DesktopRoomDescriptor[]) {
  const batches: DesktopRoomDescriptor[][] = []

  for (let index = 0; index < rooms.length; index += MAX_ROOM_IDS_PER_CLAIM) {
    batches.push(rooms.slice(index, index + MAX_ROOM_IDS_PER_CLAIM))
  }

  return batches
}

function distinctRoutes(routes: ProfileRoute[]) {
  const seen = new Set<string>()

  return (Array.isArray(routes) ? routes : []).filter(route => {
    const key = String(route?.connectionId || '') || '__active__'

    if (seen.has(key)) {
      return false
    }

    seen.add(key)

    return true
  })
}

/** Keep gateway availability honest without claiming queued work. */
export async function refreshDesktopRoomPresence({
  routes,
  consumerId,
  rooms,
  request
}: Pick<RunDesktopRoomCommandCycleInput, 'consumerId' | 'request' | 'rooms' | 'routes'>) {
  const descriptors = desktopRoomDescriptors(rooms)

  for (const route of distinctRoutes(routes)) {
    for (const batch of roomBatches(descriptors)) {
      try {
        await request(route, 'groups.desktop.presence', {
          consumer_id: consumerId,
          room_authorities: authorityParams(batch)
        })
      } catch {
        // One unavailable gateway or old backend must not suppress the rest.
        break
      }
    }
  }
}

function boundedError(error: unknown) {
  const text = String(error instanceof Error ? error.message : 'Desktop could not apply the Group Chat command')
    .replace(/\s+/g, ' ')
    .trim()

  return text.slice(0, 240)
}

function identifier(value: unknown) {
  const text = typeof value === 'string' ? value.trim() : ''

  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/.test(text) ? text : ''
}

function commandRecord(value: unknown): value is DesktopRoomCommand {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function commandValidationError(command: DesktopRoomCommand, rooms: DesktopRoomDescriptor[]) {
  const roomId = typeof command.room_id === 'string' ? command.room_id.trim() : ''

  if (!identifier(command.command_id) || !identifier(command.lease_token)) {
    return 'The gateway returned an invalid Group Chat command identity.'
  }

  if (!roomId || !rooms.some(room => room.roomId === roomId)) {
    return 'The Group Chat command no longer matches an advertised room.'
  }

  if (!['send', 'stop'].includes(String(command.action || ''))) {
    return 'The gateway returned an unsupported Group Chat command.'
  }

  if (!command.payload || typeof command.payload !== 'object' || Array.isArray(command.payload)) {
    return 'The gateway returned an invalid Group Chat command payload.'
  }

  return ''
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
  if (!shouldContinue()) {
    return []
  }

  const descriptors = desktopRoomDescriptors(rooms)

  if (!descriptors.length) {
    return []
  }

  const outcomes: DesktopRoomCommandOutcome[] = []
  let remaining = MAX_COMMANDS_PER_WAKE

  for (const route of distinctRoutes(routes)) {
    const connectionId = String(route?.connectionId || '')
    const routeKey = connectionId || '__active__'
    const seen = new Set<string>()

    for (const batch of roomBatches(descriptors)) {
      if (remaining <= 0) {
        return outcomes
      }

      const roomAuthorities = authorityParams(batch)
      let stopClaiming = false

      while (remaining > 0) {
        if (!shouldContinue()) {
          return outcomes
        }

        const claimLimit = Math.min(MAX_COMMANDS_PER_CLAIM, remaining)
        const claimStartedAt = Date.now()
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

        const entries = (
          Array.isArray((claimed as { commands?: unknown[] } | null)?.commands)
            ? (claimed as { commands: unknown[] }).commands || []
            : []
        ).slice(0, MAX_COMMANDS_PER_WAKE)

        // Mixed-version responses may ignore limits or filters. Only one
        // usable command starts now; all others remain reclaimable by lease.
        const commands = entries
          .filter(commandRecord)
          .filter(
            command =>
              identifier(command.command_id) &&
              identifier(command.lease_token) &&
              (!actions?.length || actions.includes(String(command.action)))
          )
          .slice(0, claimLimit)

        remaining -= Math.max(1, commands.length)

        if (commands.some(command => seen.has(String(command.command_id)))) {
          break
        }

        for (const command of commands) {
          seen.add(String(command.command_id))

          if (!shouldContinue()) {
            return outcomes
          }

          if (Date.now() - claimStartedAt >= LEASE_TTL_MS) {
            stopClaiming = true

            break
          }

          let success = false
          let result: unknown
          let executionError: unknown = null
          let renewTimer: ReturnType<typeof setInterval> | null = null
          let leaseLost = false
          let renewing = false
          let renewPromise: null | Promise<void> = null
          let expiryTimer: ReturnType<typeof setTimeout> | null = null
          let expired!: () => void
          let leaseExpiresAt = claimStartedAt + LEASE_TTL_MS

          const expiry = new Promise<void>(resolve => {
            expired = resolve
          })

          const abortController = typeof AbortController === 'function' ? new AbortController() : null
          const leaseToken = String(command?.lease_token || '')
          const validationError = commandValidationError(command, descriptors)

          const loseLease = () => {
            leaseLost = true
            abortController?.abort('lease-lost')
            expired()
          }

          const expireAt = (startedAt: number) => {
            leaseExpiresAt = startedAt + LEASE_TTL_MS

            if (expiryTimer !== null) {
              clearTimeout(expiryTimer)
            }

            expiryTimer = setTimeout(loseLease, Math.max(0, startedAt + LEASE_TTL_MS - Date.now()))
          }

          if (!validationError && typeof setInterval === 'function') {
            expireAt(claimStartedAt)
            renewTimer = setInterval(() => {
              if (!shouldContinue()) {
                loseLease()
              }

              if (leaseLost || renewing) {
                return
              }

              renewing = true
              const renewStartedAt = Date.now()
              renewPromise = request(route, 'groups.desktop.renew', {
                consumer_id: consumerId,
                command_id: command.command_id,
                lease_token: leaseToken
              })
                .then(() => {
                  if (!leaseLost) {
                    expireAt(renewStartedAt)
                  }
                })
                .catch(loseLease)
                .finally(() => {
                  renewing = false
                })
            }, LEASE_RENEW_INTERVAL_MS)
          }

          try {
            if (validationError) {
              throw new Error(validationError)
            }

            result = await execute(command, descriptors, {
              leaseValid: () => !leaseLost && shouldContinue() && Date.now() < leaseExpiresAt,
              signal: abortController?.signal || null,
              route,
              consumerId,
              request: (method, params) => request(route, method, params)
            })
          } catch (error) {
            executionError = error
          } finally {
            if (renewTimer !== null && typeof clearInterval === 'function') {
              clearInterval(renewTimer)
            }

            await Promise.race([renewPromise, expiry])

            if (expiryTimer !== null) {
              clearTimeout(expiryTimer)
            }
          }

          if (
            !shouldContinue() ||
            leaseLost ||
            (executionError as { retryable?: boolean } | null)?.retryable === true
          ) {
            stopClaiming = true
            outcomes.push({
              commandId: String(command.command_id || ''),
              connectionId: routeKey,
              success: false,
              retryable: true,
              ...(leaseLost ? { leaseLost: true } : {})
            })

            continue
          }

          if (executionError) {
            result = { message: boundedError(executionError) }
          } else {
            success = true
          }

          const completion = {
            consumer_id: consumerId,
            command_id: command.command_id,
            lease_token: leaseToken,
            success,
            result
          }

          try {
            await request(route, 'groups.desktop.complete', completion)
          } catch {
            try {
              await request(route, 'groups.desktop.complete', completion)
            } catch {
              // An uncommitted result remains leased, then replays by ID.
              stopClaiming = true
            }
          }

          outcomes.push({
            commandId: String(command.command_id || ''),
            connectionId: routeKey,
            success
          })
        }

        if (stopClaiming || commands.length < claimLimit) {
          break
        }
      }
    }
  }

  return outcomes
}
