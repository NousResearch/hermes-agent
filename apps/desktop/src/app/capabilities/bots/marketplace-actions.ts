import { atom } from 'nanostores'

import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { activeGatewayConnectionId, requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import { notify } from '@/store/notifications'

const PASSIVE_MARKETPLACE_METHODS = new Set([
  'bots.catalog',
  'bots.installed',
  'bots.routines.list',
  'bots.status'
])

export async function requestBotMarketplace<T>(
  method: string,
  params: Record<string, unknown>,
  scope: ProfileScope
): Promise<T> {
  const profile = scope && typeof scope === 'object' ? scope.profile?.trim() || 'default' : scope?.trim() || 'default'
  const spawnPriority = PASSIVE_MARKETPLACE_METHODS.has(method) ? 'background' : 'foreground'

  if (scope && typeof scope === 'object') {
    return requestGatewayForAgent<T>(scope.connectionId || null, profile, method, params, undefined, undefined, {
      spawnPriority
    })
  }

  return requestGatewayForProfile<T>(profile, method, params, undefined, undefined, { spawnPriority })
}

export async function invalidateBotRoster(): Promise<void> {
  await queryClient.invalidateQueries({ queryKey: ['hermes-bots', 'roster'] })
  await queryClient.invalidateQueries({ queryKey: ['capabilities-agent-roster'] })
}

export async function openInstalledBot(name: string, scope: ProfileScope): Promise<unknown> {
  const [{ saveSelectedRosterBot }, { createCanonicalChat }, { botWorkspaceOwnerKey, setBotsWorkspaceOwner }] =
    await Promise.all([
      import('@/plugins/hermes-bots/bot-state'),
      import('@/plugins/hermes-bots/canonical-chat'),
      import('@/plugins/hermes-bots/routing')
    ])

  const bot = await installedBotOwner(name, scope)

  saveSelectedRosterBot(bot)
  setBotsWorkspaceOwner(botWorkspaceOwnerKey(bot), bot)

  return createCanonicalChat(bot)
}

async function installedBotOwner(name: string, scope: ProfileScope): Promise<RosterRow> {
  const sourceProfile = scope && typeof scope === 'object' ? scope.profile?.trim() || 'default' : scope?.trim() || 'default'

  const discoveredRoute =
    scope && typeof scope === 'object'
      ? null
      : (await window.hermesDesktop?.getProfileRoutes?.([sourceProfile]))?.find(route => route.profile === sourceProfile) ?? null

  const connectionId =
    scope && typeof scope === 'object'
      ? scope.connectionId?.trim() || 'local'
      : discoveredRoute?.connectionId || activeGatewayConnectionId() || 'local'

  const connectionMode = discoveredRoute?.mode === 'local' || connectionId === 'local' ? 'local' : 'remote'

  return {
    name,
    connectionId,
    sourceScoped: true,
    remoteSource: connectionMode === 'remote',
    route: {
      connectionId,
      mode: connectionMode,
      profile: name,
      targetProfile: name
    }
  }
}

export const $foregroundBotProfile = atom('')
const inflightKickoffs = new Map<string, Promise<unknown>>()

/** Seat the committed profile in Bot Mode, then submit its reviewed starter to
 * the title-resolved canonical chat. Only concurrent clicks are coalesced here;
 * the backend's durable first_task state decides whether a later explicit retry
 * is allowed (including after deleting and reinstalling the same profile). */
export function kickoffInstalledBot(name: string, scope: ProfileScope, kickoffPrompt = ''): Promise<unknown> {
  const text = kickoffPrompt.trim()

  const scopeKey = scope && typeof scope === 'object'
    ? `${scope.connectionId || 'local'}\u0000${scope.profile || 'default'}`
    : scope || 'default'

  const kickoffKey = `${scopeKey}\u0000${name}\u0000${text}`
  const existing = inflightKickoffs.get(kickoffKey)

  if (existing) {
    return existing
  }

  const run = (async () => {
    if (!text) {
      throw new Error('The installed bot starter prompt is empty.')
    }

    const [
      { saveSelectedRosterBot },
      { createCanonicalChat },
      { botWorkspaceOwnerKey, requestForBot, setBotsWorkspaceOwner }
    ] = await Promise.all([
      import('@/plugins/hermes-bots/bot-state'),
      import('@/plugins/hermes-bots/canonical-chat'),
      import('@/plugins/hermes-bots/routing')
    ])

    const bot = await installedBotOwner(name, scope)

    saveSelectedRosterBot(bot)
    $foregroundBotProfile.set(name)
    const ownerKey = botWorkspaceOwnerKey(bot)
    setBotsWorkspaceOwner(ownerKey, bot)

    const storedSessionId = await createCanonicalChat(bot)

    if (!storedSessionId) {
      throw new Error(`Could not resolve ${name}'s Bot Chat.`)
    }

    const resumed = await requestForBot<{ session_id?: string }>(bot, 'session.resume', {
      session_id: storedSessionId,
      profile: bot.route?.targetProfile || bot.name,
      omit_messages: true
    })

    if (!resumed?.session_id) {
      throw new Error(`Could not resume ${name}'s Bot Chat.`)
    }

    return requestForBot(bot, 'prompt.submit', {
      session_id: resumed.session_id,
      text
    })
  })()

  inflightKickoffs.set(kickoffKey, run)
  void run.finally(() => {
    if (inflightKickoffs.get(kickoffKey) === run) {
      inflightKickoffs.delete(kickoffKey)
    }
  }).catch(() => undefined)

  return run
}

export const notifyBotMarketplace = notify
