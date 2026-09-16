import type { DesktopAgentRoster, DesktopRosterAgent } from '@/global'
import { requestGatewayForAgent } from '@/store/gateway'

const DIRECT_BOT_DELIVERY_TIMEOUT_MS = 1_500_000

export interface DirectBotInvocation {
  prompt: string
  target: string
}

export interface DirectBotReply {
  bot: DesktopRosterAgent
  reply: string
}

export function parseDirectBotInvocation(arg: string): DirectBotInvocation | null {
  const match = /^\s*(\S+)\s+([\s\S]*\S)\s*$/.exec(arg)

  return match ? { target: match[1], prompt: match[2] } : null
}

function routeLabel(agent: DesktopRosterAgent): string {
  return `@${agent.handle}`
}

export function resolveDirectBotTarget(roster: DesktopAgentRoster, rawTarget: string): DesktopRosterAgent {
  const target = rawTarget.trim().replace(/^@/, '').toLowerCase()
  const agents = roster.agents.filter(agent => agent.handle.toLowerCase() === target)

  if (agents.length === 1) {
    return agents[0]
  }

  const byProfile = roster.agents.filter(agent => agent.profile.toLowerCase() === target)

  if (byProfile.length === 1) {
    return byProfile[0]
  }

  if (byProfile.length > 1) {
    throw new Error(
      `Bot "${rawTarget}" is ambiguous. Use a source-qualified handle: ${byProfile.map(routeLabel).join(', ')}.`
    )
  }

  const suggestions = roster.agents
    .filter(agent => agent.handle.toLowerCase().includes(target) || agent.profile.toLowerCase().includes(target))
    .slice(0, 5)
    .map(routeLabel)

  throw new Error(
    suggestions.length
      ? `Unknown Bot "${rawTarget}". Did you mean ${suggestions.join(', ')}?`
      : `Unknown Bot "${rawTarget}". Open Bots to refresh the roster.`
  )
}

export function directBotCompletions(roster: DesktopAgentRoster, prefix: string) {
  const needle = prefix.trim().replace(/^@/, '').toLowerCase()

  return roster.agents
    .filter(agent => !needle || agent.handle.toLowerCase().startsWith(needle))
    .slice(0, 12)
    .map(agent => ({
      text: `/to ${agent.handle} `,
      display: `@${agent.handle}`,
      meta: `Bot · ${agent.connectionLabel}`,
      group: 'Bots'
    }))
}

export async function deliverDirectBot(
  roster: DesktopAgentRoster,
  invocation: DirectBotInvocation,
  source: { connectionId: string; profile: string }
): Promise<DirectBotReply> {
  const bot = resolveDirectBotTarget(roster, invocation.target)
  const owner = roster.sources.find(candidate => candidate.connectionId === bot.connectionId)

  if (!owner?.reachable && owner?.error !== 'connect-on-demand') {
    throw new Error(`Bot @${bot.handle} is unreachable on ${bot.connectionLabel}: ${owner?.error || 'source offline'}.`)
  }

  const result = await requestGatewayForAgent<{ reply?: string }>(
    bot.connectionId,
    bot.targetProfile || bot.profile,
    'bot_relay.deliver',
    {
      profile: bot.targetProfile || bot.profile,
      message: invocation.prompt,
      from_profile: source.profile,
      from_handle: source.profile === 'default' ? 'hermes' : source.profile,
      from_connection: source.connectionId
    },
    DIRECT_BOT_DELIVERY_TIMEOUT_MS
  )

  return { bot, reply: String(result?.reply || '').trim() }
}