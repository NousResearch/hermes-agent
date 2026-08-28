/**
 * The SOUL a bot is born with, and the agent-to-agent messaging protocol
 * section every SOUL has to keep.
 *
 * Three surfaces need a piece of this and none of them can own it: the create
 * dialog composes a new SOUL, the advanced editor re-appends the protocol when
 * a custom SOUL is saved, and the roster backfills profiles that predate it.
 */

import { host } from '@hermes/plugin-sdk'

import { botHandle, serverInjectsProtocol } from './data'
import { displayName } from './labels'
import type { RosterRow } from './types'

/** The agent-to-agent messaging protocol, reusable so a CUSTOM SOUL keeps
 *  the handoff protocol too — a custom SOUL used to silently drop it,
 *  breaking @mentions for customized bots (@wesleysimplicio, #16). */
function messagingProtocolSection(name: string, roster: RosterRow[] | null | undefined): string {
  const teammates = (roster || []).filter(b => b.name !== name)
  const handle = botHandle(name)

  return [
    '## Messaging other agents',
    '',
    'You work alongside other named agents. Every agent (including you) has',
    'ONE canonical conversation titled "Bot Chat" — created with the agent,',
    'so it always exists. Agent-to-agent messages are delivered straight',
    'into it, like a DM. To message a teammate, run:',
    '',
    '```',
    'hermes -p <agent-name> chat --in ~ -c "Bot Chat" --create-if-missing -Q -q "Message from \uD83E\uDD16 ' +
      handle +
      ' (@' +
      handle +
      '): your message"',
    '',
    'Run the send with background=true and notify_on_complete=true on the',
    'terminal tool, then finish your turn — the reply arrives later as a',
    'background process notification. Never block waiting for it.',
    '```',
    '',
    '(`--in ~ -c "Bot Chat" --create-if-missing` resumes their canonical',
    'conversation in the home workspace, creating it if the target has no',
    '"Bot Chat" yet. `-Q` keeps output clean. Always open with the',
    '"Message from \uD83E\uDD16 ' + handle + ' (@' + handle + '):" prefix so they know',
    'who is talking (the @handle lets the app show your avatar to them).',
    'Their reply prints to stdout — relay the relevant part back to the',
    'user, and say which agent it came from.)',
    '',
    'If a message in YOUR chat starts with "Message from \uD83E\uDD16 <name>", it is',
    'a teammate messaging you, not the user. Answer it directly — your reply',
    'reaches them via their own delivery — and use the same command if you',
    'need to start a conversation yourself.',
    '',
    'When the user writes @<agent-name> or says "ask <name> to ..." /',
    '"tell <name> ...", that is a handoff: message that agent, wait for the',
    'reply, and report back.',
    '',
    'The roster grows over time — run `hermes profile list` for the LIVE',
    'teammate list before a handoff. Teammates when you were created:',
    ...(teammates.length
      ? teammates.map(b => `- \`${b.name}\`${b.description ? ` — ${b.description}` : ''}`)
      : ['- (none yet)'])
  ].join('\n')
}

/** True when SOUL.md already carries the Bot Mode handoff section.
 *  #16 appends this at create-time; pre-existing profiles (especially
 *  `default`) never went through composeSoul and silently lack it. */
function hasMessagingProtocol(soul: null | string | undefined): boolean {
  return /(^|\n)## Messaging other agents(\s|$)/.test(soul || '')
}

/** Idempotent: append the protocol once, never duplicate a custom SOUL
 *  that already has it (clone-from-default after a backfill, Edit save).
 *  No-op when the backend injects the protocol into the system prompt
 *  itself (bot_mode_protocol) — SOUL.md stays the user's identity text. */
export function ensureMessagingProtocol(
  soul: null | string | undefined,
  name: string,
  roster: RosterRow[] | null | undefined
) {
  const text = (soul || '').trim()

  if (serverInjectsProtocol || hasMessagingProtocol(text)) {return text}
  const section = messagingProtocolSection(name, roster)

  return text ? text + '\n\n' + section : section
}

const soulProtocolChecked = new Set<string>()
const soulProtocolInflight = new Set<string>()

/** One-shot per profile per session: if an existing SOUL has no protocol,
 *  append it. This is the install-time fix for default / pre-Bot-Mode
 *  personas that #16 never touched. Never overwrites identity text. */
export function backfillMessagingProtocol(roster: RosterRow[] | null | undefined) {
  // Newer backends teach the protocol via the system prompt — never touch
  // user SOUL files when the server already covers every session.
  if (serverInjectsProtocol) {
    return
  }

  for (const bot of roster || []) {
    const name = bot && bot.name

    if (!name || soulProtocolChecked.has(name) || soulProtocolInflight.has(name)) {
      continue
    }

    soulProtocolInflight.add(name)
    host
      .request<{ soul?: string }>('profiles.describe', {
        name
      })
      .then(res => {
        const soul = (res && res.soul) || ''

        if (hasMessagingProtocol(soul)) {
          soulProtocolChecked.add(name)

          return null
        }

        return host
          .request('profiles.configure', {
            name,
            soul: ensureMessagingProtocol(soul, name, roster)
          })
          .then(() => {
            soulProtocolChecked.add(name)
          })
      })
      .catch(() => {
        // Older gateway or a one-off describe/configure miss — do not hammer.
        soulProtocolChecked.add(name)
      })
      .finally(() => {
        soulProtocolInflight.delete(name)
      })
  }
}

/** SOUL.md for a new bot: identity (or the user's custom SOUL) + the
 *  messaging protocol — which ships UNLESS the backend injects it into the
 *  system prompt itself (bot_mode_protocol capability). */
interface ComposeSoulOptions {
  /** The user's own SOUL text, when the create form supplied one. */
  customSoul?: null | string
  description?: null | string
  name: string
  roster?: RosterRow[] | null
  title?: null | string
}

export function composeSoul({ name, title, description, roster, customSoul }: ComposeSoulOptions): string {
  if (customSoul && customSoul.trim()) {
    return ensureMessagingProtocol(customSoul, name, roster)
  }

  const lines = [
    `# ${displayName({
      name,
      title
    })}`,
    '',
    title ? `**Role:** ${title}` : null,
    description ? `**Mission:** ${description}` : null,
    '',
    `You are ${displayName({
      name,
      title
    })}, a persistent named agent (profile \`${name}\`) on this machine.`,
    'You keep your own memory, skills, and conversation history across sessions.'
  ]

  const identity = lines.filter(line => line !== null).join('\n')

  return serverInjectsProtocol ? identity : identity + '\n\n' + messagingProtocolSection(name, roster)
}
