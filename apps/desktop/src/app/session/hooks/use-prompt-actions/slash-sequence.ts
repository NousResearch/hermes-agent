import { parseSlashCommand } from '@hermes/shared'
import type { MutableRefObject } from 'react'

import {
  canonicalDesktopSlashCommand,
  type CommandsCatalogLike,
  rememberDesktopCommandsCatalog,
  splitDesktopSlashCommandSequence
} from '@/lib/desktop-slash-commands'

import type { GatewayRequest } from './utils'

/** Commands whose purpose is to select or create another chat. */
const SESSION_CHANGING_COMMANDS = ['/branch', '/new', '/resume']

export interface SlashSequenceEnv {
  activeSessionIdRef: MutableRefObject<null | string>
  /** Invocation-time runtime whose catalog decides the line's grammar. */
  catalogSessionId?: string
  rawCommand: string
  /** Segments already split by an earlier invocation that re-homed them. */
  remainingSequence?: string[]
  requestGateway: GatewayRequest
  /** Run the remaining segments as a fresh invocation on the chat a command opened. */
  rehome: (remainingSequence: string[], sessionId?: string) => Promise<boolean>
  runSlash: (command: string, sessionHint?: string) => Promise<void>
  /** Explicit caller target; otherwise each command resolves the invocation-time chat. */
  sessionHint?: string
}

/** Split a typed line into its slash commands, hydrating the grammar catalog when needed. */
async function resolveSequence({ catalogSessionId, rawCommand, remainingSequence, requestGateway }: SlashSequenceEnv) {
  if (remainingSequence) {
    return remainingSequence
  }

  const sequence = splitDesktopSlashCommandSequence(rawCommand)

  // The local table intentionally contains only Desktop-owned commands;
  // backend-owned built-ins (for example /goal) get their argument grammar
  // from commands.catalog. A fast paste+Enter can beat the completion
  // request, so hydrate once here before deciding that a second slash token
  // is merely part of the first command's argument.
  if ([...rawCommand.matchAll(/(?:^|\s)\/[^\s/]+/g)].length <= 1) {
    return sequence
  }

  try {
    const catalog = await requestGateway<CommandsCatalogLike>('commands.catalog', {
      ...(catalogSessionId ? { session_id: catalogSessionId } : {})
    })

    rememberDesktopCommandsCatalog(catalog)

    return splitDesktopSlashCommandSequence(rawCommand)
  } catch {
    // Preserve today's single-command interpretation if an older or
    // disconnected backend cannot provide the grammar catalog.
    return sequence
  }
}

/** Run each command of a typed line in order. Resolves false when a re-homed tail failed to submit. */
export async function runSlashSequence(env: SlashSequenceEnv): Promise<boolean> {
  const { activeSessionIdRef, rehome, runSlash, sessionHint } = env
  const sequence = await resolveSequence(env)

  for (const [index, command] of sequence.entries()) {
    const activeBefore = activeSessionIdRef.current
    await runSlash(command, sessionHint)

    // Keep an ordinary sequence pinned to its invocation-time session even
    // if the user navigates while an async command is running. Only a
    // command whose purpose is to select/create another chat may re-home
    // the remaining segments: they run as a fresh invocation, so the
    // session, route and destination are captured for the chat it opened.
    const commandName = canonicalDesktopSlashCommand(parseSlashCommand(command).name)
    const remaining = sequence.slice(index + 1)

    if (
      remaining.length &&
      SESSION_CHANGING_COMMANDS.includes(commandName) &&
      activeSessionIdRef.current !== activeBefore
    ) {
      return rehome(remaining, activeSessionIdRef.current ?? undefined)
    }
  }

  return true
}
