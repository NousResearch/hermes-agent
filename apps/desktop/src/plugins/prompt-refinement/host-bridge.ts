/**
 * The plugin's only door to the running app, through `@hermes/plugin-sdk`
 * alone (bundled plugins may not import app internals). Every other module
 * takes these functions as plain dependencies, so tests and an upstream SDK
 * change both meet one small file.
 */

import { host } from '@hermes/plugin-sdk'

import type { OneShotCaller } from './engine'

// Mirrors the app's slash test (`isSlashCommandText`): leading `@ref` lines the
// composer prepends don't hide a command typed after them.
const ATTACHMENT_REF_LINE_RE = /^@[a-z][a-z0-9-]*:[^\n]*\n?/i
const SLASH_COMMAND_RE = /^\/[^\s/]*(?:\s|$)/

export function isSlashCommand(text: string): boolean {
  let rest = text

  while (ATTACHMENT_REF_LINE_RE.test(rest)) {
    rest = rest.replace(ATTACHMENT_REF_LINE_RE, '')
  }

  return SLASH_COMMAND_RE.test(rest.trimStart())
}

/** The chat the user is in (stored id; null on a fresh chat), and its changes.
 *  The composer that sends is the focused one, so a change during a review
 *  means the user moved to another chat. */
export const focusedChat = {
  current: (): null | string => host.state.focusedStoredSessionId.get() ?? null,
  listen: (listener: () => void): (() => void) => host.state.focusedStoredSessionId.listen(() => listener())
}

/** Runtime id of the focused chat's live session (null before it has one). */
export const focusedRuntimeSessionId = (): null | string => host.state.focusedSessionId.get() ?? null

/** Who answers a refinement for the focused chat: its owner connection and
 *  profile (the active ones for a fresh draft), and the model the app shows.
 *  The reuse cache keys on this so answers never cross profiles or models. */
export function focusedRoute(): { connectionId: string; profile: string; chatModel: string } {
  const owner = host.state.focusedSessionOwner.get()

  return {
    connectionId: owner?.connectionId ?? host.state.connectionId.get() ?? '',
    profile: owner?.profile ?? host.state.focusedSessionProfile.get() ?? '',
    chatModel: host.state.model.get() ?? ''
  }
}

/**
 * `llm.oneshot` caller for the engine, routed to the backend that owns
 * `routeSessionId` (its profile scope). `borrowSessionModel` adds `session_id`,
 * which makes the backend use that chat's provider/model instead of the
 * `prompt_refinement` auxiliary slot. The call never creates a session,
 * appends history, or exposes tools.
 */
export function oneShotCaller(route: { routeSessionId: null | string; borrowSessionModel: boolean }): OneShotCaller {
  return async (params, { signal, timeoutMs }) => {
    const body: Record<string, unknown> = {
      input: params.input,
      instructions: params.instructions,
      max_tokens: params.maxTokens,
      task: 'prompt_refinement',
      temperature: params.temperature
    }

    if (route.borrowSessionModel && route.routeSessionId) {
      body.session_id = route.routeSessionId
    }

    const result = await host.requestForSession<{ text?: unknown }>(route.routeSessionId, 'llm.oneshot', body, {
      signal,
      timeoutMs
    })

    return typeof result?.text === 'string' ? result.text : ''
  }
}
