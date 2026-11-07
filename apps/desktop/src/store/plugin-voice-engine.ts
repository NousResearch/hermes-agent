import { backendScopeKey } from '@hermes/shared'

import type { RegisteredVoiceEngine } from '@/app/chat/composer/contrib'
import { persistentAtom } from '@/lib/persisted'
import { activeGatewayConnectionId, activeGatewayProfileKey } from '@/store/gateway'

/**
 * Which PLUGIN voice engine the start-voice button runs, per (connection, profile).
 *
 * The core engines (`chained`, `gpt-live`) are `voice.voice_chat_mode` on the backend, which
 * refuses any other value; a plugin engine is a renderer-side choice layered on top. While one is
 * chosen and still registered it wins; choosing a core engine clears it. Scoped by the same key as
 * the voice-live status so one profile's choice never leaks into another's composer.
 */
export const $pluginVoiceEngineChoice = persistentAtom<Record<string, string>>('hermes.desktop.pluginVoiceEngine', {})

function scopeKey(): string {
  return backendScopeKey(activeGatewayConnectionId(), activeGatewayProfileKey())
}

/** The chosen engine if it is still registered; an uninstalled plugin falls back to core. */
export function resolvePluginVoiceEngine(
  engines: readonly RegisteredVoiceEngine[],
  choice: Readonly<Record<string, string>>,
  scope: string = scopeKey()
): null | RegisteredVoiceEngine {
  const id = choice[scope]

  return id ? (engines.find(engine => engine.id === id) ?? null) : null
}

/** Pick a plugin engine for this scope, or `null` to hand the button back to the core engine. */
export function choosePluginVoiceEngine(id: null | string, scope: string = scopeKey()): void {
  const current = $pluginVoiceEngineChoice.get()

  if ((current[scope] ?? null) === id) {
    return
  }

  const next = { ...current }

  if (id) {
    next[scope] = id
  } else {
    delete next[scope]
  }

  $pluginVoiceEngineChoice.set(next)
}
