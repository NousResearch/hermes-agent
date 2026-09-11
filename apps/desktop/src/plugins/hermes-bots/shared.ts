/**
 * The pieces every Bot Mode module needs and no single module can own: the
 * plugin id, the `PluginContext` `register()` captures, and the open
 * generation the bot-open and group-open click paths race on.
 *
 * The mutable ones live behind accessor pairs because an imported binding
 * cannot be reassigned — `register()` publishes the context once with
 * `setPluginCtx`, and every reader goes through `getPluginCtx()`.
 */

import type { PluginContext } from '@hermes/plugin-sdk'

export const ID = 'hermes-bots'

/** Captured in register() so components can reach plugin storage. */
let pluginCtx: PluginContext | null = null

export function getPluginCtx(): PluginContext | null {
  return pluginCtx
}

export function setPluginCtx(ctx: PluginContext | null) {
  pluginCtx = ctx
}

/** Monotonic open generation. A bot open and a group open supersede each
 *  other, so both bump it and both compare against it — and they no longer
 *  live in the same module. */
let botOpenGeneration = 0

export function getBotOpenGeneration() {
  return botOpenGeneration
}

/** Invalidate every in-flight open; returns the generation that now owns it. */
export function bumpBotOpenGeneration() {
  return ++botOpenGeneration
}
