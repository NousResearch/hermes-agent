/**
 * The pieces every Bot Mode module needs and no single module can own: the
 * plugin id, the `PluginContext` `register()` captures, and the open
 * generation the bot-open and group-open click paths race on.
 *
 * The mutable ones live behind accessor pairs because an imported binding
 * cannot be reassigned — `register()` publishes the context once with
 * `setPluginCtx`, and every reader goes through `getPluginCtx()`.
 */

import { atom, type PluginContext } from '@hermes/plugin-sdk'

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

/** A deliberate bot open still in flight — the canonical cold open publishes
 *  it after the fronted-tab miss and before source prep, the recent-session
 *  open before its own source dial. Not chat ownership — never route or
 *  highlight from this key. The generation guards the clear: a superseded
 *  flight never releases its successor's mark. A bump (another open, a
 *  group, or Sessions) drops it. The one consumer beyond the row's opening
 *  acknowledgement: selection reconciliation defers while this is set, so
 *  an in-flight deliberate open is not overwritten by the still-focused
 *  chat. */
export const $pendingBotOpen = atom<null | { generation: number; key: string }>(null)

export function getBotOpenGeneration() {
  return botOpenGeneration
}

/** Release the pending-open mark, but only for the flight that set it: a
 *  superseded flight settling late must not clear its successor's mark. */
export function settlePendingBotOpen(generation: number) {
  if ($pendingBotOpen.get()?.generation === generation) {
    $pendingBotOpen.set(null)
  }
}

/** Invalidate every in-flight open; returns the generation that now owns it. */
export function bumpBotOpenGeneration() {
  $pendingBotOpen.set(null)

  return ++botOpenGeneration
}
