/**
 * Plugin-scoped i18n for Bot Mode — bundles registered under the plugin id via
 * `ctx.i18n.register`, never touching core `en.ts`. Mirrors the kanban plugin:
 * `usePluginI18n` returns a stringly-typed `t(key, …)`, and `useBots()` binds it
 * to the message SHAPE so components keep typed `b.roster.search` access.
 *
 * Only strings Bot Mode OWNS live here. Generic verbs (Cancel, Delete, Remove,
 * Retry, Close, Loading…) and shared vocabulary core already ships in every
 * locale — weekday names, Daily/Hourly, Scheduled jobs — resolve against core
 * via `useI18n()` / `translateNow()`. Duplicating those here would be a
 * second, worse translation that drifts.
 *
 * Three kinds of literal deliberately stay hardcoded, and none of them is a
 * missed key:
 *
 *  - **Prompts sent to a model**, not shown as chrome: the room-picture image
 *    prompt and the scheduled-routine instruction. They are addressed to the
 *    model, which reads English best.
 *  - **Syntax and identifiers**: cron expressions and their examples, React
 *    keys, workspace ids.
 *  - **`'You'`**, the author marker on room-log entries. It is persisted into
 *    the log and compared as a sentinel (`group-activity.ts`), so it stays
 *    English where it is WRITTEN (`group-chat-parts.tsx`, `group-rounds.ts`);
 *    the places that RENDER the reader's own lines use `group.you` instead.
 *
 * Legacy Bot Mode copy covers en / ko / ja / zh / zh-hant. Canonical hosted
 * groups also cover Arabic, Russian, French, German and Spanish; missing keys
 * retain the official English fallback. Nouns match core
 * 机器人 / 機器人, プロファイル / 配置档案 / 設定檔, ゲートウェイ / 网关 / 閘道.
 */
import { type PluginLocaleBundles, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { bind, type Bound } from './i18n-bind'
import { en } from './i18n-en'
import { ja } from './i18n-ja'
import { ko } from './i18n-ko'
import type { BotsMessages } from './i18n-messages'
import { zh } from './i18n-zh'
import { zhHant } from './i18n-zhHant'
import { getPluginCtx } from './shared'

export const BOTS_LOCALES: PluginLocaleBundles = {
  en,
  ko,
  ja,
  zh,
  'zh-hant': zhHant,
  ar: { canonical: CANONICAL_GROUP_LOCALES.ar },
  ru: { canonical: CANONICAL_GROUP_LOCALES.ru },
  fr: { canonical: CANONICAL_GROUP_LOCALES.fr },
  de: { canonical: CANONICAL_GROUP_LOCALES.de },
  es: { canonical: CANONICAL_GROUP_LOCALES.es }
}

export type BotsText = Bound<BotsMessages>

/** The Bot Mode strings for the active locale — one hook every component reads. */
export function useBots(): BotsText {
  const t = usePluginI18n('hermes-bots')

  return useMemo(() => bind(t, en), [t])
}

/** Resolve a dotted path against the English bundle — the floor for a read
 *  that beats `ctx.i18n` into existence, so an unresolved key never ships as
 *  the literal `cron.runsHourly`. */
function english(key: string, ...args: unknown[]): string {
  const leaf = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], en)

  return typeof leaf === 'function' ? (leaf as (...a: unknown[]) => string)(...args) : String(leaf ?? key)
}

/** Resolve module-level strings afresh: ctx.i18n.t keeps its identity across
 * locale changes, so translator identity cannot cache translated leaves. */
export function botsText(): BotsText {
  return bind(getPluginCtx()?.i18n?.t ?? english, en)
}
