/**
 * Plugin-scoped i18n for hermes-achievements — bundles shipped under the
 * plugin id via ctx.i18n.register (#67303), never touching core en.ts.
 * `useAchievementsI18n()` binds `t` to the message SHAPE so components keep
 * typed `k.scoreUnlocked` / `k.nextTier(tier, n)` access (same pattern as
 * kanban's `useKanban`).
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

type AchievementsMessages = {
  nav: string
  openCommand: string
  pageTitle: string
  scoreUnlocked: string
  discovered: string
  secret: string
  scanned: (when: string) => string
  stale: string
  rescan: string
  scanning: string
  retry: string
  loadFailed: string
  loadFailedHint: string
  filterAll: string
  filterUnlocked: string
  filterDiscovered: string
  filterSecret: string
  emptyTitle: string
  emptyBody: string
  whatCounts: string
  hideWhatCounts: string
  evidenceFrom: (title: string) => string
  nextTier: (tier: string, threshold: number) => string
  maxTier: string
  secretName: string
  secretDescription: string
  scoreTip: (unlocked: number, total: number) => string
}

const en: AchievementsMessages = {
  nav: 'Achievements',
  openCommand: 'Achievements: Open',
  pageTitle: 'Achievements',
  scoreUnlocked: 'unlocked',
  discovered: 'discovered',
  secret: 'secret',
  scanned: when => `scanned ${when}`,
  stale: 'stale',
  rescan: 'Rescan',
  scanning: 'Scanning…',
  retry: 'Retry',
  loadFailed: 'Could not load achievements',
  loadFailedHint: 'is the achievements plugin enabled?',
  filterAll: 'All',
  filterUnlocked: 'Unlocked',
  filterDiscovered: 'Discovered',
  filterSecret: 'Secret',
  emptyTitle: 'No achievements here',
  emptyBody: 'Nothing in this state yet — keep using Hermes.',
  whatCounts: 'What counts?',
  hideWhatCounts: 'Hide what counts',
  evidenceFrom: title => `evidence: ${title}`,
  nextTier: (tier, threshold) => `next: ${tier} · ${threshold}`,
  maxTier: 'max tier',
  secretName: '???',
  secretDescription: 'Secret achievement — hidden until the first matching signal appears.',
  scoreTip: (unlocked, total) => `Achievements: ${unlocked}/${total} unlocked`
}

const ja: AchievementsMessages = {
  nav: '実績',
  openCommand: '実績: 開く',
  pageTitle: '実績',
  scoreUnlocked: '解除済み',
  discovered: '発見済み',
  secret: 'シークレット',
  scanned: when => `スキャン ${when}`,
  stale: '古い',
  rescan: '再スキャン',
  scanning: 'スキャン中…',
  retry: '再試行',
  loadFailed: '実績を読み込めませんでした',
  loadFailedHint: '実績プラグインは有効ですか？',
  filterAll: 'すべて',
  filterUnlocked: '解除済み',
  filterDiscovered: '発見済み',
  filterSecret: 'シークレット',
  emptyTitle: 'ここには実績がありません',
  emptyBody: 'この状態のものはまだありません — Hermes を使い続けてください。',
  whatCounts: '条件は？',
  hideWhatCounts: '条件を隠す',
  evidenceFrom: title => `証拠: ${title}`,
  nextTier: (tier, threshold) => `次: ${tier} · ${threshold}`,
  maxTier: '最大階級',
  secretName: '???',
  secretDescription: 'シークレット実績 — 最初のシグナルが検出されるまで非表示です。',
  scoreTip: (unlocked, total) => `実績: ${unlocked}/${total} 解除済み`
}

export const ACHIEVEMENTS_LOCALES = { en, ja } satisfies PluginLocaleBundles

type Bound<T> = { [K in keyof T]: T[K] extends (...args: infer A) => infer R ? (...args: A) => R : string }

/** The full messages shape, bound to the active locale's translate fn. */
export type AchievementsText = Bound<AchievementsMessages>

function bind<T extends object>(t: PluginTranslate, template: T): Bound<T> {
  const out = {} as Record<string, unknown>

  for (const [key, value] of Object.entries(template)) {
    out[key] =
      typeof value === 'function'
        ? (...args: unknown[]) => t(key, ...args)
        : t(key)
  }

  return out as Bound<T>
}

export function useAchievementsI18n(): AchievementsText {
  const t = usePluginI18n('hermes-achievements')

  return useMemo(() => bind(t, en), [t])
}