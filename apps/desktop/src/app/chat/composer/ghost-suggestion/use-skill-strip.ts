import { useStore } from '@nanostores/react'
/**
 * Skill strip — the transient suggestion hint above the composer.
 *
 * After the user pauses typing free text, a strip appears ABOVE the input
 * (not inline like the ghost) listing the top matching skills together with
 * a one-line description of what each does. It fades out by itself after a
 * short delay, so it advertises capability without ever getting in the way.
 *
 * Requirement (老大): "输入框上方会短暂的出现推荐的技能1-2秒的功能描述。
 * 若匹配多个技能则多个技能描述也同时短暂出现。" The strip re-arms every
 * time the draft changes and at least one candidate is available; a stale
 * timer from the previous draft is cancelled first so a fast typist never
 * sees a hint for text they already moved past.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { CommandsCatalogLike } from '@/lib/desktop-slash-commands'
import { peekCachedSlashCompletion } from '@/lib/slash-completion-cache'
import { $skillSuggestionsEnabled } from '@/store/skill-suggestions'

import type { GhostCandidate } from './use-ghost-suggestion'
import { useDraftValue } from './use-ghost-suggestion'

const VISIBLE_MS = 4_000
const MAX_ITEMS = 3
const DISMISS_STORAGE_KEY = 'hermes.desktop.skillStripDismissed'

export interface SkillStripItem {
  /** Slash command, e.g. `/learn`. */
  command: string
  /** One-line description from the catalog (or the ghost's rationale). */
  description: string
}

export interface SkillStripState {
  /** Items currently visible. Empty hides the strip. */
  items: SkillStripItem[]
  /** Permanently dismissed by the user (persisted in localStorage). */
  dismissed: boolean
  /** Permanently dismiss the strip (user clicked 不再显示). */
  dismissForever: () => void
}

function isTriggerActive(draft: string): boolean {
  return draft.includes('@') || draft.includes('/') || draft.includes(':')
}

/**
 * Map each ghost candidate to a `command + description` pair. The live
 * catalog is the source of truth; a small built-in fallback covers the
 * core commands so the strip still shows something before the slash
 * panel has ever populated the catalog cache. If the catalog entry for
 * a command is missing (uninstalled while the LLM response was in
 * flight), the item is dropped entirely.
 */
const FALLBACK_DESCRIPTIONS: Record<string, string> = {
  '/learn': '学习一个主题或技能，例如无人机考证、编程语言、行业知识，支持渐进式学习路径与自测',
  '/commit': '提交当前代码变更，自动生成规范化的 commit message，支持事务性回滚',
  '/commit-push': '提交并推送代码到远程仓库，一键完成本地提交与远程同步',
  '/voice': '语音输入，用口述代替打字，自动转写为文字并填入输入框',
  '/gif-search': '搜索动图，按关键词查找 GIF 表情包并插入对话',
  '/help': '命令与快捷键完整列表，查看所有可用命令与使用说明'
}

/** Look up a command's one-line description: live catalog first, then the
 * built-in fallback table. */
export function describeCommand(
  command: string,
  catalog: CommandsCatalogLike | null | undefined
): string | null {
  for (const [c, description] of catalog?.pairs ?? []) {
    if ((c.startsWith('/') ? c : `/${c}`) === command && description) {
      return description
    }
  }

  return FALLBACK_DESCRIPTIONS[command] ?? null
}

function resolveDescriptions(
  candidates: GhostCandidate[],
  catalog: CommandsCatalogLike | null | undefined
): SkillStripItem[] {
  const byCommand = new Map<string, string>()

  for (const [command, description] of catalog?.pairs ?? []) {
    byCommand.set(command.startsWith('/') ? command : `/${command}`, description || '')
  }

  const items: SkillStripItem[] = []

  for (const candidate of candidates) {
    const description = byCommand.get(candidate.command) ?? FALLBACK_DESCRIPTIONS[candidate.command]

    if (description === undefined) {
      continue
    }

    items.push({ command: candidate.command, description })

    if (items.length >= MAX_ITEMS) {
      break
    }
  }

  return items
}

export function useSkillStrip(
  draftRef: { current: string },
  candidates: GhostCandidate[]
): SkillStripState {
  const [items, setItems] = useState<SkillStripItem[]>([])

  const [dismissed, setDismissed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(DISMISS_STORAGE_KEY) === '1'
    } catch {
      return false
    }
  })

  const hideTimerRef = useRef<number | undefined>(undefined)
  const draft = useDraftValue(draftRef)
  const enabled = useStore($skillSuggestionsEnabled)

  // Re-arm the visible strip whenever the draft changes and candidates exist.
  // eslint-disable-next-line no-restricted-syntax -- timer handle (auto-hide), not a reactive-value mirror
  useEffect(() => {
    if (dismissed || !enabled) {
      return
    }

    if (draft.length < 2 || isTriggerActive(draft) || candidates.length === 0) {
      // Not a suggestion moment — clear any pending show.
      setItems([])

      return
    }

    const catalog = peekCachedSlashCompletion<CommandsCatalogLike>('catalog')
    const resolved = resolveDescriptions(candidates, catalog)

    if (resolved.length === 0) {
      setItems([])

      return
    }

    setItems(resolved)

    if (hideTimerRef.current !== undefined) {
      window.clearTimeout(hideTimerRef.current)
    }

    hideTimerRef.current = window.setTimeout(() => setItems([]), VISIBLE_MS)

    return () => {
      if (hideTimerRef.current !== undefined) {
        window.clearTimeout(hideTimerRef.current)
        hideTimerRef.current = undefined
      }
    }
  }, [draft, candidates, dismissed, enabled])

  const dismissForever = useCallback(() => {
    setDismissed(true)

    try {
      localStorage.setItem(DISMISS_STORAGE_KEY, '1')
    } catch {
      // localStorage unavailable — the strip just stays for this session.
    }
  }, [])

  return useMemo(
    () => ({ items, dismissed, dismissForever }),
    [items, dismissed, dismissForever]
  )
}
