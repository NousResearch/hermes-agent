import { useMemo, useState } from 'react'

import { Settings2, Wrench } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { EnvVarInfo } from '@/types/hermes'

import { EnvVarRow, useEnvCredentials } from './env-credentials'
import { asText } from './helpers'
import { LoadingState, SettingsContent } from './primitives'

// Providers live on their own page; messaging-platform credentials live on the
// dedicated Messaging page (and are hidden here via `channel_managed`). This
// view covers tool API keys plus server/setting env vars (API server, webhook,
// gateway), which fold into the Settings tab.
const KEY_TABS = [
  { icon: Wrench, id: 'tool', label: 'Tools' },
  { icon: Settings2, id: 'setting', label: 'Settings' }
] as const

type KeyCategoryId = (typeof KEY_TABS)[number]['id']

const CATEGORY_LABELS: Record<KeyCategoryId, string> = {
  setting: 'Settings',
  tool: 'Tools'
}

// Backend categories that surface under each tab. Server/gateway vars carry the
// `messaging` category server-side but belong with general settings here, since
// the platform-credential half of `messaging` is owned by the Messaging page.
const TAB_CATEGORIES: Record<KeyCategoryId, readonly string[]> = {
  setting: ['setting', 'messaging'],
  tool: ['tool']
}

function tabForCategory(category: string): KeyCategoryId | null {
  for (const tab of KEY_TABS) {
    if (TAB_CATEGORIES[tab.id].includes(category)) {
      return tab.id
    }
  }

  return null
}

function CategoryTabs({
  active,
  counts,
  onSelect
}: {
  active: KeyCategoryId
  counts: Record<KeyCategoryId, number>
  onSelect: (id: KeyCategoryId) => void
}) {
  return (
    <div className="mb-4 inline-flex w-full gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/30 p-1">
      {KEY_TABS.map(tab => {
        const isActive = active === tab.id
        const count = counts[tab.id]

        return (
          <button
            className={cn(
              'flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-[length:var(--conversation-text-font-size)] font-medium transition-colors',
              isActive
                ? 'bg-(--ui-chat-surface-background) text-foreground shadow-sm'
                : 'text-(--ui-text-secondary) hover:text-foreground'
            )}
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            type="button"
          >
            <tab.icon className="size-3.5 shrink-0" />
            <span className="truncate">{tab.label}</span>
            {count > 0 && (
              <span
                className={cn(
                  'rounded-full px-1.5 text-[0.6875rem] tabular-nums',
                  isActive ? 'bg-primary/12 text-primary' : 'bg-(--ui-bg-tertiary)/60 text-muted-foreground'
                )}
              >
                {count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

export function KeysSettings() {
  const { rowProps, vars } = useEnvCredentials()
  const [activeCategory, setActiveCategory] = useState<KeyCategoryId>('tool')

  const groups = useMemo(() => {
    if (!vars) {
      return []
    }

    return KEY_TABS.map(t => t.id).flatMap(tab => {
      const cats = TAB_CATEGORIES[tab]

      const entries = Object.entries(vars)
        .filter(([, info]) => !info.channel_managed && cats.includes(asText(info.category)))
        .sort(([a], [b]) => a.localeCompare(b))

      return entries.length === 0 ? [] : [{ category: tab, label: CATEGORY_LABELS[tab], entries }]
    })
  }, [vars])

  // Tab badge counts reflect how many keys are set per tab. Channel-managed
  // credentials are owned by the Messaging page and excluded here.
  const categoryCounts = useMemo<Record<KeyCategoryId, number>>(() => {
    const counts: Record<KeyCategoryId, number> = { setting: 0, tool: 0 }

    if (!vars) {
      return counts
    }

    for (const info of Object.values(vars)) {
      if (!info.is_set || info.channel_managed) {
        continue
      }

      const tab = tabForCategory(asText(info.category))

      if (tab) {
        counts[tab] += 1
      }
    }

    return counts
  }, [vars])

  if (!vars) {
    return <LoadingState label="Loading API keys and credentials..." />
  }

  const visible = groups.filter(g => g.category === activeCategory)

  return (
    <SettingsContent>
      <CategoryTabs active={activeCategory} counts={categoryCounts} onSelect={setActiveCategory} />

      {visible.map(group => (
        <section className="mb-6" key={group.category}>
          <div className="grid gap-2">
            {group.entries.map(([key, info]: [string, EnvVarInfo]) => (
              <EnvVarRow info={info} key={key} varKey={key} {...rowProps} />
            ))}
          </div>
        </section>
      ))}

      {visible.length === 0 && (
        <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          Nothing configured in this category yet.
        </div>
      )}
    </SettingsContent>
  )
}
