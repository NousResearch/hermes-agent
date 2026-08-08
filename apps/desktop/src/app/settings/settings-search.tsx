import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate, useSearchParams } from 'react-router'

import { Command, CommandInput } from '@/components/ui/command'
import { useI18n } from '@/i18n'
import type { IconComponent } from '@/lib/icons'
import {
  Archive,
  BarChart3,
  Bell,
  Globe,
  Info,
  Keyboard,
  KeyRound,
  Package,
  SearchIcon,
  Settings2,
  Zap
} from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'

import { SETTINGS_ROUTE } from '../routes'

import { FIELD_DESCRIPTIONS, FIELD_LABELS, SECTIONS } from './constants'
import { fieldCopyForSchemaKey } from './field-copy'

interface SearchItem {
  id: string
  label: string
  icon: IconComponent
  sectionId: string
  routeValue: string
  type: 'config' | 'page'
  keywords: string
}

// Settings nav keys whose labels back the non-config page catalog. These must
// stay in sync with `t.settings.nav` — the localized label is resolved at
// render time, never hardcoded.
type SettingsPageLabelKey =
  | 'about'
  | 'archivedChats'
  | 'billing'
  | 'gateway'
  | 'keybinds'
  | 'keysSettings'
  | 'keysTools'
  | 'notifications'
  | 'plugins'
  | 'providerAccounts'
  | 'providerApiKeys'
  | 'providerCustomEndpoints'

interface NonConfigPage {
  icon: IconComponent
  labelKey: SettingsPageLabelKey
  keywords: string[]
  tab: string
}

const NON_CONFIG_PAGES: NonConfigPage[] = [
  {
    icon: Zap,
    keywords: ['accounts', 'sign in', 'oauth', 'login', 'subscription', 'models', 'provider'],
    labelKey: 'providerAccounts',
    tab: 'providers&pview=accounts'
  },
  {
    icon: KeyRound,
    keywords: ['providers', 'api key', 'keys', 'secrets', 'tokens'],
    labelKey: 'providerApiKeys',
    tab: 'providers&pview=keys'
  },
  {
    icon: Globe,
    keywords: ['custom endpoint', 'openai compatible', 'base url', 'api url', 'endpoint'],
    labelKey: 'providerCustomEndpoints',
    tab: 'providers&pview=custom-endpoints'
  },
  { icon: Globe, keywords: ['connection', 'messaging', 'gateway'], labelKey: 'gateway', tab: 'gateway' },
  {
    icon: KeyRound,
    keywords: ['api', 'secrets', 'tokens', 'credentials', 'browser', 'search'],
    labelKey: 'keysTools',
    tab: 'keys&kview=tools'
  },
  {
    icon: Settings2,
    keywords: ['gateway', 'proxy', 'server', 'webhook', 'env'],
    labelKey: 'keysSettings',
    tab: 'keys&kview=settings'
  },
  { icon: Keyboard, keywords: ['keyboard', 'shortcuts', 'hotkeys'], labelKey: 'keybinds', tab: 'keybinds' },
  { icon: Bell, keywords: ['alerts', 'completion sound'], labelKey: 'notifications', tab: 'notifications' },
  { icon: Package, keywords: ['extensions'], labelKey: 'plugins', tab: 'plugins' },
  {
    icon: BarChart3,
    keywords: ['billing', 'usage', 'plan', 'credits', 'payments'],
    labelKey: 'billing',
    tab: 'billing'
  },
  { icon: Archive, keywords: ['history', 'archived'], labelKey: 'archivedChats', tab: 'sessions' },
  { icon: Info, keywords: ['version', 'about'], labelKey: 'about', tab: 'about' }
]

// Scoring: mirror the command-palette approach — AND semantics (every term must
// match), then grade by match quality on the visible label.
const scoreItem = (label: string, keywords: string, needle: string): number => {
  const normalizedLabel = label.toLowerCase()
  const normalizedKeywords = keywords.toLowerCase()
  const terms = needle.split(/\s+/).filter(Boolean)

  if (terms.some(term => !normalizedLabel.includes(term) && !normalizedKeywords.includes(term))) {
    return 0
  }

  if (normalizedLabel === needle) {
    return 1
  }

  if (normalizedLabel.startsWith(needle)) {
    return 0.9
  }

  const words = normalizedLabel.split(/[^\p{L}\p{N}]+/u).filter(Boolean)

  if (words.includes(needle)) {
    return 0.85
  }

  if (words.some(word => word.startsWith(needle))) {
    return 0.8
  }

  if (normalizedLabel.includes(needle)) {
    return 0.7
  }

  if (terms.every(term => normalizedLabel.includes(term))) {
    return 0.6
  }

  return 0.4
}

/** Build the flat search index from current i18n translations. */
function buildSearchIndex(
  sections: typeof SECTIONS,
  tFieldLabels: Record<string, string>,
  tFieldDescriptions: Record<string, string>,
  tSections: Record<string, string>,
  tNav: Record<string, string>,
  pageResult: (page: string) => string
): SearchItem[] {
  const items: SearchItem[] = []

  for (const section of sections) {
    const sectionName = tSections[section.id] ?? section.label

    for (const key of section.keys) {
      const label = `${sectionName}: ${
        fieldCopyForSchemaKey(tFieldLabels, key) ??
        fieldCopyForSchemaKey(FIELD_LABELS, key) ??
        key.split('.').pop() ??
        key
      }`

      const desc =
        fieldCopyForSchemaKey(tFieldDescriptions, key) ?? fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, key) ?? ''

      items.push({
        id: `field-${key}`,
        icon: section.icon,
        keywords: [sectionName, section.label, key, label, desc].join(' ').toLowerCase(),
        label,
        routeValue: `${SETTINGS_ROUTE}?tab=config:${section.id}&field=${encodeURIComponent(key)}`,
        sectionId: section.id,
        type: 'config'
      })
    }
  }

  for (const page of NON_CONFIG_PAGES) {
    const label = pageResult(tNav[page.labelKey])
    items.push({
      id: `page-${page.tab}`,
      icon: page.icon,
      keywords: [label, ...page.keywords, 'settings'].join(' ').toLowerCase(),
      label,
      routeValue: `${SETTINGS_ROUTE}?tab=${page.tab}`,
      sectionId: 'page',
      type: 'page'
    })
  }

  return items
}

export function SettingsSearch() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const blurTimerRef = useRef(0)

  // Focus the search bar on mount so keyboard users land directly on it.
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Clean up the blur timer on unmount to avoid setState on unmounted component.
  useEffect(() => {
    return () => window.clearTimeout(blurTimerRef.current)
  }, [])

  // Build the search index from current i18n — stable deps only (t changes on
  // locale switch, SECTIONS is static).
  const allItems = useMemo(
    () =>
      buildSearchIndex(
        SECTIONS,
        t.settings.fieldLabels,
        t.settings.fieldDescriptions,
        t.settings.sections,
        t.settings.nav,
        t.settings.searchPageResult
      ),
    [
      t.settings.fieldLabels,
      t.settings.fieldDescriptions,
      t.settings.nav,
      t.settings.searchPageResult,
      t.settings.sections
    ]
  )

  const ranked = useMemo(() => {
    const needle = normalize(search)

    if (!needle) {
      return []
    }

    return allItems
      .map(item => ({ item, score: scoreItem(item.label, item.keywords, needle) }))
      .filter(entry => entry.score > 0)
      .sort((a, b) => b.score - a.score)
      .map(entry => entry.item)
      .slice(0, 12)
  }, [allItems, search])

  // Reset active index when results change.
  useEffect(() => {
    setActiveIndex(0)
  }, [ranked])

  const handleSelect = (item: SearchItem) => {
    const currentTab = searchParams.get('tab')

    if (item.type === 'config') {
      const targetTab = item.routeValue.match(/tab=([^&]+)/)?.[1]

      if (currentTab === targetTab) {
        const params = new URLSearchParams(searchParams)
        const fieldMatch = item.routeValue.match(/field=([^&]+)/)

        if (fieldMatch) {
          params.set('field', decodeURIComponent(fieldMatch[1]))
          navigate({ search: params.toString() }, { replace: true })
        }
      } else {
        navigate(item.routeValue)
      }
    } else {
      navigate(item.routeValue)
    }

    setOpen(false)
    setSearch('')
  }

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      setOpen(false)
      setSearch('')
      event.preventDefault()
    } else if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex(prev => Math.min(prev + 1, ranked.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex(prev => Math.max(prev - 1, 0))
    } else if (event.key === 'Enter' && ranked[activeIndex]) {
      event.preventDefault()
      handleSelect(ranked[activeIndex])
    }
  }

  const showDropdown = open && search.length > 0

  // Portal the dropdown outside the OverlayView's overflow:hidden container.
  const dropdownContent =
    showDropdown && containerRef.current
      ? (() => {
          const rect = containerRef.current.getBoundingClientRect()

          return createPortal(
            // z-[140] matches the select-popover rung (ui/select.tsx) — this is the
            // same surface class (portaled floating menu) and stays above dialogs
            // (z-[120]/z-[130]) without competing with tooltips/palette (z-[200]+).
            <div
              className="fixed z-[140] overflow-y-auto rounded-b-md border border-t-0 border-border bg-popover shadow-lg"
              style={{ left: rect.left, top: rect.bottom, width: rect.width, maxHeight: 240 }}
            >
              {ranked.length === 0 ? (
                <div className="py-6 text-center text-sm text-muted-foreground">{t.commandCenter.noResults}</div>
              ) : (
                ranked.map((item, index) => {
                  const Icon = item.icon

                  return (
                    <div
                      className={cn(
                        'flex cursor-pointer items-center gap-2 px-3 py-2 text-sm transition-colors',
                        index === activeIndex ? 'bg-accent' : 'hover:bg-accent/50'
                      )}
                      key={item.id}
                      onClick={() => handleSelect(item)}
                      onMouseEnter={() => setActiveIndex(index)}
                    >
                      <Icon className="size-3.5 shrink-0 text-muted-foreground" />
                      <span className="truncate">{item.label}</span>
                    </div>
                  )
                })
              )}
            </div>,
            document.body
          )
        })()
      : null

  return (
    <div className="shrink-0 border-b border-border" ref={containerRef}>
      <Command className="rounded-none bg-transparent" shouldFilter={false}>
        <div className="flex h-9 items-center gap-2 px-3">
          <SearchIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <CommandInput
            className="h-8 border-none p-0 text-sm shadow-none"
            onBlur={() => {
              blurTimerRef.current = window.setTimeout(() => setOpen(false), 200)
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={handleKeyDown}
            onValueChange={value => {
              setSearch(value)
              setOpen(true)
            }}
            placeholder={t.settings.searchPlaceholder.config}
            ref={inputRef}
            value={search}
          />
          {search && (
            <kbd
              className={cn(
                'ml-auto shrink-0 rounded border border-border/60 bg-muted/50 px-1.5 py-0.5 text-[10px] leading-none text-muted-foreground'
              )}
            >
              esc
            </kbd>
          )}
        </div>
      </Command>
      {dropdownContent}
    </div>
  )
}
