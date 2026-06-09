import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { getHermesConfigRecord, listSessions } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import {
  Activity,
  Archive,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  Clock,
  Cpu,
  Globe,
  type IconComponent,
  Info,
  KeyRound,
  MessageCircle,
  Monitor,
  Moon,
  Package,
  Palette,
  Plus,
  Settings,
  Settings2,
  Sun,
  Users,
  Wrench,
  Zap
} from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $commandPaletteOpen, closeCommandPalette, setCommandPaletteOpen } from '@/store/command-palette'
import { type ThemeMode, useTheme } from '@/themes/context'

import {
  AGENTS_ROUTE,
  ARTIFACTS_ROUTE,
  COMMAND_CENTER_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  sessionRoute,
  SETTINGS_ROUTE,
  SKILLS_ROUTE
} from '../routes'
import { FIELD_LABELS, SECTIONS } from '../settings/constants'
import { prettyName } from '../settings/helpers'

interface PaletteItem {
  active?: boolean
  icon: IconComponent
  id: string
  /** Keep the palette open after running (live-preview pickers like theme/mode). */
  keepOpen?: boolean
  keywords?: string[]
  label: string
  /** Action to run when selected. Mutually exclusive with `to`. */
  run?: () => void
  /** Open a nested palette page (VS Code-style "choose X → options"). */
  to?: string
}

interface PaletteGroup {
  heading: string
  items: PaletteItem[]
}

/** A nested page reachable from a root item via `to`. */
interface PalettePage {
  groups: PaletteGroup[]
  placeholder: string
  title: string
}

interface SessionEntry {
  id: string
  preview?: string
  title: string
}

type SessionRow = Awaited<ReturnType<typeof listSessions>>['sessions'][number]

const toSessionEntry = (session: SessionRow): SessionEntry => ({
  id: session.id,
  preview: session.preview ?? undefined,
  title: sessionTitle(session)
})

const NON_CONFIG_SETTINGS: ReadonlyArray<{ icon: IconComponent; keywords?: string[]; label: string; tab: string }> = [
  {
    icon: Zap,
    keywords: ['accounts', 'sign in', 'oauth', 'login', 'subscription', 'models', 'anthropic', 'openai'],
    label: 'Providers',
    tab: 'providers&pview=accounts'
  },
  {
    icon: KeyRound,
    keywords: ['providers', 'api key', 'keys', 'secrets', 'tokens'],
    label: 'Provider API keys',
    tab: 'providers&pview=keys'
  },
  { icon: Globe, keywords: ['connection', 'messaging'], label: 'Gateway', tab: 'gateway' },
  {
    icon: KeyRound,
    keywords: ['api', 'secrets', 'tokens', 'credentials', 'browser', 'search'],
    label: 'Tools & Keys',
    tab: 'keys&kview=tools'
  },
  {
    icon: Settings2,
    keywords: ['gateway', 'proxy', 'server', 'webhook', 'env'],
    label: 'Tools & Keys settings',
    tab: 'keys&kview=settings'
  },
  { icon: Wrench, keywords: ['servers', 'tools'], label: 'MCP', tab: 'mcp' },
  { icon: Archive, keywords: ['history', 'archived'], label: 'Archived Chats', tab: 'sessions' },
  { icon: Info, keywords: ['version', 'about'], label: 'About', tab: 'about' }
]

const THEME_MODES: ReadonlyArray<{ icon: IconComponent; label: string; mode: ThemeMode }> = [
  { icon: Sun, label: 'Light', mode: 'light' },
  { icon: Moon, label: 'Dark', mode: 'dark' },
  { icon: Monitor, label: 'System', mode: 'system' }
]

function fieldLabel(key: string): string {
  return FIELD_LABELS[key] ?? prettyName(key.split('.').pop() ?? key)
}

export function CommandPalette() {
  const open = useStore($commandPaletteOpen)
  const navigate = useNavigate()
  const { availableThemes, mode, resolvedMode, setMode, setTheme, themeName } = useTheme()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState<string | null>(null)

  // Server-backed sources for the type-to-search groups, fetched lazily while
  // the palette is open. react-query handles caching/dedup/staleness.
  const configQuery = useQuery({
    queryKey: ['command-palette', 'config'],
    queryFn: getHermesConfigRecord,
    enabled: open
  })

  const sessionsQuery = useQuery({
    queryKey: ['command-palette', 'sessions'],
    queryFn: () => listSessions(200, 1, 'exclude'),
    enabled: open
  })

  const archivedQuery = useQuery({
    queryKey: ['command-palette', 'archived'],
    queryFn: () => listSessions(200, 0, 'only'),
    enabled: open
  })

  const mcpServers = useMemo(() => {
    const raw = configQuery.data?.mcp_servers

    return raw && typeof raw === 'object' && !Array.isArray(raw)
      ? Object.keys(raw as Record<string, unknown>).sort()
      : []
  }, [configQuery.data])

  const sessions = useMemo(() => (sessionsQuery.data?.sessions ?? []).map(toSessionEntry), [sessionsQuery.data])
  const archivedSessions = useMemo(() => (archivedQuery.data?.sessions ?? []).map(toSessionEntry), [archivedQuery.data])

  // Reset the query/sub-page on close so it reopens clean.
  useEffect(() => {
    if (!open) {
      setSearch('')
      setPage(null)
    }
  }, [open])

  const go = useCallback((path: string) => () => navigate(path), [navigate])

  const baseGroups = useMemo<PaletteGroup[]>(() => {
    const settingsTab = (tab: string) => `${SETTINGS_ROUTE}?tab=${tab}`

    return [
      {
        heading: 'Go to',
        items: [
          { icon: Plus, id: 'nav-new', keywords: ['chat', 'create'], label: 'New session', run: go(NEW_CHAT_ROUTE) },
          { icon: Settings, id: 'nav-settings', label: 'Settings', run: go(SETTINGS_ROUTE) },
          {
            icon: Wrench,
            id: 'nav-skills',
            keywords: ['tools', 'toolsets'],
            label: 'Skills & Tools',
            run: go(SKILLS_ROUTE)
          },
          { icon: MessageCircle, id: 'nav-messaging', label: 'Messaging', run: go(MESSAGING_ROUTE) },
          { icon: Package, id: 'nav-artifacts', label: 'Artifacts', run: go(ARTIFACTS_ROUTE) },
          { icon: Clock, id: 'nav-cron', keywords: ['schedule', 'jobs'], label: 'Cron', run: go(CRON_ROUTE) },
          { icon: Users, id: 'nav-profiles', label: 'Profiles', run: go(PROFILES_ROUTE) },
          { icon: Cpu, id: 'nav-agents', label: 'Agents', run: go(AGENTS_ROUTE) }
        ]
      },
      {
        heading: 'Command Center',
        items: [
          {
            icon: Archive,
            id: 'cc-sessions',
            keywords: ['command center', 'sessions', 'pin'],
            label: 'Sessions',
            run: go(`${COMMAND_CENTER_ROUTE}?section=sessions`)
          },
          {
            icon: Activity,
            id: 'cc-system',
            keywords: ['command center', 'system', 'status', 'logs'],
            label: 'System',
            run: go(`${COMMAND_CENTER_ROUTE}?section=system`)
          },
          {
            icon: BarChart3,
            id: 'cc-usage',
            keywords: ['command center', 'usage', 'tokens', 'cost'],
            label: 'Usage',
            run: go(`${COMMAND_CENTER_ROUTE}?section=usage`)
          }
        ]
      },
      {
        // Declared before Settings: cmdk keeps group order, so this keeps the
        // theme/mode pickers on top for "theme"/"color" queries instead of
        // buried under a fuzzy Settings match.
        heading: 'Appearance',
        items: [
          {
            icon: Palette,
            id: 'appearance-theme',
            keywords: ['theme', 'appearance', 'color', 'palette', 'skin', 'dark', 'light', 'look'],
            label: 'Change theme…',
            to: 'theme'
          },
          {
            icon: Sun,
            id: 'appearance-mode',
            keywords: ['appearance', 'color mode', 'brightness', 'dark', 'light', 'system'],
            label: 'Change color mode…',
            to: 'color-mode'
          }
        ]
      },
      {
        heading: 'Settings',
        items: [
          ...SECTIONS.map(section => ({
            icon: section.icon,
            id: `set-config-${section.id}`,
            keywords: ['settings', section.label],
            label: section.label,
            run: go(settingsTab(`config:${section.id}`))
          })),
          ...NON_CONFIG_SETTINGS.map(entry => ({
            icon: entry.icon,
            id: `set-${entry.tab}`,
            keywords: ['settings', ...(entry.keywords ?? [])],
            label: entry.label,
            run: go(settingsTab(entry.tab))
          }))
        ]
      }
    ]
  }, [go])

  // The long, granular lists (settings fields, API keys, MCP servers, archived
  // chats) only surface once the user types — otherwise they'd bury the
  // navigation entries on an empty palette.
  const searchGroups = useMemo<PaletteGroup[]>(() => {
    if (!search.trim()) {
      return []
    }

    const result: PaletteGroup[] = []

    if (sessions.length > 0) {
      result.push({
        heading: 'Sessions',
        items: sessions.map(session => ({
          icon: MessageCircle,
          id: `session-${session.id}`,
          keywords: ['chat', 'session', ...(session.preview ? [session.preview] : [])],
          label: session.title,
          run: go(sessionRoute(session.id))
        }))
      })
    }

    const fieldItems = SECTIONS.flatMap(section =>
      section.keys.map(key => ({
        icon: section.icon,
        id: `field-${key}`,
        keywords: ['settings', key, section.label],
        label: `${section.label}: ${fieldLabel(key)}`,
        run: go(`${SETTINGS_ROUTE}?tab=config:${section.id}&field=${encodeURIComponent(key)}`)
      }))
    )

    result.push({ heading: 'Settings fields', items: fieldItems })

    if (mcpServers.length > 0) {
      result.push({
        heading: 'MCP servers',
        items: mcpServers.map(name => ({
          icon: Wrench,
          id: `mcp-${name}`,
          keywords: ['mcp', 'server', 'tool'],
          label: name,
          run: go(`${SETTINGS_ROUTE}?tab=mcp&server=${encodeURIComponent(name)}`)
        }))
      })
    }

    if (archivedSessions.length > 0) {
      result.push({
        heading: 'Archived chats',
        items: archivedSessions.map(session => ({
          icon: Archive,
          id: `archived-${session.id}`,
          keywords: ['archived', 'chat', 'session', ...(session.preview ? [session.preview] : [])],
          label: session.title,
          run: go(`${SETTINGS_ROUTE}?tab=sessions&session=${encodeURIComponent(session.id)}`)
        }))
      })
    }

    return result
  }, [archivedSessions, go, mcpServers, search, sessions])

  const groups = useMemo(() => [...baseGroups, ...searchGroups], [baseGroups, searchGroups])

  // Nested palette pages (VS Code-style submenus). Reusable: add an entry here
  // and point a root item at it via `to`.
  const subPages = useMemo<Record<string, PalettePage>>(
    () => ({
      theme: {
        title: 'Theme',
        placeholder: 'Choose a theme…',
        // Skins aren't inherently light/dark — the same skin renders in either
        // mode. Group by appearance so picking an entry sets skin + mode at
        // once, and keep the palette open so each pick previews live.
        groups: (['light', 'dark'] as const).map(groupMode => ({
          heading: groupMode === 'light' ? 'Light' : 'Dark',
          items: availableThemes.map(theme => ({
            active: themeName === theme.name && resolvedMode === groupMode,
            icon: groupMode === 'light' ? Sun : Moon,
            id: `theme-${theme.name}-${groupMode}`,
            keepOpen: true,
            keywords: ['theme', 'appearance', 'palette', groupMode, theme.label, theme.description ?? ''],
            label: theme.label,
            run: () => {
              setTheme(theme.name)
              setMode(groupMode)
            }
          }))
        }))
      },
      'color-mode': {
        title: 'Color mode',
        placeholder: 'Choose color mode…',
        groups: [
          {
            heading: 'Color mode',
            items: THEME_MODES.map(entry => ({
              active: mode === entry.mode,
              icon: entry.icon,
              id: `mode-${entry.mode}`,
              keepOpen: true,
              keywords: ['appearance', 'brightness', entry.label],
              label: entry.label,
              run: () => setMode(entry.mode)
            }))
          }
        ]
      }
    }),
    [availableThemes, mode, resolvedMode, setMode, setTheme, themeName]
  )

  const activePage = page ? subPages[page] : null
  const visibleGroups = activePage ? activePage.groups : groups
  const placeholder = activePage ? activePage.placeholder : 'Search commands and settings...'

  const handleSelect = (item: PaletteItem) => {
    if (item.to) {
      setPage(item.to)
      setSearch('')

      return
    }

    item.run?.()

    if (!item.keepOpen) {
      closeCommandPalette()
    }
  }

  return (
    <DialogPrimitive.Root onOpenChange={setCommandPaletteOpen} open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[200] bg-black/15 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[14vh] z-[210] w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-lg duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95"
        >
          <DialogPrimitive.Title className="sr-only">Command palette</DialogPrimitive.Title>
          <Command className="bg-transparent" loop>
            {activePage && (
              <button
                className="flex w-full items-center gap-1.5 border-b border-border px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
                onClick={() => setPage(null)}
                type="button"
              >
                <ChevronLeft className="size-3.5" />
                <span>Back</span>
                <span className="text-muted-foreground/50">/</span>
                <span className="font-medium text-foreground">{activePage.title}</span>
              </button>
            )}
            <CommandInput
              onKeyDown={event => {
                if (!activePage) {
                  return
                }

                // In a submenu: Esc and empty-input Backspace step back out
                // instead of closing the whole palette.
                if (event.key === 'Escape' || (event.key === 'Backspace' && search === '')) {
                  event.preventDefault()
                  event.stopPropagation()
                  setPage(null)
                }
              }}
              onValueChange={setSearch}
              placeholder={placeholder}
              value={search}
            />
            <CommandList className="max-h-[min(24rem,60vh)]">
              <CommandEmpty>No results found.</CommandEmpty>
              {visibleGroups.map(group => (
                <CommandGroup
                  className="**:[[cmdk-group-heading]]:uppercase **:[[cmdk-group-heading]]:tracking-wider **:[[cmdk-group-heading]]:text-[0.6875rem] **:[[cmdk-group-heading]]:text-muted-foreground/70"
                  heading={group.heading}
                  key={group.heading}
                >
                  {group.items.map(item => {
                    const Icon = item.icon

                    return (
                      <CommandItem
                        className="gap-2.5"
                        key={item.id}
                        keywords={item.keywords}
                        onSelect={() => handleSelect(item)}
                        value={`${item.label} ${item.keywords?.join(' ') ?? ''} ${item.id}`}
                      >
                        <Icon className="size-4 shrink-0 text-muted-foreground" />
                        <span className="truncate">{item.label}</span>
                        {item.to ? (
                          <ChevronRight className="ml-auto size-4 shrink-0 text-muted-foreground/70" />
                        ) : (
                          <Check className={cn('ml-auto size-4 text-foreground', !item.active && 'invisible')} />
                        )}
                      </CommandItem>
                    )
                  })}
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
