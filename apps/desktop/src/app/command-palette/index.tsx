import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { HUD_HEADING, HUD_ITEM, HUD_POSITION, HUD_SURFACE, HUD_TEXT } from '@/app/floating-hud'
import { setTerminalTakeover } from '@/app/right-sidebar/store'
import { Command, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { KbdCombo } from '@/components/ui/kbd'
import { getHermesConfigRecord, listAllProfileSessions } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import {
  Activity,
  Archive,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Clock,
  Cpu,
  Download,
  Egg,
  GitBranch,
  Globe,
  type IconComponent,
  Info,
  KeyRound,
  Layers3,
  MessageCircle,
  Monitor,
  Moon,
  Package,
  Palette,
  PawPrint,
  Plus,
  RefreshCw,
  Settings,
  Settings2,
  SlidersHorizontal,
  Starmap,
  Sun,
  Terminal,
  Users,
  Wrench,
  Zap
} from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $repoWorktrees } from '@/store/coding-status'
import {
  $commandPaletteOpen,
  $commandPalettePage,
  closeCommandPalette,
  setCommandPaletteOpen
} from '@/store/command-palette'
import { $bindings } from '@/store/keybinds'
import { openPetGenerate } from '@/store/pet-generate'
import { requestStartWorkSession } from '@/store/projects'
import { runGatewayRestart } from '@/store/system-actions'
import { applyBackendUpdate } from '@/store/updates'
import { luminance } from '@/themes/color'
import { type ThemeMode, useTheme } from '@/themes/context'
import { isUserTheme, resolveTheme } from '@/themes/user-themes'

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
  SKILLS_ROUTE,
  STARMAP_ROUTE
} from '../routes'
import { FIELD_LABELS, SECTIONS } from '../settings/constants'
import { fieldCopyForSchemaKey } from '../settings/field-copy'
import { prettyName } from '../settings/helpers'

import { MarketplaceThemePage } from './marketplace-theme-page'
import { PetInlineToggle, PetPalettePage } from './pet-palette-page'

interface PaletteItem {
  /** Keybind action id — its live combo renders as a hotkey hint. */
  action?: string
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
  /** Optional: a headingless group renders as a bare action row (e.g. the
   *  "Install theme…" entry pinned atop the theme picker). */
  heading?: string
  items: PaletteItem[]
}

// Nested page → its parent, so Back / Esc step up one level instead of closing
// the palette. Pages absent here go straight back to the root list.
const PAGE_PARENTS: Record<string, string> = { 'install-theme': 'theme' }

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

// Ranking happens in React, not cmdk. We score, sort, and prune the groups
// ourselves and hand cmdk an already-ordered list with `shouldFilter={false}`,
// leaving it as pure keyboard/selection machinery. (cmdk's own group
// re-sorting silently no-ops: its sort() queries groups by an internal id that
// never matches the heading text it writes into `data-value`, so groups always
// keep source order — which put a generic keyword match like "Capabilities" on
// top and the auto-highlight on it while an exact "Tools" row sat below.)
//
// cmdk still auto-selects the first DOM item whenever the search changes, so
// rendering best-match-first is what puts the highlight on the best match.
//
// AND semantics: every typed word must appear in the label or keywords. The
// grade rewards matches on the visible label — exact > prefix > whole word >
// word prefix > substring > scattered terms > keyword-only — so typing "tools"
// selects the row that says Tools, not a row that hides it in keywords.
const scoreItem = (item: PaletteItem, needle: string): number => {
  const label = item.label.toLowerCase()
  const keys = (item.keywords ?? []).join(' ').toLowerCase()
  const terms = needle.split(/\s+/).filter(Boolean)

  if (terms.some(term => !label.includes(term) && !keys.includes(term))) {
    return 0
  }

  if (label === needle) {
    return 1
  }

  if (label.startsWith(needle)) {
    return 0.9
  }

  const words = label.split(/[^\p{L}\p{N}]+/u).filter(Boolean)

  if (words.includes(needle)) {
    return 0.85
  }

  if (words.some(word => word.startsWith(needle))) {
    return 0.8
  }

  if (label.includes(needle)) {
    return 0.7
  }

  if (terms.every(term => label.includes(term))) {
    return 0.6
  }

  // Matched only via keywords — the weakest, generic-row signal.
  return 0.4
}

// Order items within each group by score, order groups by their best item, and
// drop everything that doesn't match. Ties keep their original order (stable
// sort), so curated group/item ordering still breaks even scores.
const rankGroups = (groups: PaletteGroup[], search: string): PaletteGroup[] => {
  const needle = normalize(search)

  if (!needle) {
    return groups
  }

  return groups
    .map(group => {
      const scored = group.items
        .map(item => ({ item, score: scoreItem(item, needle) }))
        .filter(entry => entry.score > 0)
        .sort((a, b) => b.score - a.score)

      return { group: { ...group, items: scored.map(entry => entry.item) }, max: scored[0]?.score ?? 0 }
    })
    .filter(entry => entry.max > 0)
    .sort((a, b) => b.max - a.max)
    .map(entry => entry.group)
}

// cmdk selection values must be unique; labels alone can repeat (the same
// theme lists under both Light and Dark). The id suffix disambiguates.
const paletteValue = (item: PaletteItem): string => `${item.label}\u0001${item.id}`

// Hermes session ids: <YYYYMMDD>_<HHMMSS>_<6 hex>. Used to offer a direct
// "Go to session ‹id›" jump for ids that aren't in the recent-200 list.
const SESSION_ID_RE = /^\d{8}_\d{6}_[a-f0-9]{6}$/

type SessionRow = Awaited<ReturnType<typeof listAllProfileSessions>>['sessions'][number]

const toSessionEntry = (session: SessionRow): SessionEntry => ({
  id: session.id,
  preview: session.preview ?? undefined,
  title: sessionTitle(session)
})

type NonConfigSettingsLabel =
  | 'about'
  | 'archivedChats'
  | 'gateway'
  | 'keysSettings'
  | 'keysTools'
  | 'mcp'
  | 'providerAccounts'
  | 'providerApiKeys'

const NON_CONFIG_SETTINGS: ReadonlyArray<{
  icon: IconComponent
  keywords?: string[]
  labelKey: NonConfigSettingsLabel
  tab: string
}> = [
  {
    icon: Zap,
    keywords: ['accounts', 'sign in', 'oauth', 'login', 'subscription', 'models', 'anthropic', 'openai'],
    labelKey: 'providerAccounts',
    tab: 'providers&pview=accounts'
  },
  {
    icon: KeyRound,
    keywords: ['providers', 'api key', 'keys', 'secrets', 'tokens'],
    labelKey: 'providerApiKeys',
    tab: 'providers&pview=keys'
  },
  { icon: Globe, keywords: ['connection', 'messaging'], labelKey: 'gateway', tab: 'gateway' },
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
  { icon: Archive, keywords: ['history', 'archived'], labelKey: 'archivedChats', tab: 'sessions' },
  { icon: Info, keywords: ['version', 'about'], labelKey: 'about', tab: 'about' }
]

const THEME_MODES: ReadonlyArray<{ icon: IconComponent; mode: ThemeMode }> = [
  { icon: Sun, mode: 'light' },
  { icon: Moon, mode: 'dark' },
  { icon: Monitor, mode: 'system' }
]

// Which Light/Dark groups a theme belongs in. Built-ins render in both modes
// (the engine synthesises the missing side). Imported VS Code themes only carry
// the variant(s) the extension shipped — a single dark theme like Dracula lives
// under Dark only, while a GitHub/Solarized family (light + dark) lives in both.
function themeSupportsMode(name: string, target: 'light' | 'dark'): boolean {
  if (!isUserTheme(name)) {
    return true
  }

  const resolved = resolveTheme(name)

  if (!resolved) {
    return true
  }

  const background =
    target === 'dark' ? (resolved.darkColors ?? resolved.colors).background : resolved.colors.background

  return target === 'dark' ? luminance(background) <= 0.5 : luminance(background) > 0.5
}

export function CommandPalette() {
  const { t } = useI18n()
  const open = useStore($commandPaletteOpen)
  const pendingPage = useStore($commandPalettePage)
  const bindings = useStore($bindings)
  const worktrees = useStore($repoWorktrees)
  const navigate = useNavigate()
  const { availableThemes, resolvedMode, setMode, setTheme, themeName } = useTheme()
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
    queryFn: () => listAllProfileSessions(200, 1, 'exclude'),
    enabled: open
  })

  const archivedQuery = useQuery({
    queryKey: ['command-palette', 'archived'],
    queryFn: () => listAllProfileSessions(200, 0, 'only'),
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

  // Deep-link into a nested page (e.g. `/pet list` → pets picker).
  useEffect(() => {
    if (open && pendingPage) {
      setPage(pendingPage)
      $commandPalettePage.set(null)
    }
  }, [open, pendingPage])

  const go = useCallback((path: string) => () => navigate(path), [navigate])

  // Step up one nested page (or back to the root list), clearing the filter so
  // the parent page doesn't reopen mid-search.
  const goBack = useCallback(() => {
    setSearch('')
    setPage(prev => (prev ? (PAGE_PARENTS[prev] ?? null) : null))
  }, [])

  const settingsSectionLabel = useCallback(
    (section: (typeof SECTIONS)[number]) => t.settings.sections[section.id] ?? section.label,
    [t.settings.sections]
  )

  const configFieldLabel = useCallback(
    (key: string) =>
      fieldCopyForSchemaKey(t.settings.fieldLabels, key) ??
      fieldCopyForSchemaKey(FIELD_LABELS, key) ??
      prettyName(key.split('.').pop() ?? key),
    [t.settings.fieldLabels]
  )

  const baseGroups = useMemo<PaletteGroup[]>(() => {
    const settingsTab = (tab: string) => `${SETTINGS_ROUTE}?tab=${tab}`
    const cc = t.commandCenter

    // The active repo's worktrees → "new conversation in <branch>". This is the
    // ⌘K-typed "I want to work on <branch>" reflex: each entry seeds a fresh
    // session anchored to that worktree's checkout (requestStartWorkSession),
    // so git is the source of truth and edits land in the right tree.
    const branchGroup: PaletteGroup[] =
      worktrees.length > 0
        ? [
            {
              heading: cc.branches,
              items: worktrees.map(wt => {
                const name = wt.branch?.trim() || wt.path.split('/').pop() || wt.path

                return {
                  icon: GitBranch,
                  id: `worktree-${wt.path}`,
                  keywords: ['branch', 'worktree', 'switch', name, wt.path],
                  label: cc.startInBranch(name),
                  run: () => requestStartWorkSession(wt.path)
                }
              })
            }
          ]
        : []

    return [
      {
        heading: cc.goTo,
        items: [
          {
            action: 'session.new',
            icon: Plus,
            id: 'nav-new',
            keywords: ['chat', 'create'],
            label: cc.nav.newChat.title,
            run: go(NEW_CHAT_ROUTE)
          },
          {
            action: 'view.showTerminal',
            icon: Terminal,
            id: 'nav-terminal',
            keywords: ['terminal', 'shell', 'console'],
            label: t.keybinds.actions['view.showTerminal'],
            run: () => setTerminalTakeover(true)
          },
          {
            action: 'nav.settings',
            icon: Settings,
            id: 'nav-settings',
            label: cc.nav.settings.title,
            run: go(SETTINGS_ROUTE)
          },
          {
            action: 'nav.skills',
            icon: Wrench,
            id: 'nav-skills',
            keywords: ['skills', 'tools', 'toolsets', 'mcp', 'capabilities'],
            label: cc.nav.skills.title,
            run: go(SKILLS_ROUTE)
          },
          {
            action: 'nav.messaging',
            icon: MessageCircle,
            id: 'nav-messaging',
            label: cc.nav.messaging.title,
            run: go(MESSAGING_ROUTE)
          },
          {
            action: 'nav.artifacts',
            icon: Package,
            id: 'nav-artifacts',
            label: cc.nav.artifacts.title,
            run: go(ARTIFACTS_ROUTE)
          },
          {
            action: 'nav.cron',
            icon: Clock,
            id: 'nav-cron',
            keywords: ['schedule', 'jobs'],
            label: t.shell.statusbar.cron,
            run: go(CRON_ROUTE)
          },
          { action: 'nav.profiles', icon: Users, id: 'nav-profiles', label: t.profiles.title, run: go(PROFILES_ROUTE) },
          { action: 'nav.agents', icon: Cpu, id: 'nav-agents', label: t.agents.title, run: go(AGENTS_ROUTE) },
          {
            icon: Starmap,
            id: 'nav-starmap',
            keywords: ['star map', 'memory', 'memories', 'skills', 'graph', 'learning', 'constellation'],
            label: t.starmap.title,
            run: go(STARMAP_ROUTE)
          }
        ]
      },
      ...branchGroup,
      {
        heading: cc.commandCenter,
        items: [
          {
            icon: Archive,
            id: 'cc-sessions',
            keywords: ['command center', 'sessions', 'pin'],
            label: cc.sections.sessions,
            run: go(`${COMMAND_CENTER_ROUTE}?section=sessions`)
          },
          {
            icon: Activity,
            id: 'cc-system',
            keywords: ['command center', 'system', 'status', 'logs'],
            label: cc.sections.system,
            run: go(`${COMMAND_CENTER_ROUTE}?section=system`)
          },
          {
            icon: BarChart3,
            id: 'cc-usage',
            keywords: ['command center', 'usage', 'tokens', 'cost'],
            label: cc.sections.usage,
            run: go(`${COMMAND_CENTER_ROUTE}?section=usage`)
          },
          {
            icon: RefreshCw,
            id: 'cc-restart-gateway',
            keywords: ['gateway', 'restart', 'messaging', 'reconnect', 'system'],
            label: cc.restartGateway,
            run: () => void runGatewayRestart()
          },
          {
            icon: Download,
            id: 'cc-update-hermes',
            keywords: ['update', 'upgrade', 'hermes', 'version', 'system', 'restart'],
            label: cc.updateHermes,
            run: () => void applyBackendUpdate()
          }
        ]
      },
      {
        // Declared before Settings: cmdk keeps group order, so this keeps the
        // theme/mode pickers on top for "theme"/"color" queries instead of
        // buried under a fuzzy Settings match.
        heading: cc.appearance,
        items: [
          {
            icon: Palette,
            id: 'appearance-theme',
            keywords: ['theme', 'appearance', 'color', 'palette', 'skin', 'dark', 'light', 'look'],
            label: cc.changeTheme,
            to: 'theme'
          },
          {
            icon: Sun,
            id: 'appearance-mode',
            keywords: ['appearance', 'color mode', 'brightness', 'dark', 'light', 'system'],
            label: cc.changeColorMode,
            to: 'color-mode'
          },
          {
            icon: PawPrint,
            id: 'appearance-pets',
            keywords: ['pet', 'petdex', 'mascot', 'pets', '/pet', 'paw'],
            label: cc.pets.title,
            to: 'pets'
          },
          {
            icon: Egg,
            id: 'appearance-generate-pet',
            keywords: ['pet', 'generate', 'create', 'make', 'new pet', 'mascot', 'hatch', 'ai'],
            label: cc.generatePet.title,
            run: () => openPetGenerate()
          }
        ]
      },
      {
        heading: cc.settings,
        items: [
          ...SECTIONS.map(section => ({
            icon: section.icon,
            id: `set-config-${section.id}`,
            keywords: ['settings', section.label, settingsSectionLabel(section)],
            label: settingsSectionLabel(section),
            run: go(settingsTab(`config:${section.id}`))
          })),
          ...NON_CONFIG_SETTINGS.map(entry => ({
            icon: entry.icon,
            id: `set-${entry.tab}`,
            keywords: ['settings', ...(entry.keywords ?? [])],
            label: t.settings.nav[entry.labelKey],
            run: go(settingsTab(entry.tab))
          }))
        ]
      }
    ]
  }, [go, settingsSectionLabel, t, worktrees])

  // The long, granular lists (settings fields, API keys, MCP servers, archived
  // chats) only surface once the user types — otherwise they'd bury the
  // navigation entries on an empty palette.
  const searchGroups = useMemo<PaletteGroup[]>(() => {
    if (!search.trim()) {
      return []
    }

    const result: PaletteGroup[] = []

    // Paste a raw session id → jump straight to it, even if it predates the
    // recent-200 window the lists below are built from.
    const directId = search.trim()

    if (SESSION_ID_RE.test(directId)) {
      result.push({
        items: [
          {
            icon: MessageCircle,
            id: `goto-${directId}`,
            keywords: ['session', 'id', 'go to', directId],
            label: `${t.commandCenter.goToSession} ${directId}`,
            run: go(sessionRoute(directId))
          }
        ]
      })
    }

    // Deep-link straight to a Capabilities sub-tab. The root "Go to" entry only
    // lands on the top-level Skills view; typing "mcp"/"tools"/"skills" should
    // jump to the exact tab (matches the "not just the top lvl" ask).
    const capLabel = t.commandCenter.nav.skills.title

    result.push({
      heading: capLabel,
      items: [
        {
          icon: Wrench,
          id: 'cap-skills',
          keywords: ['skills', 'capabilities'],
          label: `${capLabel}: ${t.skills.tabSkills}`,
          run: go(`${SKILLS_ROUTE}?tab=skills`)
        },
        {
          icon: SlidersHorizontal,
          id: 'cap-toolsets',
          keywords: ['tools', 'toolsets', 'capabilities'],
          label: `${capLabel}: ${t.skills.tabToolsets}`,
          run: go(`${SKILLS_ROUTE}?tab=toolsets`)
        },
        {
          icon: Layers3,
          id: 'cap-mcp',
          keywords: ['mcp', 'servers', 'tools', 'capabilities', 'model context protocol'],
          label: `${capLabel}: ${t.skills.tabMcp}`,
          run: go(`${SKILLS_ROUTE}?tab=mcp`)
        }
      ]
    })

    // Apply a theme directly from the root search (e.g. "nous" → Nous). Live
    // preview via keepOpen, mirroring the nested theme picker. If the theme
    // can't render the current light/dark mode, flip to the one it supports.
    result.push({
      heading: t.settings.appearance.themeTitle,
      items: availableThemes.map(theme => ({
        icon: Palette,
        id: `search-theme-${theme.name}`,
        keepOpen: true,
        keywords: ['theme', 'appearance', 'color', 'skin', theme.name, theme.description],
        label: theme.label,
        run: () => {
          setTheme(theme.name)

          if (!themeSupportsMode(theme.name, resolvedMode)) {
            setMode(resolvedMode === 'dark' ? 'light' : 'dark')
          }
        }
      }))
    })

    // Switch light/dark/system directly (typing "dark" shouldn't require the
    // nested color-mode page).
    result.push({
      heading: t.settings.appearance.colorMode,
      items: THEME_MODES.map(entry => ({
        icon: entry.icon,
        id: `search-mode-${entry.mode}`,
        keepOpen: true,
        keywords: ['appearance', 'color mode', 'brightness', entry.mode, t.settings.modeOptions[entry.mode].label],
        label: t.settings.modeOptions[entry.mode].label,
        run: () => setMode(entry.mode)
      }))
    })

    if (sessions.length > 0) {
      result.push({
        heading: t.commandCenter.sections.sessions,
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
        keywords: ['settings', key, section.label, settingsSectionLabel(section)],
        label: `${settingsSectionLabel(section)}: ${configFieldLabel(key)}`,
        run: go(`${SETTINGS_ROUTE}?tab=config:${section.id}&field=${encodeURIComponent(key)}`)
      }))
    )

    result.push({ heading: t.commandCenter.settingsFields, items: fieldItems })

    if (mcpServers.length > 0) {
      result.push({
        heading: t.commandCenter.mcpServers,
        items: mcpServers.map(name => ({
          icon: Wrench,
          id: `mcp-${name}`,
          keywords: ['mcp', 'server', 'tool'],
          label: name,
          run: go(`${SKILLS_ROUTE}?tab=mcp&server=${encodeURIComponent(name)}`)
        }))
      })
    }

    if (archivedSessions.length > 0) {
      result.push({
        heading: t.commandCenter.archivedChats,
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
  }, [
    archivedSessions,
    availableThemes,
    configFieldLabel,
    go,
    mcpServers,
    resolvedMode,
    search,
    sessions,
    setMode,
    setTheme,
    settingsSectionLabel,
    t
  ])

  const groups = useMemo(() => [...baseGroups, ...searchGroups], [baseGroups, searchGroups])

  // Nested palette pages (VS Code-style submenus). Reusable: add an entry here
  // and point a root item at it via `to`.
  const subPages = useMemo<Record<string, PalettePage>>(
    () => ({
      theme: {
        title: t.settings.appearance.themeTitle,
        placeholder: t.settings.appearance.themeDesc,
        groups: [
          // Pinned at the top: drills into the Marketplace browser.
          {
            items: [
              {
                icon: Download,
                id: 'theme-install',
                keywords: ['install', 'marketplace', 'vscode', 'vs code', 'download', 'new', 'color'],
                label: t.commandCenter.installTheme.title,
                to: 'install-theme'
              }
            ]
          },
          // Built-ins and imported families list under the mode(s) they support;
          // picking sets skin + mode at once. A multi-variant import (GitHub,
          // Solarized) appears in both groups and switches variants with the mode.
          ...(['light', 'dark'] as const).map(groupMode => ({
            heading: groupMode === 'light' ? t.settings.modeOptions.light.label : t.settings.modeOptions.dark.label,
            items: availableThemes
              .filter(theme => themeSupportsMode(theme.name, groupMode))
              .map(theme => ({
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
        ]
      },
      'color-mode': {
        title: t.settings.appearance.colorMode,
        placeholder: t.settings.appearance.colorModeDesc,
        groups: [
          {
            heading: t.settings.appearance.colorMode,
            items: THEME_MODES.map(entry => ({
              icon: entry.icon,
              id: `mode-${entry.mode}`,
              keepOpen: true,
              keywords: ['appearance', 'brightness', t.settings.modeOptions[entry.mode].label],
              label: t.settings.modeOptions[entry.mode].label,
              run: () => setMode(entry.mode)
            }))
          }
        ]
      },
      // Server-driven page: browse petdex gallery, adopt/switch, toggle off.
      pets: {
        title: t.commandCenter.pets.title,
        placeholder: t.commandCenter.pets.placeholder,
        groups: []
      },
      // Server-driven page: items come from the Marketplace, rendered by
      // <MarketplaceThemePage> (loader + live search + per-row install).
      'install-theme': {
        title: t.commandCenter.installTheme.pageTitle,
        placeholder: t.commandCenter.installTheme.placeholder,
        groups: []
      }
    }),
    [availableThemes, resolvedMode, setMode, setTheme, t, themeName]
  )

  const activePage = page ? subPages[page] : null
  const unrankedGroups = activePage ? activePage.groups : groups
  const visibleGroups = useMemo(() => rankGroups(unrankedGroups, search), [unrankedGroups, search])
  const placeholder = activePage ? activePage.placeholder : t.commandCenter.searchPlaceholder

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
        {/* Transparent overlay: keeps click-away + focus trap, but no dim/blur. */}
        <DialogPrimitive.Overlay className="fixed inset-0 z-[200]" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className={cn(
            HUD_POSITION,
            HUD_SURFACE,
            'z-[210] w-[min(34rem,calc(100vw-2rem))] overflow-hidden duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95'
          )}
        >
          <DialogPrimitive.Title className="sr-only">{t.commandCenter.paletteTitle}</DialogPrimitive.Title>
          <Command className="bg-transparent" loop shouldFilter={false}>
            {activePage && (
              <button
                className="flex w-full items-center gap-1.5 border-b border-border px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
                onClick={goBack}
                type="button"
              >
                <ChevronLeft className="size-3.5" />
                <span>{t.commandCenter.back}</span>
                <span className="text-muted-foreground/50">/</span>
                <span className="font-medium text-foreground">{activePage.title}</span>
              </button>
            )}
            <CommandInput
              className={HUD_TEXT}
              onKeyDown={event => {
                if (!activePage) {
                  return
                }

                // In a submenu: Esc and empty-input Backspace step back out
                // instead of closing the whole palette.
                if (event.key === 'Escape' || (event.key === 'Backspace' && search === '')) {
                  event.preventDefault()
                  event.stopPropagation()
                  goBack()

                  return
                }
              }}
              onValueChange={setSearch}
              placeholder={placeholder}
              right={page === 'pets' ? <PetInlineToggle /> : undefined}
              value={search}
            />
            <CommandList className="dt-portal-scrollbar max-h-[min(20rem,56vh)]">
              {/* Server-driven pages render their own list; the rest show groups. */}
              {page === 'pets' ? (
                <PetPalettePage
                  onGenerate={() => {
                    closeCommandPalette()
                    openPetGenerate()
                  }}
                  search={search}
                />
              ) : page === 'install-theme' ? (
                <MarketplaceThemePage onPickTheme={setTheme} search={search} />
              ) : (
                <>
                  {/* Filtering happens in rankGroups, so cmdk's own CommandEmpty
                      (keyed to its internal filter count) would never fire. */}
                  {visibleGroups.length === 0 && (
                    <div className="py-6 text-center text-sm text-muted-foreground">{t.commandCenter.noResults}</div>
                  )}
                  {visibleGroups.map((group, index) => (
                    <CommandGroup
                      className={HUD_HEADING}
                      heading={group.heading}
                      key={group.heading ?? `palette-group-${index}`}
                    >
                      {group.items.map(item => {
                        const Icon = item.icon
                        const combo = item.action ? bindings[item.action]?.[0] : undefined

                        return (
                          <CommandItem
                            className={cn(HUD_ITEM, HUD_TEXT)}
                            key={item.id}
                            keywords={item.keywords}
                            onSelect={() => handleSelect(item)}
                            value={paletteValue(item)}
                          >
                            <Icon className="size-3.5 shrink-0 text-muted-foreground" />
                            <span className="truncate">{item.label}</span>
                            {combo && <KbdCombo className="ml-auto opacity-55" combo={combo} size="sm" />}
                            {item.to && (
                              <ChevronRight
                                className={cn('size-3.5 shrink-0 text-muted-foreground/70', !combo && 'ml-auto')}
                              />
                            )}
                          </CommandItem>
                        )
                      })}
                    </CommandGroup>
                  ))}
                </>
              )}
            </CommandList>
          </Command>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
