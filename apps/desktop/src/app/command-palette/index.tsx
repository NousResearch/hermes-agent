import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { memo, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import {
  HUD_HEADING,
  HUD_ITEM,
  HUD_NOTE,
  HUD_NOTE_VARIANT,
  HUD_POSITION,
  HUD_SURFACE,
  HUD_TEXT
} from '@/app/floating-hud'
import { setTerminalTakeover } from '@/app/right-sidebar/store'
import { codiconIcon } from '@/components/ui/codicon'
import { Command, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { KbdCombo } from '@/components/ui/kbd'
import { getHermesConfigRecord, listAllProfileSessions } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import {
  Activity,
  AppWindow,
  Archive,
  BarChart3,
  Check,
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
import { resolveVersionStatus } from '@/lib/version-status'
import { $repoWorktrees } from '@/store/coding-status'
import {
  $commandPaletteOpen,
  $commandPalettePage,
  closeCommandPalette,
  setCommandPaletteOpen
} from '@/store/command-palette'
import { $bindings } from '@/store/keybinds'
import { $dismissedAutoProjectIds, filterVisibleProjects } from '@/store/layout'
import { openPetGenerate } from '@/store/pet-generate'
import { $projectTree, goToProject, openFolderAsProject, requestStartWorkSession } from '@/store/projects'
import { $connection } from '@/store/session'
import { runGatewayRestart } from '@/store/system-actions'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateStatus,
  requestActiveUpdate
} from '@/store/updates'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'
import { luminance } from '@/themes/color'
import { type ThemeMode, useTheme } from '@/themes/context'
import { isUserTheme, resolveTheme } from '@/themes/user-themes'

import { openSession, openSessionIntentFromModifiers } from '../open-session'
import {
  AGENTS_ROUTE,
  ARTIFACTS_ROUTE,
  COMMAND_CENTER_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  navigateToWorkspacePage,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  SETTINGS_ROUTE,
  SKILLS_ROUTE,
  STARMAP_ROUTE
} from '../routes'
import { FIELD_LABELS, SECTIONS } from '../settings/constants'
import { fieldCopyForSchemaKey } from '../settings/field-copy'
import { prettyName } from '../settings/helpers'

import { usePaletteContributions } from './contrib'
import { MarketplaceThemePage } from './marketplace-theme-page'
import { PetInlineToggle, PetPalettePage } from './pet-palette-page'

interface PaletteItem {
  /** Keybind action id — its live combo renders as a hotkey hint. */
  action?: string
  /** Renders a trailing check: this row IS the current setting (theme, mode). */
  active?: boolean
  /** Static trailing combo hint for a modifier-variant select (e.g. `mod+enter`). */
  comboHint?: string
  /** Short note beside the label — state the row acts on (a version, a count). */
  detail?: string
  /** `state` when the row will change what `detail` says (a toggle's on/off). */
  detailVariant?: keyof typeof HUD_NOTE_VARIANT
  icon: IconComponent
  id: string
  /** Keep the palette open after running (live-preview pickers like theme/mode). */
  keepOpen?: boolean
  keywords?: string[]
  label: string
  /** Label shown while ⌘/⌃ is held — previews the modifier-variant action. */
  modLabel?: string
  /**
   * When set, ⌘/⌃-select (or ⌘-Enter) opens a new tab and ⇧⌘-select pops a
   * window — matching sidebar session rows. Plain select stays in-place.
   * Receives the last selector event so the modifiers can be read.
   */
  runWithEvent?: (event?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => void
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
  git_branch?: null | string
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

const EMPTY_GROUPS: PaletteGroup[] = []

// Backstop only. The palette normally retires on the content's real
// `animationend`, so the CSS owns the close duration; this just guarantees the
// body can't stay mounted forever somewhere animations never run (jsdom,
// `animation: none`). Deliberately longer than any plausible exit so it never
// races the real signal and truncates the fade.
const EXIT_FALLBACK_MS = 1000

/**
 * The palette's row list, split out so an OPENING palette paints before it
 * renders rows. This component mounts with the portal, so `useDeferredValue`'s
 * initial value applies per open: the first commit is the frame + input
 * (instant), and the several-hundred-row list arrives in an interruptible
 * follow-up render. Opening ⌘K must never wait on building the list.
 */
const PaletteGroups = memo(function PaletteGroups({
  bindings,
  groups,
  modHeld,
  noResultsLabel,
  onSelectItem,
  onSelectMods,
  search
}: {
  bindings: Record<string, string[]>
  groups: PaletteGroup[]
  modHeld: boolean
  noResultsLabel: string
  onSelectItem: (item: PaletteItem) => void
  onSelectMods: (event: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) => void
  search: string
}) {
  const deferred = useDeferredValue(groups, EMPTY_GROUPS)
  // While the rows are still catching up, an empty list means "not rendered
  // yet", not "nothing matched" — don't flash the empty state on open.
  const pending = deferred !== groups

  return (
    <>
      {/* Filtering happens in rankGroups, so cmdk's own CommandEmpty
          (keyed to its internal filter count) would never fire. */}
      {deferred.length === 0 && !pending && (
        <div className="py-6 text-center text-sm text-muted-foreground">{noResultsLabel}</div>
      )}
      {deferred.map((group, index) => (
        <CommandGroup className={HUD_HEADING} heading={group.heading} key={group.heading ?? `palette-group-${index}`}>
          {group.items.map(item => (
            <PaletteRow
              bindings={bindings}
              item={item}
              key={item.id}
              modHeld={modHeld}
              onSelectItem={onSelectItem}
              onSelectMods={onSelectMods}
              search={search}
            />
          ))}
        </CommandGroup>
      ))}
    </>
  )
})

const PaletteRow = memo(function PaletteRow({
  bindings,
  item,
  modHeld,
  onSelectMods,
  onSelectItem,
  search
}: {
  bindings: Record<string, string[]>
  item: PaletteItem
  modHeld: boolean
  onSelectMods: (event: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) => void
  onSelectItem: (item: PaletteItem) => void
  search: string
}) {
  const Icon = item.icon
  // The row's live keybind, else a static modifier-variant hint (⌘↵). One slot,
  // so every downstream `ml-auto` fallback below keeps working unchanged.
  const combo = (item.action ? bindings[item.action]?.[0] : undefined) ?? item.comboHint
  // While ⌘/⌃ is held, a row with a modifier variant previews it: the label
  // swaps to the variant's copy so Enter reads as what it will actually do.
  const modPreview = modHeld && Boolean(item.modLabel)

  return (
    <CommandItem
      className={cn(HUD_ITEM, HUD_TEXT)}
      keywords={item.keywords}
      onMouseDown={onSelectMods}
      onSelect={() => onSelectItem(item)}
      value={paletteValue(item)}
    >
      <Icon className="size-3.5 shrink-0 text-muted-foreground" />
      <span className={cn('truncate', modPreview && 'text-muted-foreground/80')}>
        {modPreview ? (
          item.modLabel
        ) : (
          /* Same per-term split as scoreItem's AND matcher, so the emphasis
             shows exactly which words earned the row its rank. */
          <HighlightMatches query={search.split(/\s+/)} text={item.label} />
        )}
      </span>
      {item.detail && (
        <span className={cn(HUD_NOTE, HUD_NOTE_VARIANT[item.detailVariant ?? 'muted'])}>{item.detail}</span>
      )}
      {combo && (
        <KbdCombo className={cn('ml-auto', modPreview ? 'opacity-90' : 'opacity-55')} combo={combo} size="sm" />
      )}
      {item.to && <ChevronRight className={cn('size-3.5 shrink-0 text-muted-foreground/70', !combo && 'ml-auto')} />}
      {item.active && <Check className={cn('size-3.5 shrink-0 text-primary', !combo && !item.to && 'ml-auto')} />}
    </CommandItem>
  )
})

// Hermes session ids: <YYYYMMDD>_<HHMMSS>_<6 hex>. Used to offer a direct
// "Go to session ‹id›" jump for ids that aren't in the recent-200 list.
const SESSION_ID_RE = /^\d{8}_\d{6}_[a-f0-9]{6}$/

// A typed/pasted folder path: absolute (`/…`) or a Windows drive (`C:\…`).
// Deliberately NOT `~/…`: the upsert's membership check (projectIdForCwd)
// compares literal strings against the tree's absolute paths, so an unexpanded
// home path would always miss and double-create.
const FOLDER_PATH_RE = /^(\/|[A-Za-z]:[/\\]).+/

type SessionRow = Awaited<ReturnType<typeof listAllProfileSessions>>['sessions'][number]

const toSessionEntry = (session: SessionRow): SessionEntry => ({
  git_branch: session.git_branch ?? null,
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
    keywords: ['providers', 'api key', 'keys', 'secrets', 'tokens', 'egress', 'iron proxy', 'sandbox proxy'],
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
    keywords: ['gateway', 'proxy', 'server', 'webhook', 'env', 'egress proxy', 'iron proxy'],
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

/**
 * ⌘K is an overlay that is stateful to itself: pressing it must open a frame
 * immediately, and must not be held up by whatever else the shell is doing. So
 * the mounted cost of a CLOSED palette is one store subscription and nothing
 * else.
 *
 * Everything expensive — a dozen store subscriptions (connection, update
 * status/apply, keybinds, worktrees, projects, theme, i18n), three server
 * queries, and the group builders that assemble a few hundred rows — lives in
 * `CommandPaletteBody`, which only exists while the palette is on screen.
 * Before this split those hooks ran on every render of the always-mounted
 * component: an in-flight update rewrote `$updateApply` per progress line and
 * rebuilt the entire row set each time, for a surface nobody could see.
 *
 * `mounted` lags `open` by the close animation rather than tracking it exactly.
 * Unmounting the body the instant `open` flips false would rip the content out
 * of the tree before Radix could play `data-[state=closed]`, so the overlay
 * would vanish instead of closing. The body reports its own exit via
 * `onExited` (the content's real `animationend`), so nothing here has to know
 * how long that animation is — the CSS owns the duration.
 *
 * The `openCount` key remounts the body per open, which is what lets local
 * search/sub-page state reset without a close effect.
 */
export function CommandPalette() {
  const open = useStore($commandPaletteOpen)
  const [mounted, setMounted] = useState(open)
  const [openCount, setOpenCount] = useState(0)

  const retire = useCallback(() => {
    // Only retire the body if the palette is still closed — a reopen mid-fade
    // must not unmount the fresh instance.
    if (!$commandPaletteOpen.get()) {
      setMounted(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      setOpenCount(count => count + 1)
      setMounted(true)

      return
    }

    // Safety net for environments where the exit animation never runs (jsdom,
    // `animation: none`), so the body can't be stranded mounted. The real
    // unmount is `onExited` below; whichever fires first wins.
    const timer = setTimeout(retire, EXIT_FALLBACK_MS)

    return () => clearTimeout(timer)
  }, [open, retire])

  return (
    <DialogPrimitive.Root onOpenChange={setCommandPaletteOpen} open={open}>
      {mounted && <CommandPaletteBody key={openCount} onExited={retire} />}
    </DialogPrimitive.Root>
  )
}

function CommandPaletteBody({ onExited }: { onExited: () => void }) {
  const { t } = useI18n()
  const pendingPage = useStore($commandPalettePage)
  const bindings = useStore($bindings)
  const worktrees = useStore($repoWorktrees)
  const projectTree = useStore($projectTree)
  const dismissedAutoProjects = useStore($dismissedAutoProjectIds)
  const navigate = useNavigate()
  const { availableThemes, mode, resolvedMode, setMode, setTheme, themeName } = useTheme()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState<string | null>(null)

  // The Update row names the same install the statusbar names — same target
  // selection, same resolver. Reduced to the label string: an in-flight apply
  // rewrites these stores on every progress line, and only a changed string
  // should rebuild the palette's groups.
  const connection = useStore($connection)
  const desktopVersion = useStore($desktopVersion)
  const clientStatus = useStore($updateStatus)
  const clientApply = useStore($updateApply)
  const backendStatus = useStore($backendUpdateStatus)
  const backendApply = useStore($backendUpdateApply)

  const updateVersionLabel = useMemo(() => {
    const backend = connection?.mode === 'remote'
    const apply = backend ? backendApply : clientApply
    const status = backend ? backendStatus : clientStatus

    return resolveVersionStatus({
      applying: apply.applying || apply.stage === 'restart',
      behind: status?.behind ?? 0,
      copy: t.shell.statusbar,
      remote: backend,
      restarting: apply.stage === 'restart',
      sha: status?.currentSha?.slice(0, 7) ?? null,
      target: backend ? 'backend' : 'client',
      updateAvailable: status?.updateAvailable,
      version: backend ? status?.currentVersion : desktopVersion?.appVersion
    }).label
  }, [backendApply, backendStatus, clientApply, clientStatus, connection?.mode, desktopVersion?.appVersion, t])

  // cmdk's onSelect doesn't forward the triggering event — keep the last
  // click/keydown modifiers so session rows can honour ⌘-Enter / ⌘-click.
  const lastSelectMods = useRef<{ ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }>({
    ctrlKey: false,
    metaKey: false,
    shiftKey: false
  })

  const noteSelectMods = (event: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) => {
    lastSelectMods.current = {
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
      shiftKey: event.shiftKey
    }
  }

  // Live ⌘/⌃-held state while the palette is open: rows with a modifier
  // variant (projects) preview it by swapping their label. Window-level
  // listeners because focus sits in the search input; blur clears so a
  // ⌘-Tab away doesn't strand the preview on.
  const [modHeld, setModHeld] = useState(false)

  useEffect(() => {
    const sync = (event: KeyboardEvent) => setModHeld(event.metaKey || event.ctrlKey)
    const clear = () => setModHeld(false)

    window.addEventListener('keydown', sync, { capture: true })
    window.addEventListener('keyup', sync, { capture: true })
    window.addEventListener('blur', clear)

    return () => {
      window.removeEventListener('keydown', sync, { capture: true })
      window.removeEventListener('keyup', sync, { capture: true })
      window.removeEventListener('blur', clear)
    }
  }, [])

  // Server-backed sources for the type-to-search groups. This component only
  // exists while the palette is open, so the queries are inherently lazy — no
  // `enabled` gate needed. react-query handles caching/dedup/staleness, so a
  // reopen paints from cache and revalidates in the background.
  const configQuery = useQuery({
    queryKey: ['command-palette', 'config'],
    queryFn: getHermesConfigRecord
  })

  const sessionsQuery = useQuery({
    queryKey: ['command-palette', 'sessions'],
    queryFn: () => listAllProfileSessions(200, 1, 'exclude')
  })

  const archivedQuery = useQuery({
    queryKey: ['command-palette', 'archived'],
    queryFn: () => listAllProfileSessions(200, 0, 'only')
  })

  const mcpServers = useMemo(() => {
    const raw = configQuery.data?.mcp_servers

    return raw && typeof raw === 'object' && !Array.isArray(raw)
      ? Object.keys(raw as Record<string, unknown>).sort()
      : []
  }, [configQuery.data])

  const sessions = useMemo(() => (sessionsQuery.data?.sessions ?? []).map(toSessionEntry), [sessionsQuery.data])
  const archivedSessions = useMemo(() => (archivedQuery.data?.sessions ?? []).map(toSessionEntry), [archivedQuery.data])

  // Search/sub-page are local to a mount, and this component remounts per open
  // (keyed by open count), so each open starts clean without a reset effect.

  // Deep-link into a nested page (e.g. `/pet list` → pets picker).
  useEffect(() => {
    if (pendingPage) {
      setPage(pendingPage)
      $commandPalettePage.set(null)
    }
  }, [pendingPage])

  const go = useCallback((path: string) => () => navigateToWorkspacePage(navigate, path), [navigate])

  // Sessions: plain select = open beside what's already loaded (focus existing
  // tile/main, else a new tab — main only when it's a blank draft);
  // ⌘/⌃-select / ⌘-Enter = force a new tab; ⇧⌘ = own window. Same door as the
  // sidebar, minus the sidebar's licence to spend main.
  const goSession = useCallback(
    (sessionId: string) => (event?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => {
      openSession(sessionId, navigate, openSessionIntentFromModifiers(event, 'stack'))
    },
    [navigate]
  )

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

  // Running a keepOpen row (a toggle) changes state the rows themselves report,
  // so the groups have to rebuild without the palette closing. A counter keeps
  // that generic — the palette never learns which stores its contributions read.
  const [selectTick, setSelectTick] = useState(0)

  const contributedItems = usePaletteContributions()

  // The active repo's worktrees → "new conversation in <branch>". This is the
  // ⌘K-typed "I want to work on <branch>" reflex: each entry seeds a fresh
  // session anchored to that worktree's checkout (requestStartWorkSession),
  // so git is the source of truth and edits land in the right tree.
  const branchGroup = useMemo<PaletteGroup[]>(
    () =>
      worktrees.length > 0
        ? [
            {
              heading: t.commandCenter.branches,
              items: worktrees.map(wt => {
                const name = wt.branch?.trim() || wt.path.split('/').pop() || wt.path

                return {
                  icon: GitBranch,
                  id: `worktree-${wt.path}`,
                  keywords: ['branch', 'worktree', 'switch', name, wt.path],
                  label: t.commandCenter.startInBranch(name),
                  run: () => requestStartWorkSession(wt.path)
                }
              })
            }
          ]
        : [],
    [t, worktrees]
  )

  const baseGroups = useMemo<PaletteGroup[]>(() => {
    const settingsTab = (tab: string) => `${SETTINGS_ROUTE}?tab=${tab}`
    const cc = t.commandCenter

    // Projects are the primary way the desktop scopes work, so they're jumpable
    // from the palette. Plain select is a pure scope switch (sidebar enters the
    // project — never spends main); ⌘-Enter / ⌘-click also starts a new session
    // at the project root (stacked as a tab when main holds a chat), previewed
    // by the label swap while ⌘ is held. Rows carry the project's own codicon,
    // matching the sidebar. The pinned "Open folder…" row is the ⌘O upsert.
    const projectGroup: PaletteGroup = {
      heading: cc.projects,
      items: [
        {
          action: 'workspace.openFolder',
          icon: codiconIcon('folder-opened'),
          id: 'project-open-folder',
          keywords: ['open', 'folder', 'directory', 'project', 'add', 'import', 'workspace'],
          label: cc.openFolder,
          run: () => void openFolderAsProject()
        },
        ...filterVisibleProjects(projectTree, dismissedAutoProjects).map(project => ({
          comboHint: 'mod+enter',
          icon: codiconIcon(project.icon || (project.isNoProject ? 'home' : 'folder-library')),
          id: `project-${project.id}`,
          keywords: ['project', 'workspace', 'go to', project.label, ...(project.path ? [project.path] : [])],
          label: project.label,
          modLabel: cc.newSessionInProject(project.label),
          runWithEvent: (event?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) =>
            goToProject(project.id, { newSession: Boolean(event?.metaKey || event?.ctrlKey) })
        }))
      ]
    }

    // Group order is the tiebreaker rankGroups falls back on (stable sort), and
    // exact ties are the common case — "yolo" hits both "Toggle yolo" and a
    // worktree named bb/yolo-* as a whole word. So this order IS the priority:
    // where you're going, then what you can do, then what you can configure.
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
          ...(canOpenNewWindow()
            ? [
                {
                  action: 'session.newWindow',
                  icon: AppWindow,
                  id: 'nav-new-window',
                  keywords: ['window', 'instance', 'open', 'new'],
                  label: t.keybinds.actions['session.newWindow'],
                  run: () => void openNewWindow()
                }
              ]
            : []),
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
      projectGroup,
      // Registry-contributed rows (core features + plugins) — one group,
      // omitted while nothing contributes.
      ...(contributedItems.length > 0
        ? [
            {
              heading: cc.commands,
              items: contributedItems.map(item => ({
                action: item.action,
                // Read on mount and after every select (the deps below), so a
                // row that reports state can't show the state it just left.
                detail: item.detail?.(),
                detailVariant: item.detailVariant,
                icon: item.icon ?? Zap,
                id: item.key,
                keepOpen: item.keepOpen,
                keywords: item.keywords,
                label: item.label,
                run: item.run
              }))
            }
          ]
        : []),
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
            detail: updateVersionLabel,
            icon: Download,
            id: 'cc-update-hermes',
            keywords: ['update', 'upgrade', 'hermes', 'version', 'system', 'restart'],
            label: cc.updateHermes,
            run: () => requestActiveUpdate()
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
    // `selectTick` is a deliberate re-read trigger, not a value: rows report
    // live state through `detail()`, so the groups must rebuild after a select
    // that kept the palette open — eslint only sees an unused dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    contributedItems,
    dismissedAutoProjects,
    go,
    projectTree,
    selectTick,
    settingsSectionLabel,
    t,
    updateVersionLabel
  ])

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
            runWithEvent: goSession(directId)
          }
        ]
      })
    }

    // Paste/type an absolute folder path → open it as a project directly (the
    // ⌘O upsert without the native picker). Same reflex as the raw-session-id
    // row above.
    if (FOLDER_PATH_RE.test(directId)) {
      result.push({
        items: [
          {
            icon: codiconIcon('folder-opened'),
            id: `open-folder-${directId}`,
            keywords: ['open', 'folder', 'project', directId],
            label: t.commandCenter.openFolderAt(directId),
            run: () => void openFolderAsProject(directId)
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
        active: themeName === theme.name,
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
        active: mode === entry.mode,
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
          keywords: [
            'chat',
            'session',
            ...(session.preview ? [session.preview] : []),
            ...(session.git_branch ? [session.git_branch] : [])
          ],
          label: session.title,
          runWithEvent: goSession(session.id)
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
          keywords: [
            'archived',
            'chat',
            'session',
            ...(session.preview ? [session.preview] : []),
            ...(session.git_branch ? [session.git_branch] : [])
          ],
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
    goSession,
    mcpServers,
    mode,
    resolvedMode,
    search,
    sessions,
    setMode,
    setTheme,
    settingsSectionLabel,
    t,
    themeName
  ])

  // Branch rows rank below BOTH the fixed groups and the typed-only lists: they
  // scale with whatever worktrees happen to exist, so on a tie they're the least
  // likely thing meant. Everything above is either always-present chrome or a
  // list the search itself asked for.
  const groups = useMemo(
    () => [...baseGroups, ...searchGroups, ...branchGroup],
    [baseGroups, branchGroup, searchGroups]
  )

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
              active: mode === entry.mode,
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
    [availableThemes, mode, resolvedMode, setMode, setTheme, t, themeName]
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

    if (item.runWithEvent) {
      item.runWithEvent(lastSelectMods.current)
    } else {
      item.run?.()
    }

    // Clear stashed modifiers so a plain Enter after a ⌘-click isn't sticky.
    lastSelectMods.current = { ctrlKey: false, metaKey: false, shiftKey: false }

    if (!item.keepOpen) {
      closeCommandPalette()

      return
    }

    // Staying open means the rows are still on screen — re-read anything they
    // report (a toggle's on/off) so the note isn't showing the previous state.
    setSelectTick(tick => tick + 1)
  }

  return (
    <DialogPrimitive.Portal>
      {/* Transparent overlay: keeps click-away + focus trap, but no dim/blur. */}
      <DialogPrimitive.Overlay className="fixed inset-0 z-(--z-over-modal)" />
      <DialogPrimitive.Content
        aria-describedby={undefined}
        className={cn(
          HUD_POSITION,
          HUD_SURFACE,
          'z-(--z-over-modal-content) w-[min(34rem,calc(100vw-2rem))] overflow-hidden duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95'
        )}
        // The close animation finishing is what retires this whole subtree —
        // the CSS owns the duration, not a hardcoded timer. Guarded on the
        // content itself (descendants animate too) and on the closed state, so
        // an OPEN animation never unmounts the palette we just opened.
        onAnimationEnd={event => {
          if (event.target === event.currentTarget && event.currentTarget.dataset.state === 'closed') {
            onExited()
          }
        }}
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
              // Capture modifiers before cmdk's Enter fires onSelect (which
              // swipes the inviting MouseEvent and hands us nothing).
              noteSelectMods(event)

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
              <PaletteGroups
                bindings={bindings}
                groups={visibleGroups}
                modHeld={modHeld}
                noResultsLabel={t.commandCenter.noResults}
                onSelectItem={handleSelect}
                onSelectMods={noteSelectMods}
                search={search}
              />
            )}
          </CommandList>
        </Command>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}
