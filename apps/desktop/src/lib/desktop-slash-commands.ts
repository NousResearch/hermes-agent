export interface CommandsCatalogSection {
  name: string
  pairs: [string, string][]
}

export interface CommandsCatalogLike {
  categories?: CommandsCatalogSection[]
  pairs?: [string, string][]
  skill_count?: number
  warning?: string
}

export interface DesktopSlashCompletion {
  display: string
  meta: string
  text: string
}

export interface DesktopThemeCommandOption {
  description: string
  label: string
  name: string
}

const DESKTOP_COMMAND_META = [
  ['/agents', 'Show active desktop sessions and running tasks'],
  ['/background', 'Run a prompt in the background'],
  ['/branch', 'Branch the latest message into a new chat'],
  ['/compress', 'Compress this conversation context'],
  ['/debug', 'Create a debug report'],
  ['/goal', 'Manage the standing goal for this session'],
  ['/help', 'Show desktop slash commands'],
  ['/new', 'Start a new desktop chat'],
  ['/queue', 'Queue a prompt for the next turn'],
  ['/resume', 'Resume a saved session'],
  ['/retry', 'Retry the last user message'],
  ['/rollback', 'List or restore filesystem checkpoints'],
  ['/skin', 'Switch desktop theme or cycle to the next one'],
  ['/status', 'Show current session status'],
  ['/steer', 'Steer the current run after the next tool call'],
  ['/stop', 'Stop running background processes'],
  ['/title', 'Rename the current session'],
  ['/undo', 'Remove the last user/assistant exchange'],
  ['/usage', 'Show token usage for this session']
] as const

const DESKTOP_COMMANDS: ReadonlySet<string> = new Set(DESKTOP_COMMAND_META.map(([command]) => command))

const DESKTOP_ALIASES = new Map([
  ['/bg', '/background'],
  ['/btw', '/background'],
  ['/fork', '/branch'],
  ['/q', '/queue'],
  ['/reload_mcp', '/reload-mcp'],
  ['/reload_skills', '/reload-skills'],
  ['/reset', '/new'],
  ['/tasks', '/agents']
])

const DESKTOP_COMMAND_DESCRIPTIONS: ReadonlyMap<string, string> = new Map(DESKTOP_COMMAND_META)

const PICKER_OWNED_COMMANDS = new Set(['/model', '/provider'])

const TERMINAL_ONLY_COMMANDS = new Set([
  '/browser',
  '/busy',
  '/clear',
  '/commands',
  '/compact',
  '/config',
  '/copy',
  '/cron',
  '/details',
  '/exit',
  '/footer',
  '/gateway',
  '/gquota',
  '/history',
  '/image',
  '/indicator',
  '/logs',
  '/mouse',
  '/paste',
  '/platforms',
  '/plugins',
  '/quit',
  '/redraw',
  '/reload',
  '/restart',
  '/save',
  '/sb',
  '/set-home',
  '/sethome',
  '/snap',
  '/snapshot',
  '/statusbar',
  '/toolsets',
  '/tools',
  '/update',
  '/verbose'
])

const MESSAGING_ONLY_COMMANDS = new Set(['/approve', '/deny'])

const SETTINGS_OWNED_COMMANDS = new Set(['/skills'])

const ADVANCED_COMMANDS = new Set([
  '/curator',
  '/fast',
  '/insights',
  '/kanban',
  '/personality',
  '/profile',
  '/reasoning',
  '/reload-mcp',
  '/reload-skills',
  '/voice',
  '/yolo'
])

const BLOCKED_COMMANDS = new Set([
  ...PICKER_OWNED_COMMANDS,
  ...TERMINAL_ONLY_COMMANDS,
  ...MESSAGING_ONLY_COMMANDS,
  ...SETTINGS_OWNED_COMMANDS,
  ...ADVANCED_COMMANDS
])

function normalizeCommand(command: string): string {
  const trimmed = command.trim()
  const base = (trimmed.startsWith('/') ? trimmed : `/${trimmed}`).split(/\s+/, 1)[0]?.toLowerCase() || ''

  return base
}

export function canonicalDesktopSlashCommand(command: string): string {
  const normalized = normalizeCommand(command)

  return DESKTOP_ALIASES.get(normalized) || normalized
}

export function isDesktopSlashCommand(command: string): boolean {
  const normalized = normalizeCommand(command)
  const canonical = canonicalDesktopSlashCommand(normalized)

  if (BLOCKED_COMMANDS.has(normalized) || BLOCKED_COMMANDS.has(canonical)) {
    return false
  }

  return DESKTOP_COMMANDS.has(canonical) || !isKnownHermesSlashCommand(normalized)
}

export function isDesktopSlashSuggestion(command: string): boolean {
  const normalized = normalizeCommand(command)
  const canonical = canonicalDesktopSlashCommand(normalized)

  return DESKTOP_COMMANDS.has(canonical) && !DESKTOP_ALIASES.has(normalized)
}

export function desktopSlashUnavailableMessage(command: string): string | null {
  const normalized = normalizeCommand(command)
  const canonical = canonicalDesktopSlashCommand(normalized)

  if (PICKER_OWNED_COMMANDS.has(canonical)) {
    return `/${canonical.slice(1)} uses the desktop model picker instead of a slash command.`
  }

  if (SETTINGS_OWNED_COMMANDS.has(canonical)) {
    return `/${canonical.slice(1)} is managed from the desktop sidebar.`
  }

  if (MESSAGING_ONLY_COMMANDS.has(canonical)) {
    return `/${canonical.slice(1)} is only used from messaging platforms.`
  }

  if (ADVANCED_COMMANDS.has(canonical)) {
    return `/${canonical.slice(1)} is not shown in the desktop slash palette. Use the relevant desktop control or terminal interface instead.`
  }

  if (TERMINAL_ONLY_COMMANDS.has(normalized) || TERMINAL_ONLY_COMMANDS.has(canonical)) {
    return `/${canonical.slice(1)} is only available in the terminal interface.`
  }

  return null
}

export function desktopSlashDescription(command: string, fallback = ''): string {
  const canonical = canonicalDesktopSlashCommand(command)

  return DESKTOP_COMMAND_DESCRIPTIONS.get(canonical) || fallback
}

export function desktopSkinSlashCompletions(
  themes: DesktopThemeCommandOption[],
  activeThemeName: string,
  argPrefix: string
): DesktopSlashCompletion[] {
  const prefix = argPrefix.trim().toLowerCase()

  const commands: DesktopSlashCompletion[] = [
    {
      text: '/skin list',
      display: '/skin list',
      meta: 'Show available desktop themes'
    },
    {
      text: '/skin next',
      display: '/skin next',
      meta: 'Cycle to the next desktop theme'
    },
    ...themes.map(theme => ({
      text: `/skin ${theme.name}`,
      display: `/skin ${theme.name}`,
      meta: `${theme.label}${theme.name === activeThemeName ? ' (current)' : ''} - ${theme.description}`
    }))
  ]

  if (!prefix) {
    return commands
  }

  return commands.filter(item => item.text.slice('/skin '.length).toLowerCase().startsWith(prefix))
}

export function filterDesktopCommandsCatalog(catalog: CommandsCatalogLike): CommandsCatalogLike {
  const categories = catalog.categories
    ?.map(section => ({
      ...section,
      pairs: section.pairs
        .filter(([command]) => isDesktopSlashSuggestion(command))
        .map(([command, description]) => [command, desktopSlashDescription(command, description)] as [string, string])
    }))
    .filter(section => section.pairs.length > 0)

  const pairs = catalog.pairs
    ?.filter(([command]) => isDesktopSlashSuggestion(command))
    .map(([command, description]) => [command, desktopSlashDescription(command, description)] as [string, string])

  return {
    ...catalog,
    ...(categories ? { categories } : {}),
    ...(pairs ? { pairs } : {})
  }
}

function isKnownHermesSlashCommand(command: string): boolean {
  return DESKTOP_COMMANDS.has(command) || DESKTOP_ALIASES.has(command) || BLOCKED_COMMANDS.has(command)
}
