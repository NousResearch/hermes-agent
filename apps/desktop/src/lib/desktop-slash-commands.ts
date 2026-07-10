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

/**
 * Local client action a command resolves to. Each id maps to exactly one
 * handler in the dispatcher (`use-prompt-actions`), so adding a command never
 * means adding a branch to a switch ladder — you add a row here + a handler
 * keyed by the id.
 */
export type DesktopActionId =
  | 'branch'
  | 'browser'
  | 'handoff'
  | 'hatch'
  | 'help'
  | 'journey'
  | 'new'
  | 'pet'
  | 'profile'
  | 'skin'
  | 'title'
  | 'yolo'

/** A command fulfilled by opening a desktop overlay picker. */
export type DesktopPickerId = 'model' | 'session'

/** Why a known Hermes command has no desktop UI surface. */
export type DesktopUnavailableReason = 'advanced' | 'messaging' | 'settings' | 'terminal'

/**
 * How the desktop fulfils a command. This is the single discriminator the
 * dispatcher, popover, pills, and pickers all read — no parallel block-lists.
 *
 * - `action`     → handled by a local client handler (new chat, branch, …)
 * - `picker`     → opens an overlay (`/model`, `/resume`); a typed arg is
 *                  resolved by that picker instead of falling through
 * - `rpc`        → dedicated gateway RPC named on the surface. The dispatcher
 *                  calls it directly with the params built by `buildParams`,
 *                  bypassing `slash.exec` / `command.dispatch`. Reserved for
 *                  commands that have a first-class RPC handler in
 *                  `tui_gateway/server.py` (e.g. `/compress` → session.compress).
 * - `exec`       → runs on the backend via slash.exec / command.dispatch and
 *                  renders its text output inline. Only commands WITHOUT a
 *                  dedicated RPC should stay here (skills, quick commands,
 *                  plugins, and legacy commands handled inline by
 *                  command.dispatch).
 * - `unavailable`→ a known command with genuinely no desktop UI (terminal-only,
 *                  messaging-only, …); shows a reason instead of executing
 */
export type DesktopCommandSurface =
  | { kind: 'action'; action: DesktopActionId }
  | { kind: 'picker'; picker: DesktopPickerId }
  | { kind: 'rpc'; rpc: string; timeoutMs?: number; buildParams: (ctx: SlashCommandBuildCtx) => Record<string, unknown> }
  | { kind: 'exec' }
  | { kind: 'unavailable'; reason: DesktopUnavailableReason }

/**
 * Inputs a `buildParams` function receives. The dispatcher passes session id,
 * the typed arg, and the canonical command name so handlers can construct
 * the exact JSON the gateway method expects.
 */
export interface SlashCommandBuildCtx {
  arg: string
  command: string
  name: string
  sessionId: string
}

export interface DesktopCommandSpec {
  /** Canonical command, leading slash included (e.g. `/resume`). */
  name: string
  /** Popover/help label; omitted for unavailable commands (never surfaced). */
  description?: string
  aliases?: string[]
  surface: DesktopCommandSurface
  /**
   * Hide from the slash popover / completions while still letting it execute.
   * Used for picker commands reachable from chrome (the model picker lives on
   * the status bar), so the popover doesn't dead-end on inline completion.
   */
  hidden?: boolean
  /**
   * The command has an inline options "screen" (theme / personality / session /
   * platform / toolset list). Picking the bare command in the popover expands to
   * that argument step instead of committing — mirroring typing `/<cmd> ` by hand.
   */
  args?: boolean
}

const exec = (): DesktopCommandSurface => ({ kind: 'exec' })
const action = (id: DesktopActionId): DesktopCommandSurface => ({ kind: 'action', action: id })
const picker = (id: DesktopPickerId): DesktopCommandSurface => ({ kind: 'picker', picker: id })
const unavailable = (reason: DesktopUnavailableReason): DesktopCommandSurface => ({ kind: 'unavailable', reason })

/**
 * Route a command directly to its dedicated gateway RPC. Prefer this over
 * `exec()` whenever `tui_gateway/server.py` exposes a `@method(...)` for the
 * command — bypassing `slash.exec` keeps the path short and the response
 * structured.
 *
 * The dispatcher calls `requestGateway(surface.rpc, surface.buildParams(ctx))`
 * and then runs `renderRpcResult` to format the response.
 */
const rpc = (
  rpcName: string,
  buildParams: (ctx: SlashCommandBuildCtx) => Record<string, unknown>,
  timeoutMs?: number
): DesktopCommandSurface => ({ kind: 'rpc', rpc: rpcName, timeoutMs, buildParams })

/**
 * THE source of truth for desktop slash commands. Everything below — execution
 * gating, popover suggestions, catalog filtering, pill grouping, and the
 * dispatcher's behavior — derives from this one table.
 */
const DESKTOP_COMMAND_SPECS: readonly DesktopCommandSpec[] = [
  // Local client actions
  { name: '/new', description: 'Start a new desktop chat', aliases: ['/reset'], surface: action('new') },
  {
    name: '/branch',
    description: 'Branch the latest message into a new chat',
    aliases: ['/fork'],
    surface: action('branch')
  },
  { name: '/yolo', description: 'Toggle YOLO — auto-approve dangerous commands', surface: action('yolo') },
  {
    name: '/handoff',
    description: 'Hand off this session to a messaging platform',
    surface: action('handoff'),
    args: true
  },
  { name: '/profile', description: 'Switch the active Hermes profile', surface: action('profile') },
  { name: '/skin', description: 'Switch desktop theme or cycle to the next one', surface: action('skin'), args: true },
  { name: '/title', description: 'Rename the current session', surface: action('title') },
  { name: '/help', description: 'Show desktop slash commands', aliases: ['/commands'], surface: action('help') },
  {
    name: '/browser',
    description: 'Manage browser CDP connection [connect|disconnect|status] (local gateway only)',
    surface: action('browser'),
    args: true
  },
  {
    name: '/journey',
    description: 'Open the memory graph — skills + memories over time',
    aliases: ['/learning', '/memory-graph'],
    surface: action('journey')
  },

  // Overlay pickers
  { name: '/model', description: 'Switch the model for this session', surface: picker('model'), hidden: true },
  {
    name: '/resume',
    description: 'Resume a saved session',
    aliases: ['/sessions', '/switch'],
    surface: picker('session'),
    args: true
  },

  // Backend-executed commands that render useful inline output
  {
    name: '/agents',
    description: 'Show active desktop sessions and running tasks',
    aliases: ['/tasks'],
    surface: rpc('agents.list', ctx => ({ session_id: ctx.sessionId }))
  },
  { name: '/background', description: 'Run a prompt in the background', aliases: ['/bg', '/btw'], surface: exec() },
  // /compress has a dedicated `session.compress` RPC on the gateway. Routing
  // it through `exec()` (slash.exec → command.dispatch) used to fall through
  // to `not a quick/plugin/skill command: compress` because command.dispatch
  // has no inline branch for `compress`. Bypass that path entirely.
  //
  // The RPC itself runs `_compress_context` which makes an LLM summarise call
  // while holding `session["history_lock"]`. That can easily exceed the
  // 30s default gateway RPC timeout — give it 3min to cover slow models and
  // very long sessions without surfacing a false "request timed out".
  { name: '/compress', description: 'Compress this conversation context', surface: rpc('session.compress', ctx => ({ session_id: ctx.sessionId, focus_topic: ctx.arg.trim() }), 180_000) },
  { name: '/debug', description: 'Create a debug report', surface: exec() },
  { name: '/goal', description: 'Manage the standing goal for this session', surface: exec() },
  { name: '/personality', description: 'Switch personality for this session', surface: exec(), args: true },
  {
    name: '/pet',
    surface: action('pet'),
    args: true
  },
  {
    name: '/hatch',
    surface: action('hatch')
  },
  { name: '/queue', description: 'Queue a prompt for the next turn', aliases: ['/q'], surface: exec() },
  { name: '/retry', description: 'Retry the last user message', surface: exec() },
  { name: '/rollback', description: 'List or restore filesystem checkpoints', surface: exec() },
  { name: '/save', description: 'Save the current transcript to JSON', surface: rpc('session.save', ctx => ({ session_id: ctx.sessionId })) },
  { name: '/status', description: 'Show current session status', surface: rpc('session.status', ctx => ({ session_id: ctx.sessionId })) },
  { name: '/steer', description: 'Steer the current run after the next tool call', surface: rpc('session.steer', ctx => ({ session_id: ctx.sessionId, text: ctx.arg.trim() })) },
  { name: '/stop', description: 'Stop running background processes', surface: rpc('process.stop', () => ({})) },
  { name: '/tools', description: 'List or toggle tools available to the agent', surface: exec(), args: true },
  { name: '/undo', description: 'Remove the last user/assistant exchange', surface: exec() },
  { name: '/usage', description: 'Show token usage for this session', surface: rpc('session.usage', ctx => ({ session_id: ctx.sessionId })) },
  { name: '/version', description: 'Show Hermes Agent version', surface: exec() },

  // No desktop surface, but carry an alias (underscore spelling variants).
  { name: '/reload-mcp', aliases: ['/reload_mcp'], surface: unavailable('advanced') },
  { name: '/reload-skills', aliases: ['/reload_skills'], surface: unavailable('advanced') }
]

// Known commands with no desktop surface (and no alias) — a flat name list
// per reason beats 40 identical object literals.
const NO_DESKTOP_SURFACE: Record<DesktopUnavailableReason, readonly string[]> = {
  terminal: [
    '/busy',
    '/clear',
    '/compact',
    '/config',
    '/copy',
    '/cron',
    '/details',
    '/exit',
    '/footer',
    '/gateway',
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
    '/sb',
    '/set-home',
    '/sethome',
    '/snap',
    '/snapshot',
    '/statusbar',
    '/toolsets',
    '/update',
    '/verbose'
  ],
  messaging: ['/approve', '/deny'],
  settings: ['/skills', '/pets'],
  advanced: ['/curator', '/fast', '/insights', '/kanban', '/reasoning', '/voice']
}

const ALL_SPECS: readonly DesktopCommandSpec[] = [
  ...DESKTOP_COMMAND_SPECS,
  ...(Object.entries(NO_DESKTOP_SURFACE) as [DesktopUnavailableReason, readonly string[]][]).flatMap(
    ([reason, names]) => names.map(name => ({ name, surface: unavailable(reason) }))
  )
]

const SPEC_BY_NAME = new Map<string, DesktopCommandSpec>(ALL_SPECS.map(spec => [spec.name, spec]))

const ALIAS_TO_CANONICAL = new Map<string, string>(
  ALL_SPECS.flatMap(spec => (spec.aliases ?? []).map(alias => [alias, spec.name] as const))
)

const UNAVAILABLE_MESSAGE: Record<DesktopUnavailableReason, (command: string) => string> = {
  advanced: command =>
    `${command} is not shown in the desktop slash palette. Use the relevant desktop control or terminal interface instead.`,
  messaging: command => `${command} is only used from messaging platforms.`,
  settings: command => `${command} is managed from the desktop sidebar.`,
  terminal: command => `${command} is only available in the terminal interface.`
}

const PICKER_UNAVAILABLE_MESSAGE: Record<DesktopPickerId, (command: string) => string> = {
  model: command => `${command} uses the desktop model picker instead of a slash command.`,
  session: command => `${command} uses the desktop session picker instead of a slash command.`
}

function normalizeCommand(command: string): string {
  const trimmed = command.trim()
  const base = (trimmed.startsWith('/') ? trimmed : `/${trimmed}`).split(/\s+/, 1)[0]?.toLowerCase() || ''

  return base
}

export function canonicalDesktopSlashCommand(command: string): string {
  const normalized = normalizeCommand(command)

  return ALIAS_TO_CANONICAL.get(normalized) || normalized
}

/** Resolve a command (or alias) to its desktop spec, or null for unknown/extension commands. */
export function resolveDesktopCommand(command: string): DesktopCommandSpec | null {
  return SPEC_BY_NAME.get(canonicalDesktopSlashCommand(command)) ?? null
}

function isKnownHermesSlashCommand(command: string): boolean {
  const normalized = normalizeCommand(command)

  return SPEC_BY_NAME.has(normalized) || ALIAS_TO_CANONICAL.has(normalized)
}

/**
 * An "extension" command is anything the backend surfaces that is NOT one of
 * Hermes' built-in slash commands — i.e. skill commands (`/gif-search`,
 * `/codex`, …) and user-defined quick commands. These are user-activated, so
 * they appear in the desktop slash palette and execute when typed.
 */
export function isDesktopSlashExtensionCommand(command: string): boolean {
  const normalized = normalizeCommand(command)

  if (!normalized || normalized === '/') {
    return false
  }

  return !isKnownHermesSlashCommand(normalized)
}

/** Gates execution: true unless the command is a known no-desktop-surface command. */
export function isDesktopSlashCommand(command: string): boolean {
  const spec = resolveDesktopCommand(command)

  if (spec) {
    return spec.surface.kind !== 'unavailable'
  }

  return isDesktopSlashExtensionCommand(command)
}

/** Gates discovery in the popover/completions. */
export function isDesktopSlashSuggestion(command: string): boolean {
  const normalized = normalizeCommand(command)

  // Aliases stay hidden so the popover isn't cluttered with duplicates.
  if (ALIAS_TO_CANONICAL.has(normalized)) {
    return false
  }

  const spec = SPEC_BY_NAME.get(normalized)

  if (spec) {
    return spec.surface.kind !== 'unavailable' && !spec.hidden
  }

  // Skill / quick commands the backend provides.
  return isDesktopSlashExtensionCommand(normalized)
}

/**
 * True for commands the desktop fulfils by opening an overlay picker
 * (`/model`, `/resume`/`/sessions`/`/switch`). Optionally pin to one picker.
 */
export function isPickerCommand(command: string, picker?: DesktopPickerId): boolean {
  const surface = resolveDesktopCommand(command)?.surface

  if (surface?.kind !== 'picker') {
    return false
  }

  return picker ? surface.picker === picker : true
}

/** Back-compat shim for the model picker check. */
export function isModelPickerCommand(command: string): boolean {
  return isPickerCommand(command, 'model')
}

export function desktopSlashUnavailableMessage(command: string): string | null {
  const canonical = canonicalDesktopSlashCommand(command)
  const surface = SPEC_BY_NAME.get(canonical)?.surface

  if (!surface) {
    return null
  }

  if (surface.kind === 'unavailable') {
    return UNAVAILABLE_MESSAGE[surface.reason](canonical)
  }

  if (surface.kind === 'picker') {
    return PICKER_UNAVAILABLE_MESSAGE[surface.picker](canonical)
  }

  return null
}

export function desktopSlashDescription(command: string, fallback = ''): string {
  return SPEC_BY_NAME.get(canonicalDesktopSlashCommand(command))?.description || fallback
}

/**
 * True when picking the bare command should expand to its inline argument
 * options (theme / personality / session / platform / toolset) rather than
 * committing immediately. Lets the popover act as a two-step picker.
 */
export function desktopSlashCommandTakesArgs(command: string): boolean {
  return resolveDesktopCommand(command)?.args ?? false
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

  // Recount skill commands from the filtered output so /help's footer reflects
  // what the user actually sees. Backend's skill_count includes commands the
  // desktop hides (terminal-only, picker-owned, advanced), producing a footer
  // like "60 skill commands available" while only ~29 appear in the list.
  const filteredCommands = new Set<string>()

  for (const section of categories ?? []) {
    for (const [command] of section.pairs) {
      filteredCommands.add(canonicalDesktopSlashCommand(command))
    }
  }

  for (const [command] of pairs ?? []) {
    filteredCommands.add(canonicalDesktopSlashCommand(command))
  }

  let skillCount = 0

  for (const command of filteredCommands) {
    if (isDesktopSlashExtensionCommand(command)) {
      skillCount += 1
    }
  }

  const hasSkillCount = catalog.skill_count !== undefined || skillCount > 0

  return {
    ...catalog,
    ...(categories ? { categories } : {}),
    ...(pairs ? { pairs } : {}),
    ...(hasSkillCount ? { skill_count: skillCount } : {})
  }
}
