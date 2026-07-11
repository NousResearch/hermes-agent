export type SlashReferenceCategory = 'Configuration' | 'Exit' | 'Info' | 'Session' | 'Tools & Skills'
export type SlashReferenceSurfaceTag = 'both' | 'cfg' | 'chat' | 'cli'

export interface SlashReferenceCommand {
  aliases?: readonly string[]
  argsHint?: string
  category: SlashReferenceCategory
  cliOnly?: boolean
  description: string
  gatewayConfigGate?: string
  gatewayOnly?: boolean
  name: string
  subcommands?: readonly string[]
}

export interface SlashReferenceSection {
  accent: 'cyan' | 'magenta' | 'orange' | 'yellow'
  commands: readonly SlashReferenceCommand[]
  categories: readonly SlashReferenceCategory[]
  title: string
}

export const slashReferenceCommands = [
  {
    name: 'start',
    description: 'Acknowledge platform start pings without a reply',
    category: 'Session',
    gatewayOnly: true
  },
  {
    name: 'new',
    description: 'Start a new session (fresh session ID + history)',
    category: 'Session',
    aliases: ['reset'],
    argsHint: '[name]'
  },
  {
    name: 'topic',
    description: 'Enable or inspect Telegram DM topic sessions',
    category: 'Session',
    argsHint: '[off|help|session-id]',
    gatewayOnly: true
  },
  { name: 'clear', description: 'Clear screen and start a new session', category: 'Session', cliOnly: true },
  {
    name: 'redraw',
    description: 'Force a full UI repaint (recovers from terminal drift)',
    category: 'Session',
    cliOnly: true
  },
  { name: 'history', description: 'Show conversation history', category: 'Session', cliOnly: true },
  { name: 'save', description: 'Save the current conversation', category: 'Session', cliOnly: true },
  { name: 'retry', description: 'Retry the last message (resend to agent)', category: 'Session' },
  {
    name: 'prompt',
    description: 'Compose your next prompt in $EDITOR (markdown), then send it',
    category: 'Session',
    aliases: ['compose'],
    argsHint: '[initial text]',
    cliOnly: true
  },
  { name: 'undo', description: 'Back up N user turns and re-prompt (default 1)', category: 'Session', argsHint: '[N]' },
  { name: 'title', description: 'Set a title for the current session', category: 'Session', argsHint: '[name]' },
  {
    name: 'handoff',
    description: 'Hand off this session to a messaging platform (Telegram, Discord, etc.)',
    category: 'Session',
    argsHint: '<platform>',
    cliOnly: true
  },
  {
    name: 'branch',
    description: 'Branch the current session (explore a different path)',
    category: 'Session',
    aliases: ['fork'],
    argsHint: '[name]'
  },
  {
    name: 'compress',
    description:
      "Compress conversation context (add 'here [N]' to keep recent N turns; --preview shows what would happen)",
    category: 'Session',
    aliases: ['compact'],
    argsHint: '[here [N] | focus topic | --preview|--dry-run]'
  },
  {
    name: 'rollback',
    description: 'List or restore filesystem checkpoints',
    category: 'Session',
    argsHint: '[number]'
  },
  {
    name: 'snapshot',
    description: 'Create or restore state snapshots of Hermes config/state',
    category: 'Session',
    aliases: ['snap'],
    argsHint: '[create|restore <id>|prune]',
    cliOnly: true
  },
  { name: 'stop', description: 'Kill all running background processes', category: 'Session' },
  {
    name: 'approve',
    description: 'Approve a pending dangerous command',
    category: 'Session',
    argsHint: '[session|always]',
    gatewayOnly: true
  },
  {
    name: 'deny',
    description: 'Deny a pending dangerous command (optionally with a reason)',
    category: 'Session',
    argsHint: '[all] [reason]',
    gatewayOnly: true
  },
  {
    name: 'background',
    description: 'Run a prompt in the background',
    category: 'Session',
    aliases: ['bg', 'btw'],
    argsHint: '<prompt>'
  },
  { name: 'agents', description: 'Show active agents and running tasks', category: 'Session', aliases: ['tasks'] },
  {
    name: 'journey',
    description: 'Open the learning journey timeline',
    category: 'Session',
    aliases: ['learning', 'memory-graph'],
    argsHint: '[list|delete <id>|edit <id>]',
    subcommands: ['list', 'delete', 'edit'],
    cliOnly: true
  },
  {
    name: 'queue',
    description: "Queue a prompt for the next turn (doesn't interrupt)",
    category: 'Session',
    aliases: ['q'],
    argsHint: '<prompt>'
  },
  {
    name: 'steer',
    description: 'Inject a message after the next tool call without interrupting',
    category: 'Session',
    argsHint: '<prompt>'
  },
  {
    name: 'goal',
    description: 'Set a standing goal Hermes works on across turns until achieved',
    category: 'Session',
    argsHint: '[text | draft <text> | show | pause | resume | clear | status | wait <pid> | unwait]'
  },
  {
    name: 'moa',
    description: 'Run one prompt through the default Mixture of Agents preset, then restore your model',
    category: 'Session',
    argsHint: '<prompt>'
  },
  {
    name: 'subgoal',
    description: 'Add or manage extra criteria on the active goal',
    category: 'Session',
    argsHint: '[text | remove N | clear]'
  },
  { name: 'status', description: 'Show session, model, token, and context info', category: 'Session' },
  { name: 'whoami', description: 'Show your slash command access (admin / user)', category: 'Info' },
  { name: 'profile', description: 'Show active profile name and home directory', category: 'Info' },
  {
    name: 'sethome',
    description: 'Set this chat as the home channel',
    category: 'Session',
    aliases: ['set-home'],
    gatewayOnly: true
  },
  { name: 'resume', description: 'Resume a previously-named session', category: 'Session', argsHint: '[name]' },
  { name: 'sessions', description: 'Browse and resume previous sessions', category: 'Session' },
  { name: 'config', description: 'Show current configuration', category: 'Configuration', cliOnly: true },
  {
    name: 'model',
    description: 'Switch model (persists by default)',
    category: 'Configuration',
    argsHint: '[model] [--provider name] [--global|--session] [--refresh]'
  },
  {
    name: 'codex-runtime',
    description: 'Toggle codex app-server runtime for OpenAI/Codex models',
    category: 'Configuration',
    aliases: ['codex_runtime'],
    argsHint: '[auto|codex_app_server]'
  },
  { name: 'personality', description: 'Set a predefined personality', category: 'Configuration', argsHint: '[name]' },
  {
    name: 'statusbar',
    description: 'Toggle the context/model status bar',
    category: 'Configuration',
    aliases: ['sb'],
    cliOnly: true
  },
  {
    name: 'timestamps',
    description: 'Toggle [HH:MM] timestamps on messages and /history',
    category: 'Configuration',
    aliases: ['ts'],
    argsHint: '[on|off|status]',
    subcommands: ['on', 'off', 'status'],
    cliOnly: true
  },
  {
    name: 'verbose',
    description: 'Cycle tool progress display: off -> new -> all -> verbose -> log',
    category: 'Configuration',
    cliOnly: true,
    gatewayConfigGate: 'display.tool_progress_command'
  },
  {
    name: 'footer',
    description: 'Toggle gateway runtime-metadata footer on final replies',
    category: 'Configuration',
    argsHint: '[on|off|status]',
    subcommands: ['on', 'off', 'status']
  },
  { name: 'yolo', description: 'Toggle YOLO mode (skip all dangerous command approvals)', category: 'Configuration' },
  {
    name: 'reasoning',
    description: 'Manage reasoning effort and display',
    category: 'Configuration',
    argsHint: '[level|show|hide|full|clamp]',
    subcommands: [
      'none',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
      'ultra',
      'show',
      'hide',
      'on',
      'off',
      'full',
      'clamp'
    ]
  },
  {
    name: 'fast',
    description: 'Toggle fast mode \u2014 OpenAI Priority Processing / Anthropic Fast Mode (Normal/Fast)',
    category: 'Configuration',
    argsHint: '[normal|fast|status]',
    subcommands: ['normal', 'fast', 'status', 'on', 'off']
  },
  {
    name: 'skin',
    description: 'Show or change the display skin/theme',
    category: 'Configuration',
    argsHint: '[name]',
    cliOnly: true
  },
  {
    name: 'indicator',
    description: 'Pick the TUI busy-indicator style',
    category: 'Configuration',
    argsHint: '[kaomoji|emoji|unicode|ascii]',
    subcommands: ['kaomoji', 'emoji', 'unicode', 'ascii'],
    cliOnly: true
  },
  {
    name: 'voice',
    description: 'Toggle voice mode',
    category: 'Configuration',
    argsHint: '[on|off|tts|status]',
    subcommands: ['on', 'off', 'tts', 'status']
  },
  {
    name: 'busy',
    description: 'Control what Enter does while Hermes is working',
    category: 'Configuration',
    argsHint: '[queue|steer|interrupt|status]',
    subcommands: ['queue', 'steer', 'interrupt', 'status'],
    cliOnly: true
  },
  {
    name: 'tools',
    description: 'Manage tools: /tools [list|disable|enable] [name...]',
    category: 'Tools & Skills',
    argsHint: '[list|disable|enable] [name...]',
    cliOnly: true
  },
  { name: 'toolsets', description: 'List available toolsets', category: 'Tools & Skills', cliOnly: true },
  {
    name: 'skills',
    description: 'Search, install, inspect, or manage skills',
    category: 'Tools & Skills',
    subcommands: [
      'search',
      'browse',
      'inspect',
      'install',
      'audit',
      'pending',
      'approve',
      'reject',
      'diff',
      'approval'
    ],
    cliOnly: true,
    gatewayConfigGate: 'skills.write_approval'
  },
  {
    name: 'memory',
    description: 'Review pending memory writes / toggle the approval gate',
    category: 'Tools & Skills',
    argsHint: '[pending|approve|reject|approval] [id|on|off]',
    subcommands: ['pending', 'approve', 'reject', 'approval']
  },
  {
    name: 'bundles',
    description: 'List skill bundles (aliases /<name> for multiple skills)',
    category: 'Tools & Skills'
  },
  {
    name: 'pet',
    description: 'Toggle or adopt a petdex mascot (/pet, /pet list, /pet <slug>)',
    category: 'Tools & Skills',
    argsHint: '[toggle|list|scale <n>|<slug>]',
    subcommands: ['toggle', 'list', 'scale', 'off'],
    cliOnly: true
  },
  {
    name: 'hatch',
    description: 'Generate a new petdex pet from a description',
    category: 'Tools & Skills',
    aliases: ['generate-pet'],
    argsHint: '[description]',
    cliOnly: true
  },
  {
    name: 'learn',
    description: 'Learn a reusable skill from anything you describe (dirs, URLs, this chat, notes)',
    category: 'Tools & Skills',
    argsHint: '<what to learn from>'
  },
  {
    name: 'cron',
    description: 'Manage scheduled tasks',
    category: 'Tools & Skills',
    argsHint: '[subcommand]',
    subcommands: ['list', 'add', 'create', 'edit', 'pause', 'resume', 'run', 'remove'],
    cliOnly: true
  },
  {
    name: 'suggestions',
    description: 'Review suggested automations (accept/dismiss)',
    category: 'Tools & Skills',
    aliases: ['suggest'],
    argsHint: '[accept|dismiss N | catalog]',
    subcommands: ['accept', 'dismiss', 'catalog', 'clear']
  },
  {
    name: 'blueprint',
    description: 'Set up an automation from a blueprint template',
    category: 'Tools & Skills',
    aliases: ['bp'],
    argsHint: '[name] [slot=value ...]'
  },
  {
    name: 'curator',
    description: 'Background skill maintenance (status, run, pin, archive, list-archived)',
    category: 'Tools & Skills',
    argsHint: '[subcommand]',
    subcommands: ['status', 'run', 'pause', 'resume', 'pin', 'unpin', 'restore', 'list-archived']
  },
  {
    name: 'kanban',
    description: 'Multi-profile collaboration board (tasks, links, comments)',
    category: 'Tools & Skills',
    argsHint: '[subcommand]',
    subcommands: [
      'init',
      'boards',
      'create',
      'list',
      'ls',
      'show',
      'assign',
      'reclaim',
      'reassign',
      'diagnostics',
      'diag',
      'link',
      'unlink',
      'claim',
      'comment',
      'complete',
      'edit',
      'block',
      'unblock',
      'archive',
      'tail',
      'dispatch',
      'stats',
      'notify-subscribe',
      'notify-list',
      'notify-unsubscribe',
      'log',
      'runs',
      'heartbeat',
      'assignees',
      'context',
      'specify',
      'gc'
    ]
  },
  {
    name: 'reload',
    description: 'Reload .env variables into the running session',
    category: 'Tools & Skills',
    cliOnly: true
  },
  {
    name: 'reload-mcp',
    description: 'Reload MCP servers from config',
    category: 'Tools & Skills',
    aliases: ['reload_mcp']
  },
  {
    name: 'reload-skills',
    description: 'Re-scan ~/.hermes/skills/ for newly installed or removed skills',
    category: 'Tools & Skills',
    aliases: ['reload_skills']
  },
  {
    name: 'browser',
    description: 'Connect browser tools to your live Chromium-family browser via CDP',
    category: 'Tools & Skills',
    argsHint: '[connect|disconnect|status]',
    subcommands: ['connect', 'disconnect', 'status'],
    cliOnly: true
  },
  {
    name: 'plugins',
    description: 'List installed plugins and their status',
    category: 'Tools & Skills',
    cliOnly: true
  },
  {
    name: 'commands',
    description: 'Browse all commands and skills (paginated)',
    category: 'Info',
    argsHint: '[page]',
    gatewayOnly: true
  },
  { name: 'help', description: 'Show available commands', category: 'Info' },
  {
    name: 'restart',
    description: 'Gracefully restart the gateway after draining active runs',
    category: 'Session',
    gatewayOnly: true
  },
  {
    name: 'usage',
    description: 'Show token usage and rate limits; `reset` redeems a banked Codex limit reset',
    category: 'Info',
    argsHint: '[reset [--force]]'
  },
  {
    name: 'subscription',
    description: 'View your Nous plan and change it in the browser',
    category: 'Info',
    aliases: ['upgrade'],
    cliOnly: true
  },
  { name: 'topup', description: 'Show your Nous balance and manage billing on the portal', category: 'Info' },
  { name: 'insights', description: 'Show usage insights and analytics', category: 'Info', argsHint: '[days]' },
  {
    name: 'platforms',
    description: 'Show gateway/messaging platform status',
    category: 'Info',
    aliases: ['gateway'],
    cliOnly: true
  },
  {
    name: 'platform',
    description: 'Pause, resume, or list a failing gateway platform',
    category: 'Info',
    argsHint: '<pause|resume|list> [name]',
    gatewayOnly: true
  },
  {
    name: 'copy',
    description: 'Copy the last assistant response to clipboard',
    category: 'Info',
    argsHint: '[number]',
    cliOnly: true
  },
  { name: 'paste', description: 'Attach clipboard image from your clipboard', category: 'Info', cliOnly: true },
  {
    name: 'image',
    description: 'Attach a local image file for your next prompt',
    category: 'Info',
    argsHint: '<path>',
    cliOnly: true
  },
  { name: 'update', description: 'Update Hermes Agent to the latest version', category: 'Info' },
  { name: 'version', description: 'Show Hermes Agent version', category: 'Info', aliases: ['v'] },
  {
    name: 'debug',
    description: 'Upload debug report (system info + logs) and get shareable links',
    category: 'Info',
    argsHint: '[nous|local]'
  },
  {
    name: 'quit',
    description: 'Exit the CLI (use --delete to also remove session history)',
    category: 'Exit',
    aliases: ['exit'],
    argsHint: '[--delete]',
    cliOnly: true
  }
] as const satisfies readonly SlashReferenceCommand[]

const SECTION_DEFS: readonly Omit<SlashReferenceSection, 'commands'>[] = [
  { title: 'Session / Flow', accent: 'cyan', categories: ['Session'] },
  { title: 'Config', accent: 'magenta', categories: ['Configuration'] },
  { title: 'Tools / Skills', accent: 'yellow', categories: ['Tools & Skills'] },
  { title: 'Info / Exit', accent: 'orange', categories: ['Info', 'Exit'] }
]

export const SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE =
  'Installed skills expose /<skill-name> commands. Bundles, plugins, and quick commands can add environment-specific routes.'

export function slashReferenceSurfaceTag(command: SlashReferenceCommand): SlashReferenceSurfaceTag {
  if (command.gatewayConfigGate) {
    return 'cfg'
  }

  if (command.cliOnly) {
    return 'cli'
  }

  if (command.gatewayOnly) {
    return 'chat'
  }

  return 'both'
}

export function slashReferenceCommandToken(command: SlashReferenceCommand): string {
  return `/${command.name}`
}

export function slashReferenceAliasText(command: SlashReferenceCommand, limit = command.aliases?.length ?? 0): string {
  return command.aliases?.length
    ? command.aliases
        .slice(0, Math.max(0, limit))
        .map(alias => `/${alias}`)
        .join(', ')
    : ''
}

const SLASH_REFERENCE_DISPLAY_DESCRIPTIONS: Readonly<Record<string, string>> = {
  start: 'Platform start ping',
  new: 'Start a new session',
  topic: 'Telegram DM topics',
  clear: 'Clear screen + new',
  redraw: 'Force UI repaint',
  history: 'Show transcript',
  save: 'Save conversation',
  retry: 'Resend last message',
  prompt: 'Compose in $EDITOR',
  undo: 'Back up/remove turn',
  title: 'Set session title',
  handoff: 'Move session to chat app',
  branch: 'Fork session',
  compress: 'Summarize context',
  rollback: 'List/restore checkpoints',
  snapshot: 'State snapshots',
  stop: 'Kill/interrupt runs',
  approve: 'Allow pending command',
  deny: 'Reject pending command',
  background: 'Separate background run',
  agents: 'Active agents/tasks',
  journey: 'Learning timeline',
  queue: 'Queue next prompt',
  steer: 'Mid-run nudge',
  goal: 'Persistent goal loop',
  moa: 'One-shot Mixture of Agents',
  subgoal: 'Add goal criteria',
  status: 'Session/model/context',
  sethome: 'Set delivery home',
  resume: 'Restore named session',
  sessions: 'Browse sessions',
  config: 'Show config',
  model: 'Switch model/provider',
  'codex-runtime': 'Codex app-server',
  personality: 'Set personality',
  statusbar: 'Toggle status bar',
  timestamps: 'Message times',
  verbose: 'Tool progress mode',
  footer: 'Metadata footer',
  yolo: 'Skip approvals',
  reasoning: 'Effort/display controls',
  fast: 'Priority/fast mode',
  skin: 'Display theme',
  indicator: 'Busy indicator style',
  voice: 'Voice/TTS mode',
  busy: 'Enter behavior mid-run',
  tools: 'Enable/disable tools',
  toolsets: 'List toolsets',
  skills: 'Manage/approve skills',
  memory: 'Approve memory writes',
  bundles: 'Skill bundle aliases',
  pet: 'Petdex mascot',
  hatch: 'Generate pet',
  learn: 'Create reusable skill',
  cron: 'Scheduled tasks',
  suggestions: 'Automation ideas',
  blueprint: 'Automation template',
  curator: 'Skill maintenance',
  kanban: 'Collaboration board',
  reload: 'Reload .env',
  'reload-mcp': 'Reload MCP',
  'reload-skills': 'Rescan skills',
  browser: 'Attach CDP browser',
  plugins: 'Plugin status',
  whoami: 'Slash access level',
  profile: 'Active profile/home',
  commands: 'Paged command list',
  help: 'Command help',
  restart: 'Restart gateway',
  usage: 'Tokens/cost/limits',
  subscription: 'Nous plan',
  topup: 'Balance and billing',
  insights: 'Usage analytics',
  platforms: 'Gateway status',
  platform: 'Pause/resume adapter',
  copy: 'Copy assistant reply',
  paste: 'Clipboard image',
  image: 'Attach image path',
  update: 'Update Hermes',
  version: 'Version info',
  debug: 'Debug report upload',
  quit: 'Exit CLI'
}

export function slashReferenceDisplayDescription(command: SlashReferenceCommand): string {
  return SLASH_REFERENCE_DISPLAY_DESCRIPTIONS[command.name] ?? command.description
}

export const slashReferenceSections: readonly SlashReferenceSection[] = SECTION_DEFS.map(section => ({
  ...section,
  commands: slashReferenceCommands.filter(command => section.categories.includes(command.category))
}))
