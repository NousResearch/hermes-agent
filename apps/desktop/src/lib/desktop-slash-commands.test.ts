import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  type CommandCatalogMeta,
  type CommandsCatalogLike,
  desktopSkinSlashCompletions,
  type DesktopSlashArgumentMode,
  desktopSlashCommandArgumentMode,
  desktopSlashDescription,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  isDesktopSlashSuggestion,
  isModelPickerCommand,
  isPickerCommand,
  rankSkillCommands,
  rememberDesktopCommandsCatalog,
  resolveDesktopCommand,
  slashCompletionGroup
} from './desktop-slash-commands'

function registryCatalog(
  modes: Record<string, DesktopSlashArgumentMode | null>,
  aliases: Record<string, string> = {}
): CommandsCatalogLike {
  const commands: Record<string, CommandCatalogMeta> = {}
  const canon: Record<string, string> = {}

  for (const [name, argument_mode] of Object.entries(modes)) {
    commands[name] = { argument_mode, desktop: null }
    canon[name] = name
  }

  for (const [alias, target] of Object.entries(aliases)) {
    commands[alias] = commands[target]
    canon[alias] = target
  }

  return { commands, canon }
}

const REGISTRY_CATALOG = registryCatalog(
  {
    '/approvals': 'options',
    '/review': 'text',
    '/refine': 'text',
    '/usage': null,
    '/version': null,
    '/agents': null,
    '/steer': 'text',
    '/stop': null,
    '/bg': 'text',
    '/btw': 'text',
    '/debug': null,
    '/goal': 'mixed',
    '/personality': 'options',
    '/queue': 'text',
    '/retry': null,
    '/rollback': null,
    '/tools': 'options',
    '/undo': null,
    '/loop': 'mixed',
    '/lcm': 'text'
  },
  { '/tasks': '/agents', '/background': '/bg', '/q': '/queue', '/proactive': '/loop' }
)

describe('desktop slash command curation', () => {
  beforeEach(() => {
    rememberDesktopCommandsCatalog(REGISTRY_CATALOG)
  })

  afterEach(() => {
    rememberDesktopCommandsCatalog(undefined)
  })

  it('keeps core desktop chat commands in suggestions', () => {
    expect(isDesktopSlashSuggestion('/new')).toBe(true)
    expect(isDesktopSlashSuggestion('/branch')).toBe(true)
    expect(isDesktopSlashSuggestion('/skin')).toBe(true)
    expect(isDesktopSlashSuggestion('/usage')).toBe(true)
    expect(isDesktopSlashSuggestion('/version')).toBe(true)
    expect(isDesktopSlashSuggestion('/yolo')).toBe(true)
    expect(isDesktopSlashCommand('/yolo')).toBe(true)
    expect(isDesktopSlashSuggestion('/approvals')).toBe(true)
    expect(isDesktopSlashCommand('/approvals')).toBe(true)
    expect(resolveDesktopCommand('/approvals')?.surface).toEqual({ kind: 'exec' })
    expect(isDesktopSlashSuggestion('/review')).toBe(true)
    expect(isDesktopSlashCommand('/review')).toBe(true)
    expect(resolveDesktopCommand('/review')?.surface).toEqual({ kind: 'exec' })
    expect(resolveDesktopCommand('/review')?.argumentMode).toBe('text')
  })

  it('treats registry and plugin commands as exec when the catalog says so', () => {
    expect(resolveDesktopCommand('/refine')?.argumentMode).toBe('text')
    expect(isDesktopSlashSuggestion('/refine')).toBe(true)
    expect(isDesktopSlashSuggestion('/background')).toBe(false)
    expect(isDesktopSlashCommand('/bg')).toBe(true)
    expect(desktopSlashCommandArgumentMode('/bg')).toBe('text')
    expect(isDesktopSlashCommand('/btw')).toBe(true)
    expect(desktopSlashCommandArgumentMode('/btw')).toBe('text')
    expect(resolveDesktopCommand('/lcm')?.surface).toEqual({ kind: 'exec' })
    expect(desktopSlashCommandArgumentMode('/lcm')).toBe('text')
  })

  it('groups complete.slash rows by backend kind, not the desktop table', () => {
    // A registry command the table has never heard of is still a command.
    expect(slashCompletionGroup('/refine', 'command')).toBe('Commands')
    expect(slashCompletionGroup('/docx', 'skill')).toBe('Skills')
    // Older backends omit kind — fall back to the table.
    expect(slashCompletionGroup('/new')).toBe('Commands')
    expect(slashCompletionGroup('/docx')).toBe('Skills')
  })

  it('surfaces skill and quick commands (extensions) in suggestions and lets them run', () => {
    expect(isDesktopSlashSuggestion('/my-skill')).toBe(true)
    expect(isDesktopSlashSuggestion('/gif-search')).toBe(true)
    expect(isDesktopSlashCommand('/my-skill')).toBe(true)
  })

  it('hides terminal, messaging, and dedicated-UI commands from suggestions', () => {
    expect(isDesktopSlashSuggestion('/clear')).toBe(false)
    expect(isDesktopSlashSuggestion('/density')).toBe(false)
    expect(isDesktopSlashSuggestion('/redraw')).toBe(false)
    expect(isDesktopSlashSuggestion('/approve')).toBe(false)
    expect(isDesktopSlashSuggestion('/model')).toBe(false)
    expect(isDesktopSlashSuggestion('/skills')).toBe(false)
    expect(isDesktopSlashSuggestion('/voice')).toBe(false)
    expect(isDesktopSlashSuggestion('/curator')).toBe(false)
  })

  it('/voice points at the composer voice button instead of the generic advanced message', () => {
    // /voice arms server-side capture — on the desktop the composer's own
    // voice conversation (mic menu / Ctrl+B) is the surface. A user typing
    // /voice must be told where the button IS, not shrugged at.
    expect(resolveDesktopCommand('/voice')?.surface).toEqual({ kind: 'unavailable', reason: 'composer-voice' })
    expect(isDesktopSlashCommand('/voice')).toBe(false)

    const message = desktopSlashUnavailableMessage('/voice')
    expect(message).toContain('microphone button')
    expect(message).toContain('Ctrl+B')
  })

  it('routes /compact to /compress (context compression), not the TUI display toggle', () => {
    expect(resolveDesktopCommand('/compact')?.name).toBe('/compress')
    expect(isDesktopSlashCommand('/compact')).toBe(true)
    // Alias stays out of the popover so /compress is the single visible entry.
    expect(isDesktopSlashSuggestion('/compact')).toBe(false)
    expect(isDesktopSlashSuggestion('/compress')).toBe(true)
  })

  it('surfaces /tools, /save, and /personality on the desktop', () => {
    expect(isDesktopSlashSuggestion('/tools')).toBe(true)
    expect(isDesktopSlashSuggestion('/save')).toBe(true)
    expect(isDesktopSlashSuggestion('/personality')).toBe(true)
    expect(isDesktopSlashCommand('/tools')).toBe(true)
    expect(isDesktopSlashCommand('/save')).toBe(true)
    expect(isDesktopSlashCommand('/personality')).toBe(true)
    expect(desktopSlashUnavailableMessage('/tools')).toBeNull()
    expect(desktopSlashUnavailableMessage('/save')).toBeNull()
    expect(desktopSlashUnavailableMessage('/personality')).toBeNull()
  })

  it('routes /pet through the desktop action handler and drops /pets', () => {
    expect(resolveDesktopCommand('/pet')?.surface).toEqual({ kind: 'action', action: 'pet' })
    expect(desktopSlashCommandArgumentMode('/pet')).toBe('options')
    expect(isDesktopSlashSuggestion('/pet')).toBe(true)
    expect(isDesktopSlashCommand('/pet')).toBe(true)
    expect(resolveDesktopCommand('/pets')?.surface).toEqual({ kind: 'unavailable', reason: 'settings' })
    expect(isDesktopSlashSuggestion('/pets')).toBe(false)
    expect(isDesktopSlashCommand('/pets')).toBe(false)
  })

  it('routes /wake through the desktop wake action instead of the slash worker', () => {
    expect(resolveDesktopCommand('/wake')?.surface).toEqual({ kind: 'action', action: 'wake' })
    expect(desktopSlashCommandArgumentMode('/wake')).toBe('options')
    expect(isDesktopSlashSuggestion('/wake')).toBe(true)
    expect(isDesktopSlashCommand('/wake')).toBe(true)
    expect(desktopSlashUnavailableMessage('/wake')).toBeNull()
  })

  it('treats /browser as an executable action command (local-gateway connect)', () => {
    // /browser used to be terminal-only; it now resolves to a desktop action
    // handler that routes browser.manage RPC when the gateway is local.
    expect(isDesktopSlashCommand('/browser')).toBe(true)
    expect(isDesktopSlashSuggestion('/browser')).toBe(true)
    expect(desktopSlashUnavailableMessage('/browser')).toBeNull()
    expect(resolveDesktopCommand('/browser')?.surface).toEqual({ kind: 'action', action: 'browser' })
    // Bare /browser expands to its sub-action options in the popover.
    expect(desktopSlashCommandArgumentMode('/browser')).toBe('options')
  })

  it('routes /compress through the session-compression action', () => {
    // /compress must be an action (session.compress RPC), not exec: the slash
    // worker route times out on large sessions (#44456).
    expect(resolveDesktopCommand('/compress')?.surface).toEqual({ kind: 'action', action: 'compress' })
    expect(desktopSlashCommandArgumentMode('/compress')).toBe('text')
    expect(isDesktopSlashCommand('/compress')).toBe(true)
    expect(isDesktopSlashSuggestion('/compress')).toBe(true)
    expect(desktopSlashUnavailableMessage('/compress')).toBeNull()
    // /compact is an alias — executes but stays out of the popover.
    expect(resolveDesktopCommand('/compact')?.surface).toEqual({ kind: 'action', action: 'compress' })
    expect(isDesktopSlashCommand('/compact')).toBe(true)
    expect(isDesktopSlashSuggestion('/compact')).toBe(false)
  })

  it('routes only stateless session commands through dedicated gateway RPCs', () => {
    const expected = {
      '/save': 'session.save',
      '/status': 'session.status'
    } as const

    for (const [name, rpcName] of Object.entries(expected)) {
      const surface = resolveDesktopCommand(name)?.surface
      expect(surface?.kind).toBe('rpc')

      if (surface?.kind !== 'rpc') {
        continue
      }

      expect(surface.rpc).toBe(rpcName)
      expect(surface.buildParams({ arg: 'topic A', command: name, name: name.slice(1), sessionId: 's-1' })).toEqual({
        session_id: 's-1'
      })
    }
  })

  it('keeps commands with richer CLI semantics on the slash worker', () => {
    for (const name of ['/agents', '/steer', '/stop', '/usage']) {
      expect(resolveDesktopCommand(name)?.surface).toEqual({ kind: 'exec' })
    }
  })

  it('still routes commands without dedicated RPCs through exec()', () => {
    // /btw is an action (prompt.btw) — the slash-worker print never reached Desktop.
    const execNames = [
      '/bg',
      '/debug',
      '/goal',
      '/personality',
      '/queue',
      '/retry',
      '/rollback',
      '/tools',
      '/undo',
      '/version'
    ]

    for (const name of execNames) {
      expect(resolveDesktopCommand(name)?.surface).toEqual({ kind: 'exec' })
    }
  })

  it('routes /btw to the prompt.btw side-question action', () => {
    expect(resolveDesktopCommand('/btw')?.surface).toEqual({ kind: 'action', action: 'btw' })
    expect(isDesktopSlashCommand('/btw')).toBe(true)
    expect(isDesktopSlashSuggestion('/btw')).toBe(true)
    expect(desktopSlashUnavailableMessage('/btw')).toBeNull()
  })

  it('distinguishes free prose from finite slash option lists', () => {
    expect(desktopSlashCommandArgumentMode('/goal')).toBe('mixed')
    expect(desktopSlashCommandArgumentMode('/steer')).toBe('text')
    expect(desktopSlashCommandArgumentMode('/queue')).toBe('text')
    expect(desktopSlashCommandArgumentMode('/personality')).toBe('options')
    expect(desktopSlashCommandArgumentMode('/handoff')).toBe('options')
    expect(desktopSlashCommandArgumentMode('/version')).toBeNull()
  })

  it('routes /journey (and aliases) to the memory graph overlay action', () => {
    expect(resolveDesktopCommand('/journey')?.surface).toEqual({ kind: 'action', action: 'journey' })
    expect(resolveDesktopCommand('/memory-graph')?.surface).toEqual({ kind: 'action', action: 'journey' })
    expect(resolveDesktopCommand('/learning')?.surface).toEqual({ kind: 'action', action: 'journey' })
    expect(isDesktopSlashCommand('/journey')).toBe(true)
    expect(isDesktopSlashCommand('/memory-graph')).toBe(true)
    expect(isDesktopSlashSuggestion('/journey')).toBe(true)
    // Aliases execute but stay out of the popover.
    expect(isDesktopSlashSuggestion('/memory-graph')).toBe(false)
    expect(desktopSlashUnavailableMessage('/journey')).toBeNull()
  })

  it('allows aliases to execute without cluttering the popover', () => {
    expect(isDesktopSlashSuggestion('/reset')).toBe(false)
    expect(isDesktopSlashCommand('/reset')).toBe(true)
  })

  it('filters built-in catalog noise but keeps skill / quick-command extensions', () => {
    const filtered = filterDesktopCommandsCatalog({
      categories: [
        {
          name: 'Session',
          pairs: [
            ['/new', 'Start a new session'],
            ['/clear', 'Clear terminal screen']
          ]
        },
        {
          name: 'User commands',
          pairs: [['/ship-it', 'Run release checklist']]
        }
      ],
      pairs: [
        ['/new', 'Start a new session'],
        ['/model', 'Switch model'],
        ['/ship-it', 'Run release checklist']
      ],
      skill_count: 2
    })

    expect(filtered.categories).toEqual([
      { name: 'Session', pairs: [['/new', 'Start a new desktop chat']] },
      { name: 'User commands', pairs: [['/ship-it', 'Run release checklist']] }
    ])
    expect(filtered.pairs).toEqual([
      ['/new', 'Start a new desktop chat'],
      ['/ship-it', 'Run release checklist']
    ])
    // skill_count is recomputed from the filtered output (only /ship-it is an
    // extension command — /new is a built-in) so the /help footer matches what
    // the user actually sees rather than echoing the unfiltered backend total.
    expect(filtered.skill_count).toBe(1)
  })

  it('recomputes skill_count to reflect only extensions surfaced on desktop', () => {
    const filtered = filterDesktopCommandsCatalog({
      pairs: [
        ['/new', 'Start a new session'],
        ['/clear', 'Clear terminal screen'],
        ['/gif-search', 'Search for a gif'],
        ['/ship-it', 'Run release checklist']
      ],
      skill_count: 12
    })

    expect(filtered.pairs?.map(([cmd]) => cmd)).toEqual(['/new', '/gif-search', '/ship-it'])
    expect(filtered.skill_count).toBe(2)
  })

  it('uses desktop-specific labels for commands with different UI behavior', () => {
    expect(desktopSlashDescription('/branch', 'Branch the current session')).toBe(
      'Branch the latest message into a new chat'
    )
    expect(desktopSlashDescription('/skin', 'Show or change the display skin/theme')).toBe(
      'Switch desktop theme or cycle to the next one'
    )
  })

  it('builds /skin completions from desktop themes', () => {
    const completions = desktopSkinSlashCompletions(
      [
        { name: 'mono', label: 'Mono', description: 'Clean grayscale' },
        { name: 'midnight', label: 'Midnight', description: 'Deep blue' },
        { name: 'slate', label: 'Slate', description: 'Cool slate blue' }
      ],
      'mono',
      'm'
    )

    expect(completions).toEqual([
      {
        text: '/skin mono',
        display: '/skin mono',
        meta: 'Mono (current) - Clean grayscale'
      },
      {
        text: '/skin midnight',
        display: '/skin midnight',
        meta: 'Midnight - Deep blue'
      }
    ])
  })

  it('explains known commands that desktop owns elsewhere', () => {
    expect(desktopSlashUnavailableMessage('/model sonnet')).toContain('model picker')
    expect(desktopSlashUnavailableMessage('/skills')).toContain('desktop sidebar')
    expect(desktopSlashUnavailableMessage('/clear')).toContain('terminal interface')
  })

  it('flags /model as a picker-owned command so the desktop opens the overlay', () => {
    expect(isModelPickerCommand('/model')).toBe(true)
    expect(isModelPickerCommand('/model sonnet')).toBe(true)
    expect(isModelPickerCommand('/new')).toBe(false)
    expect(isModelPickerCommand('/skills')).toBe(false)
  })

  it('gives /resume (and its aliases) a first-class session picker surface', () => {
    expect(isPickerCommand('/resume', 'session')).toBe(true)
    expect(isPickerCommand('/sessions', 'session')).toBe(true)
    expect(isPickerCommand('/switch', 'session')).toBe(true)
    // Unlike /model, /resume shows in the popover; its aliases stay hidden.
    expect(isDesktopSlashSuggestion('/resume')).toBe(true)
    expect(isDesktopSlashSuggestion('/sessions')).toBe(false)
    expect(isDesktopSlashCommand('/switch')).toBe(true)
    // The session picker is distinct from the model picker.
    expect(isModelPickerCommand('/resume')).toBe(false)
  })

  it('resolves commands and aliases to their declared surface', () => {
    expect(resolveDesktopCommand('/new')?.surface).toEqual({ kind: 'action', action: 'new' })
    expect(resolveDesktopCommand('/reset')?.surface).toEqual({ kind: 'action', action: 'new' })
    expect(resolveDesktopCommand('/resume')?.surface).toEqual({ kind: 'picker', picker: 'session' })
    expect(resolveDesktopCommand('/usage')?.surface).toEqual({ kind: 'exec' })
    expect(resolveDesktopCommand('/clear')?.surface).toEqual({ kind: 'unavailable', reason: 'terminal' })
    // Skill / quick commands aren't in the registry.
    expect(resolveDesktopCommand('/gif-search')).toBeNull()
  })
})

describe('rankSkillCommands', () => {
  const rows = [
    { text: '/research' },
    { text: '/research-paper-writing' },
    { text: '/work' },
    { text: '/ship-it' },
    { text: '/manim-video' },
    { text: '/docx' }
  ]

  const skills = {
    '/research': { usage: 60, origin: 'local' as const },
    '/research-paper-writing': { usage: 0, origin: 'bundled' as const },
    '/work': { usage: 172, origin: 'local' as const },
    '/manim-video': { usage: 0, origin: 'bundled' as const },
    '/docx': { usage: 0, origin: 'local' as const }
  }

  it('puts the most-used skill first and breaks ties alphabetically', () => {
    expect(rankSkillCommands(rows, skills).map(row => row.text)).toEqual([
      '/work',
      '/research',
      '/docx',
      '/manim-video',
      '/research-paper-writing',
      '/ship-it'
    ])
  })

  it('drops never-used built-ins when browsing, keeping everything else', () => {
    const browsing = rankSkillCommands(rows, skills, { pruneUnusedBuiltins: true }).map(row => row.text)

    expect(browsing).toEqual(['/work', '/research', '/docx', '/ship-it'])
    // A user's own unused skill survives — only shipped-and-ignored goes.
    expect(browsing).toContain('/docx')
    // Unclassified rows (quick commands, skills newer than the map) survive too.
    expect(browsing).toContain('/ship-it')
  })

  it('leaves the backend order untouched when the catalog carries no usage', () => {
    expect(rankSkillCommands(rows, undefined, { pruneUnusedBuiltins: true })).toEqual(rows)
  })

  it('ranks an alias by the canonical command it resolves to', () => {
    const ranked = rankSkillCommands([{ text: '/sessions' }, { text: '/research' }], {
      '/research': { usage: 5, origin: 'local' },
      '/resume': { usage: 900, origin: 'local' }
    })

    expect(ranked.map(row => row.text)).toEqual(['/sessions', '/research'])
  })
})
