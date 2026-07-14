import { describe, expect, it } from 'vitest'

import {
  desktopSkinSlashCompletions,
  desktopSlashDescription,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  isDesktopSlashSuggestion,
  isModelPickerCommand,
  isPickerCommand,
  resolveDesktopCommand
} from './desktop-slash-commands'

describe('desktop slash command curation', () => {
  it('keeps core desktop chat commands in suggestions', () => {
    expect(isDesktopSlashSuggestion('/new')).toBe(true)
    expect(isDesktopSlashSuggestion('/branch')).toBe(true)
    expect(isDesktopSlashSuggestion('/skin')).toBe(true)
    expect(isDesktopSlashSuggestion('/usage')).toBe(true)
    expect(isDesktopSlashSuggestion('/version')).toBe(true)
    expect(isDesktopSlashSuggestion('/yolo')).toBe(true)
    expect(isDesktopSlashCommand('/yolo')).toBe(true)
  })

  it('surfaces skill and quick commands (extensions) in suggestions and lets them run', () => {
    expect(isDesktopSlashSuggestion('/my-skill')).toBe(true)
    expect(isDesktopSlashSuggestion('/gif-search')).toBe(true)
    expect(isDesktopSlashCommand('/my-skill')).toBe(true)
  })

  it('hides terminal, messaging, and dedicated-UI commands from suggestions', () => {
    expect(isDesktopSlashSuggestion('/clear')).toBe(false)
    expect(isDesktopSlashSuggestion('/compact')).toBe(false)
    expect(isDesktopSlashSuggestion('/redraw')).toBe(false)
    expect(isDesktopSlashSuggestion('/approve')).toBe(false)
    expect(isDesktopSlashSuggestion('/skills')).toBe(false)
    expect(isDesktopSlashSuggestion('/voice')).toBe(false)
    expect(isDesktopSlashSuggestion('/curator')).toBe(false)
  })

  it('surfaces /model, /tools, /save, and /personality on the desktop', () => {
    expect(isDesktopSlashSuggestion('/model')).toBe(true)
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
    expect(resolveDesktopCommand('/pet')?.args).toBe(true)
    expect(isDesktopSlashSuggestion('/pet')).toBe(true)
    expect(isDesktopSlashCommand('/pet')).toBe(true)
    expect(resolveDesktopCommand('/pets')?.surface).toEqual({ kind: 'unavailable', reason: 'settings' })
    expect(isDesktopSlashSuggestion('/pets')).toBe(false)
    expect(isDesktopSlashCommand('/pets')).toBe(false)
  })

  it('treats /browser as an executable action command (local-gateway connect)', () => {
    // /browser used to be terminal-only; it now resolves to a desktop action
    // handler that routes browser.manage RPC when the gateway is local.
    expect(isDesktopSlashCommand('/browser')).toBe(true)
    expect(isDesktopSlashSuggestion('/browser')).toBe(true)
    expect(desktopSlashUnavailableMessage('/browser')).toBeNull()
    expect(resolveDesktopCommand('/browser')?.surface).toEqual({ kind: 'action', action: 'browser' })
    // Bare /browser expands to its sub-action options in the popover.
    expect(resolveDesktopCommand('/browser')?.args).toBe(true)
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
      ['/model', 'Switch the model for this session'],
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
    expect(desktopSlashUnavailableMessage('/verbose')).toContain('terminal interface')
  })

  it('routes /footer to a desktop action with arg options', () => {
    expect(resolveDesktopCommand('/footer')?.surface).toEqual({ kind: 'action', action: 'footer' })
    expect(resolveDesktopCommand('/footer')?.args).toBe(true)
    expect(isDesktopSlashSuggestion('/footer')).toBe(true)
    expect(desktopSlashUnavailableMessage('/footer')).toBeNull()
  })

  it('routes /reasoning (and /effort) to the reasoning action with arg options', () => {
    expect(resolveDesktopCommand('/reasoning')?.surface).toEqual({ kind: 'action', action: 'reasoning' })
    expect(resolveDesktopCommand('/effort')?.surface).toEqual({ kind: 'action', action: 'reasoning' })
    expect(resolveDesktopCommand('/reasoning')?.args).toBe(true)
    expect(isDesktopSlashSuggestion('/reasoning')).toBe(true)
    expect(isDesktopSlashSuggestion('/effort')).toBe(false) // alias: executes, stays out of the popover
    expect(desktopSlashUnavailableMessage('/reasoning')).toBeNull()
  })

  it('treats /clear and /models as aliases of /new and /model', () => {
    expect(resolveDesktopCommand('/clear')?.name).toBe('/new')
    expect(resolveDesktopCommand('/models')?.name).toBe('/model')
    expect(isDesktopSlashCommand('/clear')).toBe(true)
    expect(isDesktopSlashCommand('/models')).toBe(true)
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
    // Like /model, /resume shows in the popover; its aliases stay hidden.
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
    expect(resolveDesktopCommand('/clear')?.surface).toEqual({ kind: 'action', action: 'new' })
    expect(resolveDesktopCommand('/verbose')?.surface).toEqual({ kind: 'unavailable', reason: 'terminal' })
    // Skill / quick commands aren't in the registry.
    expect(resolveDesktopCommand('/gif-search')).toBeNull()
  })
})
