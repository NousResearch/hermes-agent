import { describe, expect, it } from 'vitest'

import {
  desktopSkinSlashCompletions,
  desktopSlashDescription,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  isDesktopSlashExtensionCommand,
  isDesktopSlashSuggestion,
  isModelPickerCommand
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

  it('treats omitted Hermes built-ins as built-ins, not skill extensions', () => {
    for (const command of ['/bundles', '/codex-runtime', '/handoff', '/sessions', '/whoami']) {
      // Not mistaken for a skill / quick-command extension…
      expect(isDesktopSlashExtensionCommand(command)).toBe(false)
      // …so they stay out of the slash palette…
      expect(isDesktopSlashSuggestion(command)).toBe(false)
      // …and report a clear "not surfaced here" message instead of running.
      expect(isDesktopSlashCommand(command)).toBe(false)
      expect(desktopSlashUnavailableMessage(command)).toBeTruthy()
    }
  })

  it('routes the /codex_runtime alias like its canonical built-in', () => {
    expect(isDesktopSlashExtensionCommand('/codex_runtime')).toBe(false)
    expect(isDesktopSlashSuggestion('/codex_runtime')).toBe(false)
    expect(isDesktopSlashCommand('/codex_runtime')).toBe(false)
  })

  it('surfaces /subgoal alongside its companion /goal', () => {
    expect(isDesktopSlashExtensionCommand('/subgoal')).toBe(false)
    expect(isDesktopSlashSuggestion('/subgoal')).toBe(true)
    expect(isDesktopSlashCommand('/subgoal')).toBe(true)
    expect(desktopSlashDescription('/subgoal', 'fallback')).toBe('Manage extra criteria for the active goal')
  })

  it('lets the /v alias execute like /version without cluttering the popover', () => {
    expect(isDesktopSlashExtensionCommand('/v')).toBe(false)
    expect(isDesktopSlashSuggestion('/v')).toBe(false)
    expect(isDesktopSlashCommand('/v')).toBe(true)
  })

  it('hides terminal, messaging, and dedicated-UI commands from suggestions', () => {
    expect(isDesktopSlashSuggestion('/clear')).toBe(false)
    expect(isDesktopSlashSuggestion('/compact')).toBe(false)
    expect(isDesktopSlashSuggestion('/redraw')).toBe(false)
    expect(isDesktopSlashSuggestion('/approve')).toBe(false)
    expect(isDesktopSlashSuggestion('/model')).toBe(false)
    expect(isDesktopSlashSuggestion('/skills')).toBe(false)
    expect(isDesktopSlashSuggestion('/voice')).toBe(false)
    expect(isDesktopSlashSuggestion('/curator')).toBe(false)
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
})
