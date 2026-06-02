import { describe, expect, it } from 'vitest'

import {
  desktopSkinSlashCompletions,
  desktopSlashDescription,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  isDesktopSlashSuggestion
} from './desktop-slash-commands'

describe('desktop slash command curation', () => {
  it('keeps core desktop chat commands in suggestions', () => {
    expect(isDesktopSlashSuggestion('/new')).toBe(true)
    expect(isDesktopSlashSuggestion('/branch')).toBe(true)
    expect(isDesktopSlashSuggestion('/skin')).toBe(true)
    expect(isDesktopSlashSuggestion('/usage')).toBe(true)
  })

  it('lets explicitly typed extension commands run without suggesting them', () => {
    expect(isDesktopSlashSuggestion('/my-skill')).toBe(false)
    expect(isDesktopSlashCommand('/my-skill')).toBe(true)
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

  it('filters command catalogs down to core desktop commands', () => {
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

    expect(filtered.categories).toEqual([{ name: 'Session', pairs: [['/new', 'Start a new desktop chat']] }])
    expect(filtered.pairs).toEqual([['/new', 'Start a new desktop chat']])
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
})
