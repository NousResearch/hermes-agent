import { describe, expect, it, vi } from 'vitest'

import { composerPlainText, RICH_INPUT_SLOT, stageSlashCommandIntoEditor } from '@/app/chat/composer/rich-editor'
import { desktopChatActionCommands } from '@/lib/desktop-slash-commands'

import { buildChatActionsGroup } from './chat-actions'

import type { PaletteItem } from './index'

// Mirror the palette ranker's AND-match gate (scoreItem in index.tsx): a search
// hits a row when every typed term appears in its label or its keywords. Kept
// local so this test needn't import the whole palette component.
const matches = (item: PaletteItem, search: string): boolean => {
  const label = item.label.toLowerCase()
  const keys = (item.keywords ?? []).join(' ').toLowerCase()

  return search
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every(term => label.includes(term) || keys.includes(term))
}

const findRow = (items: PaletteItem[], command: string) => items.find(item => item.id === `chat-action-${command}`)

describe('desktopChatActionCommands', () => {
  const commands = desktopChatActionCommands()
  const names = commands.map(c => c.command)

  it('includes runnable chat slash actions', () => {
    expect(names).toEqual(expect.arrayContaining(['/compress', '/title', '/branch', '/handoff']))
  })

  it('excludes destructive commands (rollback / undo / clear / reset·new)', () => {
    expect(names).not.toContain('/rollback')
    expect(names).not.toContain('/undo')
    expect(names).not.toContain('/clear')
    // `/reset` is `/new`'s alias, so the whole `/new` entry is dropped.
    expect(names).not.toContain('/new')
    expect(names).not.toContain('/reset')
  })

  it('excludes overlay pickers and hidden / unavailable commands', () => {
    expect(names).not.toContain('/model') // hidden
    expect(names).not.toContain('/resume') // picker overlay
    expect(names).not.toContain('/config') // terminal-only (unavailable)
  })

  it('flags arg-taking commands', () => {
    expect(commands.find(c => c.command === '/handoff')?.takesArgs).toBe(true)
    expect(commands.find(c => c.command === '/compress')?.takesArgs).toBe(false)
  })
})

describe('buildChatActionsGroup', () => {
  const base = {
    heading: 'Chat actions',
    hint: 'Open a chat first',
    onStage: () => undefined
  }

  it('returns a labelled group of rows', () => {
    const group = buildChatActionsGroup({ ...base, hasActiveSession: true })

    expect(group).not.toBeNull()
    expect(group!.heading).toBe('Chat actions')
    expect(group!.items.length).toBeGreaterThan(0)
  })

  it('matches a row by BOTH its plain-English label and its /command string', () => {
    const { items } = buildChatActionsGroup({ ...base, hasActiveSession: true })!
    const compress = findRow(items, '/compress')!
    const handoff = findRow(items, '/handoff')!

    // Plain English.
    expect(matches(compress, 'compress')).toBe(true)
    expect(matches(handoff, 'handoff')).toBe(true)
    // Literal slash string.
    expect(matches(compress, '/compress')).toBe(true)
    expect(matches(handoff, '/handoff')).toBe(true)
    // A miss stays a miss (no accidental catch-all).
    expect(matches(compress, 'handoff')).toBe(false)
  })

  it('stages (never executes) the command on select — the only side effect is onStage', () => {
    const onStage = vi.fn()
    const { items } = buildChatActionsGroup({ ...base, hasActiveSession: true, onStage })!

    findRow(items, '/compress')!.run!()

    expect(onStage).toHaveBeenCalledTimes(1)
    expect(onStage).toHaveBeenCalledWith('/compress')
  })

  it('disables every row with a hint while no session is active', () => {
    const { items } = buildChatActionsGroup({ ...base, hasActiveSession: false })!

    expect(items.every(item => item.disabled === true)).toBe(true)
    expect(items.every(item => item.hint === 'Open a chat first')).toBe(true)
  })

  it('enables rows (no hint) once a session is active', () => {
    const { items } = buildChatActionsGroup({ ...base, hasActiveSession: true })!

    expect(items.every(item => !item.disabled)).toBe(true)
    expect(items.every(item => item.hint === undefined)).toBe(true)
  })

  it('omits the group entirely when nothing is eligible', () => {
    expect(buildChatActionsGroup({ ...base, hasActiveSession: true, commands: [] })).toBeNull()
  })
})

describe('stageSlashCommandIntoEditor — the chip-insertion path', () => {
  const makeEditor = () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)

    return editor
  }

  it('commits a no-arg command as a /command chip, not raw text', () => {
    const editor = makeEditor()

    const draft = stageSlashCommandIntoEditor(editor, '/compress', { takesArgs: false })

    const chip = editor.querySelector<HTMLElement>('[data-slash-kind]')
    expect(chip).not.toBeNull()
    expect(chip!.dataset.slashKind).toBe('command')
    expect(chip!.dataset.refText).toBe('/compress')
    // Round-trips to the exact command text a send would submit (chip + space).
    expect(draft).toBe('/compress ')
    expect(composerPlainText(editor)).toBe('/compress ')

    editor.remove()
  })

  it('parks an arg-taking command as `/command ` text so its arg step can open', () => {
    const editor = makeEditor()

    const draft = stageSlashCommandIntoEditor(editor, '/handoff', { takesArgs: true })

    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(draft).toBe('/handoff ')

    editor.remove()
  })

  it('replaces any prior draft so the slash directive leads the input', () => {
    const editor = makeEditor()
    editor.textContent = 'half-typed thought'

    stageSlashCommandIntoEditor(editor, '/title', { takesArgs: false })

    expect(composerPlainText(editor)).toBe('/title ')

    editor.remove()
  })
})
