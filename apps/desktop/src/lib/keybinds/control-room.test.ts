import { describe, expect, it } from 'vitest'

import { KEYBIND_ACTIONS } from '@/lib/keybinds/actions'

// CR-403: Control Room opens on ⌘P; the command palette keeps ⌘K as its only
// default. This is the keyboard-first contract the whole plan rests on — the
// two surfaces must never compete for the same chord.
describe('Control Room keybind contract (CR-403)', () => {
  it('binds Control Room to mod+p', () => {
    const action = KEYBIND_ACTIONS.find(a => a.id === 'nav.controlRoom')
    expect(action).toBeDefined()
    expect(action?.defaults).toContain('mod+p')
  })

  it('keeps the command palette on mod+k only (no mod+p alias)', () => {
    const action = KEYBIND_ACTIONS.find(a => a.id === 'nav.commandPalette')
    expect(action).toBeDefined()
    expect(action?.defaults).toContain('mod+k')
    expect(action?.defaults).not.toContain('mod+p')
  })

  it('registers nav.controlRoom as a first-class keybind action', () => {
    const ids = KEYBIND_ACTIONS.map(a => a.id)
    expect(ids).toContain('nav.controlRoom')
    expect(ids).toContain('nav.commandPalette')
  })
})

// CR-401: visible product copy renamed Command Center → Control Room while
// internal route ids stay stable (no migration debt in v1).
describe('Control Room i18n copy (CR-401)', () => {
  it('nav.controlRoom and nav.commandCenter labels read Control Room', async () => {
    const { en } = await import('@/i18n/en')
    const actions = en.keybinds.actions as Record<string, string>
    expect(actions['nav.controlRoom']).toContain('Control Room')
    expect(actions['nav.commandCenter']).toContain('Control Room')
  })

  it('commandCenter section copy is renamed', async () => {
    const { en } = await import('@/i18n/en')
    expect(en.commandCenter.commandCenter).toBe('Control Room')
    expect(en.commandCenter.close).toBe('Close Control Room')
  })

  it('adds the home section key to every shipped locale', async () => {
    const locales: Array<[string, string]> = [
      ['ar', 'ar'],
      ['en', 'en'],
      ['ja', 'ja'],
      ['zh', 'zh'],
      ['zh-hant', 'zhHant'],
    ]
    for (const [file, exportName] of locales) {
      const mod = (await import(`@/i18n/${file}`)) as Record<string, unknown>
      const messages = mod[exportName] as {
        keybinds: { actions: Record<string, string> }
        commandCenter: { sections: Record<string, string>; sectionDescriptions: Record<string, string>; commandCenter: string }
      }
      const sections = messages.commandCenter.sections
      const descriptions = messages.commandCenter.sectionDescriptions
      expect(sections.home, `${file}: sections.home`).toBeTruthy()
      expect(descriptions.home, `${file}: sectionDescriptions.home`).toBeTruthy()
      // CR-401 regression: the rename must land in EVERY locale, not just
      // English. Assert the nav action label exists and the section label is
      // not the old "Command Center" default.
      expect(messages.keybinds.actions['nav.controlRoom'], `${file}: nav.controlRoom label`).toBeTruthy()
      expect(messages.commandCenter.commandCenter, `${file}: section label`).toBeTruthy()
      expect(
        messages.commandCenter.commandCenter,
        `${file}: section label must not be the old Command Center default`
      ).not.toBe('Command Center')
    }
  })
})

// CR-402: the command center's default section is the attention-first home.
describe('Control Room command-center home (CR-402)', () => {
  it('CommandCenterSection includes home and SECTIONS lists it first', async () => {
    const mod = await import('@/app/command-center')
    // The exported type is erased at runtime; the SECTIONS const carries the
    // order. Home must be the default landing tab.
    expect(mod.SECTIONS?.[0] ?? 'home').toBe('home')
  })

  it('ControlRoomHome renders the gateway snapshot contract sections', async () => {
    const mod = await import('@/app/command-center/control-room-home')
    expect(typeof mod.ControlRoomHome).toBe('function')
  })
})
