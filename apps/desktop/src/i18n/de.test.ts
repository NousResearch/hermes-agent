import { describe, expect, it } from 'vitest'

import { de } from './de'
import { en } from './en'

// Guards the German locale against the regressions the hermes-sweeper flagged:
// visible copy left in English, and import/export label collisions. These are
// behaviour contracts (relationships), not snapshots of specific wording.
describe('de locale content', () => {
  it('translates visible boot-failure and update copy', () => {
    expect(de.boot.failure.title).not.toBe(en.boot.failure.title)
    expect(de.boot.failure.description).not.toBe(en.boot.failure.description)
    expect(de.notifications.seeWhatsNew).not.toBe(en.notifications.seeWhatsNew)
  })

  it('renders keybind action labels in German, not English identifiers', () => {
    // These values are display copy shown in the command palette, not IDs.
    for (const key of Object.keys(en.keybinds.actions) as (keyof typeof en.keybinds.actions)[]) {
      expect(de.keybinds.actions[key]).not.toBe(en.keybinds.actions[key])
    }
  })

  it('does not collide import and export labels', () => {
    expect(de.settings.importConfig).not.toBe(de.settings.exportConfig)
    expect(de.settings.importConfig).not.toBe(en.settings.importConfig)
  })
})
