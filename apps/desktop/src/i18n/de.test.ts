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

  it('contains no corrupted placeholders or untranslated English leaves', () => {
    // Walks every string leaf in the German catalog and rejects corruption
    // markers that would surface verbatim to the user (e.g. "***", TODO,
    // leftover English sentences). A locale that degrades should fall back to
    // English via defineLocale(), never render a placeholder.
    const corruption = /\*\*\*|FIXME|XXX|PLACEHOLDER|lorem ipsum/i
    const englishSentence = /\b(the|is|are|was|were|your|you|this|that|with|from|click|settings|open|close|save)\b/i

    const visit = (node: unknown, path: string): void => {
      if (typeof node === 'string') {
        expect(corruption.test(node), `corruption in ${path}`).toBe(false)
        // flag stray English only for leaf strings that look like full sentences
        if (node.trim().length > 0 && /\s/.test(node) && englishSentence.test(node)) {
          // allow legitimate English proper nouns / codes by checking word ratio
          const words = node.split(/\s+/).filter(Boolean)
          const englishWords = words.filter(w => englishSentence.test(w)).length
          if (words.length >= 4 && englishWords / words.length > 0.5) {
            throw new Error(`possible untranslated English leaf at ${path}: "${node}"`)
          }
        }
        return
      }
      if (node && typeof node === 'object') {
        for (const [k, v] of Object.entries(node)) {
          visit(v, path ? `${path}.${k}` : k)
        }
      }
    }
    visit(de as unknown as Record<string, unknown>, 'de')
  })
})
