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

  it('preserves distinct Starmap import semantics in German', () => {
    // importMap, importBtn, importEmpty, importSuccess, importedBadge must all differ
    expect(de.starmap.importMap).not.toBe(de.starmap.copied)
    expect(de.starmap.importBtn).not.toBe(de.starmap.copied)
    expect(de.starmap.importEmpty).not.toBe(de.starmap.copied)
    expect(de.starmap.importSuccess(1)).not.toBe(de.starmap.copied)
    expect(de.starmap.importSuccess(2)).not.toBe(de.starmap.copied)
    expect(de.starmap.importedBadge).not.toBe(de.starmap.copied)

    // success message interpolates the node count
    expect(de.starmap.importSuccess(1)).toContain('1')
    expect(de.starmap.importSuccess(3)).toContain('3')

    // all five import strings must be pairwise distinct
    const values = [
      de.starmap.importMap,
      de.starmap.importBtn,
      de.starmap.importEmpty,
      de.starmap.importSuccess(1),
      de.starmap.importedBadge
    ]
    expect(new Set(values).size).toBe(values.length)
  })

  it('uses count-free singular for resumeWhenBackgroundDone(1)', () => {
    expect(de.assistant.thread.resumeWhenBackgroundDone(1)).toBe(
      'Wird fortgesetzt, wenn die Hintergrundaufgabe abgeschlossen ist'
    )
  })

  it('interpolates count in plural resumeWhenBackgroundDone', () => {
    expect(de.assistant.thread.resumeWhenBackgroundDone(2)).toBe(
      'Wird fortgesetzt, wenn 2 Hintergrundaufgaben abgeschlossen sind'
    )
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

  it('does not duplicate config.imported and autosaveFailed', () => {
    // Regression guard: both were 'Automatisches Speichern fehlgeschlagen'
    expect(de.settings.config.imported).not.toBe(de.settings.config.autosaveFailed)
    expect(de.settings.config.imported).toBe('Konfiguration importiert')
  })

  it('translates about.justNowSuffix to German', () => {
    // Regression guard: was left as English '· just now'
    expect(de.settings.about.justNowSuffix).not.toBe(en.settings.about.justNowSuffix)
    expect(de.settings.about.justNowSuffix).toContain('gerade eben')
  })

  it('uses distinct text for appearance.importedBadge and removeTheme', () => {
    // Regression guard: both were 'Design entfernen'
    expect(de.settings.appearance.importedBadge).not.toBe(de.settings.appearance.removeTheme)
    expect(de.settings.appearance.importedBadge).toBe('Importiert')
  })

  it('translates profile delete dialog fragments to German', () => {
    // Review fix: deleteDescPrefix/Mid/Suffix were left in English.
    // DeleteProfileDialog concatenates: prefix + name + mid + path + suffix
    expect(de.profiles.deleteDescPrefix).not.toBe(en.profiles.deleteDescPrefix)
    expect(de.profiles.deleteDescMid).not.toBe(en.profiles.deleteDescMid)
    expect(de.profiles.deleteDescSuffix).not.toBe(en.profiles.deleteDescSuffix)
    // Assembled description must read as a coherent German sentence
    const assembled = `${de.profiles.deleteDescPrefix}MeinProfil${de.profiles.deleteDescMid}/pfad${de.profiles.deleteDescSuffix}`
    expect(assembled).toBe(
      'Dies löscht MeinProfil und entfernt sein /pfad Verzeichnis. Dies kann nicht rückgängig gemacht werden.'
    )
  })

  it('translates profile rename dialog prefix to German', () => {
    // Review fix: renameDescPrefix was left in English.
    // RenameProfileDialog renders: prefix + path + suffix
    expect(de.profiles.renameDescPrefix).not.toBe(en.profiles.renameDescPrefix)
    const assembled = `${de.profiles.renameDescPrefix}~/.local/bin${de.profiles.renameDescSuffix}`
    expect(assembled).toBe(
      'Das Umbenennen aktualisiert das Profilverzeichnis und alle Wrapper-Skripte in ~/.local/bin.'
    )
  })

  it('translates cron delete dialog fragments to German', () => {
    // Review fix: cron deleteDescPrefix/Suffix were left in English.
    // Cron delete dialog concatenates: prefix + title + suffix
    expect(de.cron.deleteDescPrefix).not.toBe(en.cron.deleteDescPrefix)
    expect(de.cron.deleteDescSuffix).not.toBe(en.cron.deleteDescSuffix)
    const assembled = `${de.cron.deleteDescPrefix}Morgen-Briefing${de.cron.deleteDescSuffix}`
    expect(assembled).toBe(
      'Dies entfernt Morgen-Briefing dauerhaft. Die Ausführung wird sofort gestoppt.'
    )
  })

  it('translates install.currentStage to German', () => {
    // Review fix: currentStage() returned English '-- now: …'.
    // The generic leaf walker does not execute function-valued translations,
    // so this needs a direct assertion.
    expect(de.install.currentStage('Python wird installiert')).not.toBe(
      en.install.currentStage('Python wird installiert')
    )
    expect(de.install.currentStage('Abhängigkeiten')).toBe(' -- jetzt: Abhängigkeiten')
  })
})
