import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into de.ts.
export const deSimpleMode = {
  interfaceMode: {
    title: 'Oberflächenmodus',
    hint: 'Ändert, was angezeigt wird, nicht was Hermes kann.',
    sessionNote:
      'Vom einfachen Modus festgelegt. Eine Änderung hier gilt für diese Session; wechseln Sie zu „Erweitert“, um sie dauerhaft zu übernehmen.',
    switchToAdvanced: 'Zu „Erweitert“ wechseln',
    simple: {
      label: 'Einfach',
      description: 'Zum Chatten mit Hermes. Seitenleiste und Chat; keine Terminal-, Datei- oder Diff-Bereiche.'
    },
    advanced: {
      label: 'Erweitert',
      description: 'Für Entwickler. Terminal, Dateien, Diffs, Statusleiste und Layouts – so, wie Sie sie einrichten.'
    }
  },
  chatStarters: {
    label: 'Ideen für den Anfang',
    intro: 'Fragen Sie mich alles – oder beginnen Sie mit einer dieser Ideen.',
    email: {
      title: 'Hilf mir, eine E-Mail zu schreiben',
      prompt: 'Hilf mir, eine E-Mail zu schreiben. An wen sie geht und was ich sagen möchte: '
    },
    dinners: {
      title: 'Abendessen für diese Woche planen',
      prompt: 'Plane fünf einfache Abendessen für diese Woche und gib mir eine gemeinsame Einkaufsliste für alle.'
    },
    summarize: {
      title: 'Ein Dokument zusammenfassen',
      prompt: 'Ich hänge eine Datei an – nenne mir die wichtigsten Punkte.'
    },
    lookup: {
      title: 'Etwas für mich nachschlagen',
      prompt: 'Schlag das bitte für mich nach und sag mir, was du findest: '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
