import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const deCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: 'Kontext-Engine'
  },
  compression: {
    enabled: 'Auto-Kompression',
    threshold: 'Kompression-Schwelle',
    codexGpt55Autoraise: 'Automatische Codex-Komprimierungsanhebung',
    targetRatio: 'Kompression-Ziel',
    protectLastN: 'Geschützte letzte Nachrichten',
    warmHandoff: 'Warme Übergabe'
  },
  auxiliary: {
    compression: {
      timeout: 'Timeout des Komprimierungsmodells (s)'
    }
  }
}

export const deCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: 'Strategie zur Verwaltung langer Gespräche nahe der Kontextgrenze.'
  },
  compression: {
    enabled: 'Älteren Kontext zusammenfassen, wenn Gespräche groß werden.',
    codexGpt55Autoraise: 'Komprimierung bei unterstützten ChatGPT-Codex-OAuth-Modellen auf 85 % anheben.',
    warmHandoff:
      'Das Hauptmodell schreibt die Kompressionszusammenfassung auf seinem zwischengespeicherten Prompt, sodass ein Server mit Prompt-Caching nur die neuen Nachrichten liest. Schneller, wenn die Kompression dasselbe Modell nutzt. Auto: nur wenn die Kompression das Hauptmodell nutzt und der Server zwischengespeicherte Tokens meldet. An: immer versuchen. Aus: immer das Kompressionsmodell nutzen. Bei jedem Fehler wird die normale Zusammenfassung verwendet.'
  },
  auxiliary: {
    compression: {
      timeout:
        'Sekunden, die pro Aufruf auf das Hilfsmodell für Komprimierung gewartet wird (Standard 120). Für langsame lokale Modelle erhöhen.'
    }
  }
}
