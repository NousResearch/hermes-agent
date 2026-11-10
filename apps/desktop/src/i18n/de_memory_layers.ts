import type { MemoryLayersTranslations } from './types_memory_layers'

export const deMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Aktiver Speicheranbieter',
  selectedProvider: 'Ausgewählter Speicheranbieter',
  selectedDesc: current => `Noch nicht aktiv: Sitzungen nutzen weiter ${current}, bis diese Auswahl gespeichert ist.`,
  builtinMemory: 'nur integrierten Speicher',
  noProvider: 'Keiner — nur integrierter Speicher',
  noProviderDesc:
    'Hermes nutzt nur die integrierte Ebene unten. Wähle unter Einstellungen › Speicher & Kontext einen externen Anbieter, um einen hinzuzufügen.',
  providerDesc: name => `${name} ergänzt jede Sitzung um eigene Erinnerungen, zusätzlich zur integrierten Ebene.`,
  providerNotReadyDesc: name =>
    `${name} ist der gespeicherte Anbieter, liefert aber erst Erinnerungen, wenn er bereit ist. Die integrierte Ebene funktioniert so oder so.`,
  providerUnknownDesc: name => `${name} ist der gespeicherte Anbieter; Hermes konnte nicht lesen, ob er bereit ist.`,
  statusUnavailable: 'Speicherstatus nicht verfügbar',
  statusUnavailableDesc: 'Konnte nicht lesen, welcher Speicheranbieter aktiv ist und ob er bereit ist.',
  retry: 'Erneut versuchen',
  status: {
    ready: 'bereit',
    needs_config: 'Einrichtung nötig',
    unavailable: 'nicht verfügbar',
    missing: 'nicht installiert'
  },
  builtinLayer: 'Integrierte Ebene',
  onByDefault: 'Standardmäßig an',
  on: 'An',
  off: 'Aus',
  onlyFile: file => `nur ${file}`,
  builtinDesc:
    'MEMORY.md und USER.md sind Hermes’ eigener Speicher, unabhängig von externen Anbietern. Beide werden standardmäßig in Sitzungen geladen; die Schalter für Speicher und Benutzerprofil unter Einstellungen › Speicher & Kontext schalten sie einzeln ab. Ein Anbieter ergänzt diese Ebene, er ersetzt sie nie.'
}
