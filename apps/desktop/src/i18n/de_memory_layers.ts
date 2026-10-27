import type { MemoryLayersTranslations } from './types_memory_layers'

export const deMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Aktiver Speicheranbieter',
  noProvider: 'Keiner — nur integrierter Speicher',
  noProviderDesc:
    'Hermes nutzt nur die integrierte Ebene unten. Wähle unter Speicheranbieter einen externen Anbieter, um einen hinzuzufügen.',
  providerDesc: name => `${name} ergänzt jede Sitzung um eigene Erinnerungen, zusätzlich zur integrierten Ebene.`,
  status: {
    ready: 'bereit',
    needs_config: 'Einrichtung nötig',
    unavailable: 'nicht verfügbar',
    missing: 'nicht installiert'
  },
  builtinLayer: 'Integrierte Ebene',
  alwaysOn: 'Immer aktiv',
  builtinDesc:
    'MEMORY.md und USER.md werden in jede Sitzung geladen, mit oder ohne externen Anbieter. Ein Anbieter ergänzt diese Ebene, er ersetzt sie nie.'
}
