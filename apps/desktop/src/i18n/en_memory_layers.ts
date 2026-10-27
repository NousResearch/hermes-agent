import type { MemoryLayersTranslations } from './types_memory_layers'

// Settings > Memory & Context and Command Center > Maintenance (#145175).
export const enMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Active memory provider',
  noProvider: 'None — built-in memory only',
  noProviderDesc: 'Hermes uses only the built-in layer below. Pick an external provider in Memory Provider to add one.',
  providerDesc: name => `${name} adds its own recall to every session, on top of the built-in layer.`,
  status: {
    ready: 'ready',
    needs_config: 'needs setup',
    unavailable: 'unavailable',
    missing: 'not installed'
  },
  builtinLayer: 'Built-in layer',
  alwaysOn: 'Always on',
  builtinDesc:
    'MEMORY.md and USER.md load into every session, with or without an external provider. A provider adds to this layer; it never replaces it.'
}
