import type { MemoryLayersTranslations } from './types_memory_layers'

// Settings > Memory & Context and Command Center > Maintenance (#145175).
export const enMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Active memory provider',
  selectedProvider: 'Selected memory provider',
  selectedDesc: current => `Not active yet: sessions keep using ${current} until this choice is saved.`,
  builtinMemory: 'built-in memory only',
  noProvider: 'None — built-in memory only',
  noProviderDesc:
    'Hermes uses only the built-in layer below. Choose an external provider under Settings › Memory & Context to add one.',
  providerDesc: name => `${name} adds its own recall to every session, on top of the built-in layer.`,
  providerNotReadyDesc: name =>
    `${name} is the saved provider but adds no recall until it is ready. The built-in layer works either way.`,
  providerUnknownDesc: name => `${name} is the saved provider; Hermes couldn't read whether it is ready.`,
  statusUnavailable: 'Memory status unavailable',
  statusUnavailableDesc: "Couldn't read which memory provider is active or whether it is ready.",
  retry: 'Retry',
  status: {
    ready: 'ready',
    needs_config: 'needs setup',
    unavailable: 'unavailable',
    missing: 'not installed'
  },
  builtinLayer: 'Built-in layer',
  onByDefault: 'On by default',
  on: 'On',
  off: 'Off',
  onlyFile: file => `${file} only`,
  builtinDesc:
    'MEMORY.md and USER.md are Hermes’s own memory, separate from any external provider. Both load into sessions by default; the memory and user profile toggles in Settings › Memory & Context turn each off. A provider adds to this layer; it never replaces it.'
}
