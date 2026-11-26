// Settings > Memory & Context and Command Center > Maintenance:
// the active (saved) memory provider, a selected-but-unsaved one, and the
// built-in MEMORY.md / USER.md layer; `Translations.settings.memoryLayers`.
export interface MemoryLayersTranslations {
  activeProvider: string
  selectedProvider: string
  /** `current` is the active provider's name or `builtinMemory`. */
  selectedDesc: (current: string) => string
  builtinMemory: string
  noProvider: string
  noProviderDesc: string
  providerDesc: (name: string) => string
  providerNotReadyDesc: (name: string) => string
  providerUnknownDesc: (name: string) => string
  statusUnavailable: string
  statusUnavailableDesc: string
  retry: string
  status: {
    ready: string
    needs_config: string
    unavailable: string
    missing: string
  }
  builtinLayer: string
  onByDefault: string
  on: string
  off: string
  onlyFile: (file: string) => string
  builtinDesc: string
}
