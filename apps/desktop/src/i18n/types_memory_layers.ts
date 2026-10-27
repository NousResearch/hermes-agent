// Settings > Memory & Context and Command Center > Maintenance:
// which memory provider is active and the always-on built-in layer;
// `Translations.settings.memoryLayers`.
export interface MemoryLayersTranslations {
  activeProvider: string
  noProvider: string
  noProviderDesc: string
  providerDesc: (name: string) => string
  status: {
    ready: string
    needs_config: string
    unavailable: string
    missing: string
  }
  builtinLayer: string
  alwaysOn: string
  builtinDesc: string
}
