import type { Translations } from './types'

// Settings > Custom Endpoints; `Translations.settings.customEndpoints`.
export interface CustomEndpointsTranslations {
  active: string
  apiKeySet: string
  use: string
  editTitle: string
  addTitle: string
  fields: {
    name: string
    providerId: string
    endpointUrl: string
    defaultModel: string
    context: string
    apiKey: string
    apiKeyNewPlaceholder: string
    apiKeyPlaceholder: string
    apiKeyNoKeySaved: string
    useNewChats: string
    discoverModels: string
  }
  test: string
  save: string
  newEndpoint: string
  apiMode: string
  autoDetect: string
  couldNotLoad: string
  endpointSaved: string
  saveFailed: string
  endpointReachable: string
  endpointReachableTransport: (transport: string) => string
  endpointReachableModels: (reachable: string, count: number) => string
  endpointValidationFailed: string
  validationFailed: string
  activationFailed: string
  deleteConfirm: (name: string) => string
  deleteFailed: string
  title: string
  deleteEndpoint: string
  emptyDescription: string
  emptyTitle: string
  namePlaceholder: string
  contextPlaceholder: string
}

// Composed by en.ts and the locale overlays as `settings.customEndpoints`.
export type CustomEndpointsCopy = Translations['settings']['customEndpoints']
