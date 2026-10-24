import type { CustomEndpointsTranslations } from './types_custom_endpoints'

// Settings > Providers > Custom endpoints, composed by en.ts as `settings.customEndpoints`.
export const enCustomEndpoints: CustomEndpointsTranslations = {
  active: 'Active',
  apiKeySet: 'API key set',
  use: 'Use',
  editTitle: 'Edit Endpoint',
  addTitle: 'Add Endpoint',
  fields: {
    name: 'Name',
    providerId: 'Provider ID',
    endpointUrl: 'Endpoint URL',
    defaultModel: 'Default Model',
    context: 'Context',
    apiKey: 'API Key',
    apiKeyNewPlaceholder: 'Leave blank to keep current key',
    apiKeyPlaceholder: 'Optional',
    useNewChats: 'Use for new chats',
    discoverModels: 'Discover models'
  },
  test: 'Test',
  save: 'Save',
  newEndpoint: 'New endpoint',
  apiMode: 'API Mode',
  autoDetect: 'Auto-detect',
  couldNotLoad: 'Could not load custom endpoints',
  endpointSaved: 'Custom endpoint saved.',
  saveFailed: 'Save failed',
  endpointReachable: 'Endpoint is reachable.',
  endpointReachableTransport: transport => `Endpoint is reachable (${transport} route served).`,
  endpointReachableModels: (reachable, count) => `${reachable} Found ${count} models.`,
  endpointValidationFailed: 'Endpoint validation failed.',
  validationFailed: 'Validation failed',
  activationFailed: 'Activation failed',
  deleteConfirm: name => `Delete ${name}?`,
  deleteFailed: 'Delete failed',
  title: 'Custom Endpoints',
  deleteEndpoint: 'Delete endpoint',
  emptyDescription: 'Add an OpenAI-compatible endpoint below.',
  emptyTitle: 'No custom endpoints',
  namePlaceholder: 'Axet Proxy',
  contextPlaceholder: 'Auto'
}
