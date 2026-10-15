export interface CustomEndpointsTranslations {
  credentials: {
    authentication: string
    azureCli: string
    azureCliHint: string
    command: string
    commandLabel: string
    commandHint: string
    commandPlaceholder: string
    savedCommandPlaceholder: string
  }
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
