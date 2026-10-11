import type { AuxTaskCopyMap } from './types_aux_tasks'
export interface MoaStudioTranslations {
  title: string
  description: string
  preset: string
  presetName: string
  newPresetName: string
  renamePreset: string
  addPreset: string
  duplicatePreset: string
  deletePreset: string
  deleteConfirm: (name: string) => string
  setDefault: string
  defaultBadge: string
  activeBadge: string
  useInThisChat: string
  presetEnabled: string
  enabledDescription: string
  referencesTitle: string
  reference: (index: number) => string
  referenceProvider: (index: number) => string
  referenceModel: (index: number) => string
  referenceEffort: (index: number) => string
  referenceEnabled: (index: number) => string
  aggregator: string
  aggregatorProvider: string
  aggregatorModel: string
  aggregatorEffort: string
  providerDefault: string
  addReference: string
  removeReference: (index: number) => string
  moveReferenceUp: (index: number) => string
  moveReferenceDown: (index: number) => string
  moveUp: string
  moveDown: string
  remove: string
  executionTitle: string
  cadence: string
  cadenceDescription: string
  oncePerUserTurn: string
  everyNIterations: string
  everyNCount: string
  everyToolIteration: string
  advisorTemperature: string
  advisorTemperatureDescription: string
  aggregatorTemperature: string
  aggregatorTemperatureDescription: string
  saveChanges: string
  unsaved: string
  saving: string
  saved: string
  saveFailedRetained: string
  incomplete: string
  nameBlank: string
  nameDuplicate: string
  unavailable: string
}

export interface ModelSettingsTranslations {
  setupProviderFallback: string
  setUpProvider: (name: string) => string
  staleAuxBefore: (count: number, names: string) => string
  staleAuxAfter: string
  staleAuxOtherProviders: string
  moaEnabled: string
  moaSetDefault: string
  moaNewPresetPlaceholder: string
  moaAddPreset: string
  customModel: string
  customModelPlaceholder: string
  chooseFromList: string
  moaDefault: string
  moaReferenceToggle: (enabled: boolean, index: number) => string
  moaReferenceTitle: (index: number) => string
  moaAddReference: string
  loading: string
  appliesDesc: string
  provider: string
  model: string
  applying: string
  mainAppliedTitle: string
  mainAppliedMessage: (model: string) => string
  defaultsLabel: string
  reasoning: string
  reasoningOff: string
  speed: string
  speedStandard: string
  defaultsFailed: string
  loadFailed: string
  restartRequired: string
  restartBackend: string
  restartingBackend: string
  restartFailed: string
  auxiliaryTitle: string
  resetAllToMain: string
  staleAuxDismiss: string
  auxiliaryDesc: string
  setToMain: string
  change: string
  autoUseMain: string
  inheritMainEffort: string
  inheritsFrom: (task: string) => string
  followTask: (task: string) => string
  providerDefault: string
  fallbackAdd: string
  fallbackEmpty: string
  notInCatalog: string
  moaTitle: string
  moaPreset: string
  moaDescription: string
  moaAggregator: string
  moaAggregatorBilled: string
  moaReferenceHint: string
  tasks: AuxTaskCopyMap
}
