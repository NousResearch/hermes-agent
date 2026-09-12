// The plugin authoring contract types; the SDK barrel re-exports them.

export type { PluginFocusedSessionOwner, PluginSubmitPromptOptions, PluginSubmitPromptResult } from './confirmed-submit'

export type {
  HermesPlugin,
  PetMessageTone,
  PetSayOptions,
  PluginAppActionId,
  PluginAppActionInfo,
  PluginContext,
  PluginContribution,
  PluginFileEntry,
  PluginFiles,
  PluginFilesResult,
  PluginFolder,
  PluginNativeNotificationInput,
  PluginNotificationAction,
  PluginOs,
  PluginPet,
  PluginPreview,
  PluginRestOptions,
  PluginRunActionResult,
  PluginSettingsPage,
  PluginSettingsSubpage,
  PluginStorage
} from '@/contrib/plugin'
