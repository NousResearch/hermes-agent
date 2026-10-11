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
  PluginNativeNotificationInput,
  PluginNotificationAction,
  PluginOs,
  PluginPet,
  PluginRestOptions,
  PluginRunActionResult,
  PluginSettingsPage,
  PluginSettingsSubpage,
  PluginStorage
} from '@/contrib/plugin'
