import type { PluginSettingsFormCopy } from './types_plugin_settings_form'

export const enPluginSettingsForm: PluginSettingsFormCopy = {
  save: 'Save settings',
  saved: (name: string) => `${name} settings saved.`,
  saveFailed: (name: string) => `Could not save ${name} settings`,
  required: 'Required',
  optional: '(optional)',
  secretSet: '•••••••• (set)',
  secretEmpty: 'Not set. Paste here',
  secretStoredAs: (env: string) =>
    `Stored in the profile's .env as ${env}, never in config.yaml; leave blank to keep the current value.`
}
