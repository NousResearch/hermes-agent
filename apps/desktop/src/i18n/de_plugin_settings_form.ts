import type { PluginSettingsFormCopy } from './types_plugin_settings_form'

export const dePluginSettingsForm: Partial<PluginSettingsFormCopy> = {
  save: 'Einstellungen speichern',
  saved: (name: string) => `Einstellungen von ${name} gespeichert.`,
  saveFailed: (name: string) => `Einstellungen von ${name} konnten nicht gespeichert werden`,
  required: 'Erforderlich',
  secretSet: '•••••••• (gesetzt)',
  secretEmpty: 'Leer. Hier einfügen',
  secretStoredAs: (env: string) =>
    `Wird in der .env des Profils als ${env} gespeichert, nie in config.yaml; leer lassen, um den aktuellen Wert zu behalten.`
}
