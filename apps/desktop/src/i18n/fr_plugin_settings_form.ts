import type { PluginSettingsFormCopy } from './types_plugin_settings_form'

export const frPluginSettingsForm: Partial<PluginSettingsFormCopy> = {
  save: 'Enregistrer les paramètres',
  saved: (name: string) => `Paramètres de ${name} enregistrés.`,
  saveFailed: (name: string) => `Impossible d’enregistrer les paramètres de ${name}`,
  required: 'Obligatoire',
  secretSet: '•••••••• (défini)',
  secretEmpty: 'Vide. Collez ici',
  secretStoredAs: (env: string) =>
    `Stocké dans le .env du profil sous ${env}, jamais dans config.yaml ; laissez vide pour conserver la valeur actuelle.`
}
