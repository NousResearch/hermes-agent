import type { PluginSettingsFormCopy } from './types_plugin_settings_form'

export const esPluginSettingsForm: Partial<PluginSettingsFormCopy> = {
  save: 'Guardar configuración',
  saved: (name: string) => `Configuración de ${name} guardada.`,
  saveFailed: (name: string) => `No se pudo guardar la configuración de ${name}`,
  required: 'Obligatorio',
  secretSet: '•••••••• (configurado)',
  secretEmpty: 'Vacío. Pega aquí',
  secretStoredAs: (env: string) =>
    `Se guarda en el .env del perfil como ${env}, nunca en config.yaml; déjalo en blanco para conservar el valor actual.`
}
