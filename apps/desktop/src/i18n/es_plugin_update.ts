import type { PluginUpdateTranslations } from './types_plugin_update'

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export const esPluginUpdate: PluginUpdateTranslations = {
  updateToPin: (sha: string) => `Actualizar a ${sha}`,
  updateFailed: (name: string) => `No se pudo actualizar ${name}`,
  updated: (name: string) =>
    `${name} se actualizó al commit fijado actual del catálogo. Reinicia el gateway para aplicarlo.`,
  updateConsentTitle: (name: string) => `${name} pide más`,
  updateConsentBody: (name: string, sha: string) =>
    `El nuevo commit fijado de ${name} en el catálogo (${sha}) añade superficies que la versión instalada no tiene. Aplícalo solo si confías en ellas:`,
  updateConsentConfirm: 'Aplicar actualización',
  updateScanTitle: (name: string) => `El análisis de seguridad marcó ${name}`,
  updateScanBody: (name: string, count: number) =>
    `La actualización de ${name} recibió el veredicto «precaución» (${count} ${count === 1 ? 'hallazgo' : 'hallazgos'}). Se analiza cada actualización. Aplícala solo si confías en la fuente:`,
  updateScanMore: (count: number) => `+ ${count} más`,
  updateScanConfirm: 'Actualizar de todos modos',
  updateScanBlocked: (name: string) =>
    `El análisis de seguridad bloqueó la actualización de ${name} (veredicto «peligroso»). No se puede omitir.`,
  updatedGit: (name: string) => `${name} se actualizó al último commit.`
}
