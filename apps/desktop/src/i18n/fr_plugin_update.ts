import type { PluginUpdateTranslations } from './types_plugin_update'

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export const frPluginUpdate: PluginUpdateTranslations = {
  updateToPin: (sha: string) => `Mettre à jour vers ${sha}`,
  updateFailed: (name: string) => `Impossible de mettre à jour ${name}`,
  updated: (name: string) =>
    `${name} mis à jour vers l’épinglage actuel du catalogue. Redémarrez le gateway pour l’appliquer.`,
  updateConsentTitle: (name: string) => `${name} demande davantage`,
  updateConsentBody: (name: string, sha: string) =>
    `Le nouvel épinglage de ${name} dans le catalogue (${sha}) ajoute des surfaces que la version installée n’a pas. Appliquez-le uniquement si vous leur faites confiance :`,
  updateConsentConfirm: 'Appliquer la mise à jour',
  updateScanTitle: (name: string) => `L’analyse de sécurité a signalé ${name}`,
  updateScanBody: (name: string, count: number) =>
    `La mise à jour de ${name} a reçu le verdict « prudence » (${count} ${count === 1 ? 'constat' : 'constats'}). Chaque mise à jour est analysée. Ne l’appliquez que si vous faites confiance à la source :`,
  updateScanMore: (count: number) => `+ ${count} de plus`,
  updateScanConfirm: 'Mettre à jour quand même',
  updateScanBlocked: (name: string) =>
    `L’analyse de sécurité a bloqué la mise à jour de ${name} (verdict « dangereux »). Impossible de passer outre.`,
  updatedGit: (name: string) => `${name} a été mis à jour vers le dernier commit.`
}
