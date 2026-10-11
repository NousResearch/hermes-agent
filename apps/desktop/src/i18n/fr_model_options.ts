import type { Translations } from './types'

export const frModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'Aucune option pour ce modèle',
  options: 'Options',
  thinking: 'Réflexion',
  fast: 'Rapide',
  ultrafast: 'Ultrafast',
  useStandardSpeed: 'Utiliser la vitesse standard',
  auto: 'Auto',
  cold: 'Froid',
  speedPolicy: 'Politique de vitesse',
  effort: 'Effort',
  minimal: 'Minimal',
  low: 'Faible',
  medium: 'Moyen',
  high: 'Élevé',
  xhigh: 'Très élevé',
  max: 'Max',
  ultra: 'Ultra',
  sendsOnRoute: (level: string) => `envoie ${level} sur cette route`,
  updateFailed: "Échec de la mise à jour de l'option du modèle",
  fastFailed: 'Échec de la mise à jour du mode rapide'
}
