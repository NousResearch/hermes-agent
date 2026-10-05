export const frModelMenu = {
  search: 'Rechercher des modèles',
  noModels: 'Aucun modèle trouvé',
  editModels: 'Modifier les modèles…',
  followDefault: 'Utiliser le modèle par défaut des Réglages',
  refreshModels: 'Actualiser les modèles',
  favorites: 'Favoris',
  addFavorite: 'Ajouter aux favoris',
  removeFavorite: 'Retirer des favoris',
  favoriteShortcut: '⇧ Clic',
  fast: 'Rapide',
  free: 'gratuit',
  cacheRead: 'lecture en cache',
  priceTitle: (input: string, output: string, cache: string) =>
    `Entrée ${input}/Mtok · Sortie ${output}/Mtok` + (cache ? ` · Lecture en cache ${cache}/Mtok` : ''),
  limited: 'Limité',
  limitedUntil: (time: string) => `Limité jusqu’à ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} a atteint sa limite d’utilisation. Elle se réinitialise à ${time} ; vous pouvez déjà choisir un modèle pour après.`
      : `${provider} a atteint sa limite d’utilisation. Vous pouvez déjà choisir un modèle pour après sa réinitialisation.`,
  modelResets: (time: string) => `de retour à ${time}`,
  modelLimitedTip: (time: string) =>
    `Ce modèle a atteint sa propre limite et revient à ${time}. Les autres modèles ici fonctionnent toujours.`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `${percent} % restant · réinit. ${time}` : `${percent} % restant`,
  usageTip: (provider: string) => `${provider} approche de sa limite d’utilisation.`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label} : ${percent} % restant, réinitialisation ${time}` : `${label} : ${percent} % restant`
}
