export const esModelMenu = {
  search: 'Buscar modelos',
  noModels: 'No se encontraron modelos',
  editModels: 'Editar modelos…',
  followDefault: 'Usar el predeterminado de Ajustes',
  refreshModels: 'Actualizar modelos',
  favorites: 'Favoritos',
  addFavorite: 'Añadir a favoritos',
  removeFavorite: 'Quitar de favoritos',
  favoriteShortcut: '⇧ Clic',
  fast: 'Rápido',
  free: 'gratis',
  cacheRead: 'lectura en caché',
  priceTitle: (input: string, output: string, cache: string) =>
    `Entrada ${input}/Mtok · Salida ${output}/Mtok` + (cache ? ` · Lectura en caché ${cache}/Mtok` : ''),
  limited: 'Limitado',
  limitedUntil: (time: string) => `Limitado hasta las ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} alcanzó su límite de uso. Se restablece a las ${time}; ya puedes elegir un modelo para después.`
      : `${provider} alcanzó su límite de uso. Ya puedes elegir un modelo para cuando se restablezca.`,
  modelResets: (time: string) => `vuelve a las ${time}`,
  modelLimitedTip: (time: string) =>
    `Este modelo alcanzó su propio límite y vuelve a las ${time}. Los demás modelos de aquí siguen funcionando.`
}
