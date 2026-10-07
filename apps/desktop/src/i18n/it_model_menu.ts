export const itModelMenu = {
  search: 'Cerca modelli',
  noModels: 'Nessun modello trovato',
  editModels: 'Modifica modelli…',
  followDefault: 'Usa il predefinito da Impostazioni',
  refreshModels: 'Aggiorna modelli',
  favorites: 'Preferiti',
  addFavorite: 'Aggiungi ai preferiti',
  removeFavorite: 'Rimuovi dai preferiti',
  favoriteShortcut: '⇧ Clic',
  fast: 'Rapido',
  free: 'gratuito',
  cacheRead: 'lettura cache',
  priceTitle: (input: string, output: string, cache: string) =>
    `Input ${input}/Mtok · Output ${output}/Mtok` + (cache ? ` · Lettura cache ${cache}/Mtok` : ''),
  limited: 'Limitato',
  limitedUntil: (time: string) => `Limitato fino alle ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} ha raggiunto il limite di utilizzo. Si azzera alle ${time}; puoi già scegliere un modello per dopo.`
      : `${provider} ha raggiunto il limite di utilizzo. Puoi già scegliere un modello per quando si azzera.`,
  modelResets: (time: string) => `si azzera alle ${time}`,
  modelLimitedTip: (time: string) =>
    `Questo modello ha raggiunto il suo limite e si azzera alle ${time}. Gli altri modelli di qui continuano a funzionare.`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `Rimane ${percent} % · si azzera ${time}` : `Rimane ${percent} %`,
  poolAccounts: (count: number) => `${count} account`,
  poolLimited: (limited: number, total: number) => `${limited}/${total} account limitati`,
  poolAccount: (number: number) => `Account ${number}`,
  poolUnknown: 'Utilizzo non disponibile',
  poolUnavailable: 'Accedi di nuovo',
  usageTip: (provider: string) => `${provider} è vicino al limite di utilizzo.`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}: rimane ${percent} %, si azzera ${time}` : `${label}: rimane ${percent} %`
}
