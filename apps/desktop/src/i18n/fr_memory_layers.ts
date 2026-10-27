import type { MemoryLayersTranslations } from './types_memory_layers'

export const frMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Fournisseur de mémoire actif',
  noProvider: 'Aucun — mémoire intégrée uniquement',
  noProviderDesc:
    'Hermes utilise uniquement la couche intégrée ci-dessous. Choisissez un fournisseur externe dans Fournisseur de mémoire pour en ajouter un.',
  providerDesc: name => `${name} ajoute son propre rappel à chaque session, en plus de la couche intégrée.`,
  status: {
    ready: 'prêt',
    needs_config: 'configuration requise',
    unavailable: 'indisponible',
    missing: 'non installé'
  },
  builtinLayer: 'Couche intégrée',
  alwaysOn: 'Toujours active',
  builtinDesc:
    "MEMORY.md et USER.md sont chargés dans chaque session, avec ou sans fournisseur externe. Un fournisseur s'ajoute à cette couche ; il ne la remplace jamais."
}
