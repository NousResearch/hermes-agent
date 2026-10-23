import type { TranslationOverride } from '@hermes/shared/i18n'

import { frAuxTasks } from './fr_aux_tasks'
import { frDelegationRouting } from './fr_delegation_routing'
import type { ModelSettingsTranslations } from './types_model_settings'

export const frModelSettings: TranslationOverride<ModelSettingsTranslations> = {
  setupProviderFallback: 'fournisseur',
  setUpProvider: name => `Configurer ${name}`,
  staleAuxBefore: (count, names) =>
    count > 1
      ? `${count} tâches auxiliaires (${names}) s'exécutent encore sur `
      : `${count} tâche auxiliaire (${names}) s'exécute encore sur `,
  staleAuxAfter: ', et non sur votre modèle principal.',
  staleAuxOtherProviders: "d'autres fournisseurs",
  moaEnabled: 'Activé',
  moaSetDefault: 'Définir par défaut',
  moaNewPresetPlaceholder: 'nouveau préréglage',
  moaAddPreset: 'Ajouter un préréglage',
  customModel: 'Modèle personnalisé…',
  customModelPlaceholder: 'ID du modèle',
  chooseFromList: 'Choisir dans la liste',
  moaDefault: 'Par défaut :',
  moaReferenceToggle: (enabled, index) => `${enabled ? 'Désactiver' : 'Activer'} la référence ${index}`,
  moaReferenceTitle: index => `Référence ${index}`,
  moaAddReference: 'Ajouter un modèle de référence',
  loading: 'Chargement de la configuration du modèle...',
  appliesDesc:
    "S'applique aux nouvelles sessions. Utilisez le sélecteur de modèle dans le compositeur pour changer à chaud la conversation active.",
  provider: 'Fournisseur',
  model: 'Modèle',
  applying: 'Application...',
  mainAppliedTitle: 'Modèle principal mis à jour',
  mainAppliedMessage: model => `Les nouvelles sessions utiliseront ${model}.`,
  defaultsLabel: 'Par défaut',
  reasoning: 'Raisonnement',
  reasoningOff: 'Désactivé',
  speed: 'Vitesse',
  speedStandard: 'Standard',
  defaultsFailed: "Échec de l'enregistrement des modèles par défaut",
  loadFailed: 'Impossible de charger les modèles',
  restartRequired:
    "Ce backend exécute encore l'ancien code après une mise à jour. Redémarrez-le pour charger le nouveau code.",
  restartBackend: 'Redémarrer le backend',
  restartingBackend: 'Redémarrage du backend...',
  restartFailed: 'Impossible de redémarrer le backend',
  auxiliaryTitle: 'Modèles auxiliaires',
  resetAllToMain: 'Tout réinitialiser au principal',
  staleAuxDismiss: 'Ne plus afficher',
  auxiliaryDesc:
    "Les tâches d'assistance s'exécutent sur le modèle principal par défaut. Attribuez un modèle dédié à toute tâche pour remplacer.",
  setToMain: 'Définir comme principal',
  change: 'Modifier',
  autoUseMain: 'auto · utiliser le modèle principal',
  inheritMainEffort: 'hériter · effort du modèle principal',
  inheritsFrom: task => `hérite de ${task}`,
  followTask: task => `Suivre ${task}`,
  providerDefault: '(par défaut du fournisseur)',
  fallbackAdd: 'Ajouter un secours',
  fallbackEmpty: "Aucun modèle de secours — le modèle par défaut est utilisé sauf en cas d'échec.",
  notInCatalog: "n'est pas dans la liste des modèles de ce fournisseur — les appels peuvent basculer sur un secours.",
  moaTitle: "Mélange d'agents",
  moaPreset: 'Préréglage',
  moaDescription:
    "Configurez des préréglages nommés qui apparaissent comme modèles chez le fournisseur Mélange d'agents. L'agrégateur est le modèle actif : il exécute chaque étape de la boucle d'outils et son fournisseur facture presque tout le coût de l'exécution. Par défaut, les modèles de référence ne donnent qu'un avis par tour utilisateur.",
  moaAggregator: 'Agrégateur',
  moaAggregatorBilled: "modèle actif · facturé pour l'exécution",
  moaReferenceHint: 'donne un avis une fois par tour par défaut',
  tasks: frAuxTasks,
  subagents: {
    inherit: "Hériter de l'agent principal",
    modelOnly: "Modèle personnalisé (identifiants de l'agent principal)",
    providerLabel: 'Fournisseur des sous-agents',
    modelLabel: 'Modèle des sous-agents',
    direct: 'Point de terminaison direct',
    customProvider: 'Fournisseur personnalisé...',
    customProviderLabel: 'Fournisseur personnalisé des sous-agents',
    parentModel: "Utiliser le modèle de l'agent principal",
    parentModelWarning:
      "Le fournisseur choisi doit prendre en charge le modèle de l'agent principal. Une route limitée au fournisseur peut échouer si le modèle est incompatible. Si ce fournisseur définit son propre modèle par défaut, les sous-agents utilisent celui-ci.",
    directActive:
      'Un point de terminaison direct est actif et prime sur le fournisseur enregistré. Son adresse et ses identifiants sont conservés tant que ce mode est sélectionné.',
    directClear:
      'Appliquer cette route efface le point de terminaison direct et sa clé enregistrée. Les autres réglages de requête sont conservés.',
    catalogFailed:
      'Catalogue de modèles indisponible. Saisissez un fournisseur et un modèle manuellement, ou réessayez.',
    draftHint: "Les modifications restent locales jusqu'à leur application.",
    routing: frDelegationRouting
  }
}
