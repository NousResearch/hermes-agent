import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const frDelegationRouting: DelegationRoutingTranslations = {
  title: 'Routage',
  settingsTitle: 'Paramètres par défaut des sous-agents',
  intro: 'Comment chaque sous-agent obtient son modèle.',
  modes: {
    default: { label: 'Par défaut', description: 'Chaque sous-agent utilise le modèle sous-agent ci-dessous.' },
    model_map: {
      label: 'Routes de modèle',
      description: "Nommez quelques modèles. L'agent en choisit un pour chaque sous-agent."
    },
    profile_routes: {
      label: 'Routes d’agent',
      description:
        'Les sous-agents utilisent l’un de vos agents, avec son propre modèle, ses skills et ses clés.'
    }
  },
  inUse: 'Utilisé',
  plugin: 'Plugin',
  pluginMissing: 'Routes de modèle et Routes d’agent proviennent du plugin Advanced Delegation.',
  pluginDisabled: 'Advanced Delegation est installé mais désactivé pour cet agent.',
  pluginNotLoaded:
    "Advanced Delegation est activé, mais Hermes ne l'a pas encore chargé. Redémarrez Hermes pour terminer.",
  installPlugin: 'Installer le plugin',
  installUnavailable:
    "Advanced Delegation n'est pas encore dans le catalogue de plugins. Installez-le depuis Capacités › Plugins.",
  enablePlugin: 'Activer',
  enableFailed: "Impossible d'activer Advanced Delegation",
  modeFailed: 'Impossible de modifier le routage des sous-agents',
  loadFailed: 'Impossible de charger le routage des sous-agents',
  blockedPill: 'Inactif',
  blockedDetail:
    'Ce mode est enregistré : Hermes refuse donc chaque sous-agent au lieu de passer discrètement à Par défaut.',
  blockedMissing: 'Les sous-agents ne démarreront pas tant qu’Advanced Delegation n’est pas installé.',
  blockedDisabled: 'Les sous-agents ne démarreront pas tant qu’Advanced Delegation n’est pas activé pour cet agent.',
  blockedNotLoaded:
    'Les sous-agents ne démarreront pas tant que Hermes n’a pas chargé Advanced Delegation. Redémarrez Hermes pour terminer.',
  blockedUnavailable: label =>
    `Les sous-agents ne démarreront pas tant que ${label} ne peut pas fonctionner ici. Son plugin indique qu’il ne le peut pas.`,
  blockedUnregistered: name =>
    `Les sous-agents ne démarreront pas tant qu’un plugin fournissant « ${name} » n’est pas installé et activé.`,
  thirdPartyDescription: 'Fourni par un plugin. Ses réglages se trouvent dans ce plugin.',

  modelRoutesTitle: 'Routes de modèle',
  modelRoutesEmpty:
    'Aucune route pour l’instant. Tant que vous n’en ajoutez pas, les sous-agents utilisent le modèle de l’agent principal.',
  addRoute: 'Ajouter une route',
  editRoute: 'Modifier',
  makeDefault: 'Définir par défaut',
  deleteRoute: 'Supprimer',
  defaultBadge: 'Par défaut',
  routeActions: name => `Actions pour ${name}`,
  deleteConfirmTitle: name => `Supprimer la route « ${name} » ?`,
  deleteConfirmBody: 'Les sous-agents ne pourront plus la choisir.',
  deleteConfirmNextDefault: name => `C'est la route par défaut : « ${name} » devient donc la route par défaut.`,
  deleteConfirmAction: 'Supprimer la route',
  saveFailed: "Impossible d'enregistrer les routes",
  baseUrlRoute: url => `point de terminaison ${url}`,

  dialogNew: 'Nouvelle route de modèle',
  dialogEdit: 'Modifier la route de modèle',
  dialogDesc: "L'agent lit le nom et la description pour décider quelle route prend un sous-agent.",
  nameLabel: 'Nom',
  namePlaceholder: 'ex. fast',
  nameHint: "Court, sans espaces. L'agent choisit les routes par ce nom.",
  descriptionLabel: "Quand l'utiliser",
  descriptionPlaceholder: 'ex. Économique et rapide : recherches, résumés, modifications simples.',
  descriptionHint: "Une ligne que l'agent lit avant de choisir.",
  providerLabel: 'Fournisseur',
  providerPlaceholder: 'Choisir un fournisseur',
  modelLabel: 'Modèle',
  effortLabel: 'Intensité du raisonnement',
  effortInherit: 'Valeur par défaut des sous-agents',
  tierLabel: 'Vitesse',
  tierNormal: 'Normale',
  tierFast: 'Rapide',
  tierUltrafast: 'Ultra-rapide',
  tierHint: 'Uniquement sur les modèles qui la proposent.',
  daybreakLabel: 'Daybreak',
  daybreakOff: 'Désactivé',
  daybreakOn: 'Activé',
  daybreakHint: 'Accès de confiance OpenAI avec moins de garde-fous, pour cette route uniquement.',
  daybreakUnlisted:
    'Ce modèle n’est pas listé pour Daybreak sur votre compte. Désactivez-le ou choisissez un autre modèle.',
  daybreakNeedsPlugin:
    'Votre plugin Advanced Delegation ne gère pas encore les routes Daybreak. Mettez-le à jour ou désactivez ceci.',
  optional: 'facultatif',
  errors: { required: 'Obligatoire', spaces: "Pas d'espaces", duplicate: 'Une autre route utilise déjà ce nom' },
  saveRoute: 'Enregistrer la route',
  createRoute: 'Ajouter une route',

  profileRoutesTitle: 'Agents disponibles pour les sous-agents',
  profileRoutesHint: 'Chaque agent coché apporte son propre modèle, ses skills, ses instructions et ses clés.',
  profileDescriptionPlaceholder: "Quand l'utiliser (facultatif)",
  profileDefault: 'Par défaut',
  profileEmpty: 'Aucun autre agent pour l’instant. Créez-en un depuis la barre des agents.',
  profileMissing: name => `${name} (l’agent n’existe plus)`,

  inactiveTitle: 'Inutilisé dans ce mode',
  inactiveModelMap: route =>
    `Les routes de modèle choisissent le modèle de chaque sous-agent. Les sous-agents qui ne choisissent aucune route utilisent « ${route} ».`,
  inactiveProfiles: profile =>
    `Les routes d’agent choisissent l’agent utilisé par chaque sous-agent. Les sous-agents sans choix explicite utilisent « ${profile} ».`,
  inactiveModelMapNoDefault:
    'Les routes de modèle choisissent le modèle de chaque sous-agent. Les sous-agents sans route utilisent le modèle de l’agent principal.',
  inactiveProfilesNoDefault:
    'Les routes d’agent choisissent l’agent utilisé par chaque sous-agent. Sans choix explicite, les sous-agents utilisent le modèle de l’agent principal.',
  inactivePlugin: label => `${label} choisit le modèle de chaque sous-agent, ce réglage ne s’applique donc pas.`,
  useDefault: 'Passer à Par défaut'
}
