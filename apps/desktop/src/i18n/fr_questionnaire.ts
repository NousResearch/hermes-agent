import type { Translations } from './types'

export const frQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'Passer la configuration',
  back: 'Retour',
  trailLabel: 'Vos réponses',
  changeAnswer: 'Modifier cette réponse',
  retry: 'Réessayer',
  saveFailed: 'Impossible de marquer la configuration comme terminée',
  skipped: 'Passé',
  otherLabel: 'Autre réponse',
  kinds: {
    accent: 'Couleur d’accent',
    layout: 'Disposition',
    local: 'Modèle local',
    apps: 'Apps sur cet appareil',
    connectors: 'Connecteurs'
  },
  name: {
    greeting: 'Bonjour, je suis Hermes.',
    question: 'Comment dois-je vous appeler ?',
    placeholder: 'Votre prénom'
  },
  accent: {
    title: 'Choisissez une couleur',
    custom: 'Couleur personnalisée'
  },
  layout: {
    title: 'Comment voulez-vous travailler ?',
    basic: 'Basique',
    basicDetail: 'Pour discuter avec Hermes.',
    elite: 'Elite',
    eliteDetail: 'Pour les développeurs : terminal, fichiers, diffs.'
  },
  local: {
    titleSpark: 'Votre Spark peut faire tourner Hermes hors ligne',
    title: kind => `Ce ${kind} peut faire tourner un modèle en local`,
    download: model => `Télécharger ${model}`,
    downloadDetail: 'Se télécharge en arrière-plan dès votre premier chat.',
    notNow: 'Pas maintenant',
    notNowDetail: 'Reste dans le menu des modèles sous « Exécuter en local ».',
    chip: 'Puce',
    gpu: 'GPU',
    memory: 'Mémoire',
    os: 'OS',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'Utiliser Hermes dans ces apps ?',
    plugin: 'Plugin',
    pluginNotRunning: 'Plugin · app non lancée',
    approve: 'Vous validez chaque installation dans votre premier chat.',
    found: 'Trouvées sur cet appareil. Les apps non installées ne sont pas listées.'
  },
  connectors: {
    title: 'Connecter vos apps ?',
    checking: 'Vérification des apps que votre compte peut connecter…',
    offered:
      'Voici la liste proposée par votre compte en ce moment. Vous vous connectez à chacune dans votre premier chat.'
  },
  task: {
    title: 'Par quoi commencer ?',
    question: 'Choisissez-en une ou écrivez la vôtre.',
    blender: 'Créer une scène dans Blender',
    nvidia: 'Optimiser mon GPU',
    broadcast: 'Régler mon micro et ma caméra',
    brief: 'Le point sur ma journée',
    tidy: 'Ranger mes Téléchargements'
  },
  tour: {
    title: 'Une visite guidée ?',
    question: 'Un petit tour ensuite ?',
    quick: 'Faites-moi visiter',
    quickDetail: 'Quelques étapes, moins d’une minute.',
    none: 'Je me débrouille',
    noneDetail: 'Vous pourrez demander une visite à Hermes plus tard.'
  },
  start: {
    title: name => (name ? `Prêt, ${name}.` : 'Prêt.'),
    sub: 'C’est le premier message que reçoit Hermes. La demande vient en premier ; le reste suit en dessous.',
    aboutMe: 'À propos de moi :',
    beforeYouStart: 'Avant de commencer :',
    noTask: 'Aucune tâche choisie : Hermes vous demandera ce que vous voulez faire.',
    action: 'Commencer',
    waiting: 'Préparation de votre compte gratuit…',
    failed: 'Impossible de lancer votre premier chat'
  },
  summary: {
    custom: 'Perso',
    local: model => `Local : ${model}`,
    localNo: 'Local : plus tard',
    noApps: 'Aucune app',
    noConnectors: 'Aucun connecteur',
    more: (label, count) => `${label} +${count}`,
    tour: 'Visite',
    noTour: 'Sans visite'
  },
  status: {
    settingUp: 'Configuration de votre compte gratuit',
    stillSettingUp: 'Configuration de votre compte gratuit en cours',
    unavailable: 'Compte gratuit indisponible',
    ready: 'Nous · offre gratuite',
    downloadingEngine: 'Téléchargement du moteur local',
    downloading: model => `Téléchargement de ${model}`
  },
  settings: {
    title: 'Relancer la configuration',
    description: 'Rouvre les questions du premier lancement. Vos réponses s’appliquent au profil par défaut.',
    action: 'Configurer',
    failed: 'Impossible de lancer la configuration'
  }
}
