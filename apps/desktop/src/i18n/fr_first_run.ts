import type { FirstRunCopy } from './types_first_run'

export const frFirstRun: FirstRunCopy = {
  guide: {
    title: 'Configurer Hermes',
    resume: 'Reprendre la configuration',
    upgradeRequired:
      'Hermes sur cet ordinateur doit être mis à jour pour la configuration sans session. Sur cet ordinateur, exécutez hermes update, redémarrez Hermes, puis réessayez. Vos conversations restent inchangées.',
    intro:
      'Quelques préférences facultatives, puis votre chat habituel. La configuration ne crée jamais de conversation.',
    appearance: 'Apparence',
    model: 'Choisir un modèle',
    ready: 'Votre modèle est prêt.',
    providerNeeded: 'Connectez un fournisseur ou utilisez le modèle gratuit avant d’envoyer un message.',
    freeModel: 'Utiliser le modèle gratuit',
    freeUnavailable:
      'Le modèle gratuit n’est pas disponible ici. Connectez un fournisseur ou choisissez-en un plus tard.',
    providers: 'Connecter un fournisseur',
    finish: 'Terminer la configuration',
    skip: 'Ignorer la configuration',
    back: 'Retour',
    check: 'Vérifier la disponibilité',
    checking: 'Vérification…',
    privacy: 'Le partage des statistiques d’utilisation est facultatif.'
  },
  beats: {
    name: 'Nom',
    theme: "Couleur d'accent",
    layout: 'Disposition',
    connectors: 'Applications',
    plugins: 'Plugins',
    tour: 'Visite',
    build: 'Première tâche'
  },
  namePlaceholder: 'Votre nom',
  useName: name => `Utiliser « ${name} »`,
  install: 'Installer',
  installed: 'Installé',
  loading: 'Chargement…',
  noPlugins: 'Aucun plugin à afficher pour le moment.',
  noneChosen: 'Aucun',
  noAnswer: 'Pas de réponse',
  answerRefused: 'Réponse non acceptée',
  tours: {
    quick: 'Visite rapide',
    full: 'Montre-moi tout',
    skip: 'Passer, construisons quelque chose'
  },
  skippedAll: 'Configuration ignorée',
  skip: 'Ignorer la configuration',
  skipTip: 'Termine la configuration guidée et ouvre une nouvelle discussion.',
  skipFailed: 'Impossible d’ignorer la configuration',
  slashOffTitle: 'Les commandes slash sont désactivées pendant la configuration',
  slashOffBody: 'Elles fonctionneront normalement une fois la configuration terminée.',
  localModel: {
    title: 'Cet ordinateur peut exécuter un modèle en local',
    text: (model, size) =>
      `${model} (${size}) convient à votre matériel. Il fonctionne gratuitement et vos conversations restent sur cet ordinateur.`,
    setUp: 'Le configurer',
    notNow: 'Pas maintenant'
  },
  intro: {
    greeting: 'Bonjour, je suis votre agent.',
    plan: 'Configurons tout pour vous. Ensuite, nous ferons quelque chose de génial.'
  },
  rerun: {
    title: "Relancer l'accueil",
    description: 'Ouvre un nouveau chat de configuration qui reprend les premières étapes avec votre modèle actuel.',
    action: 'Relancer',
    failed: "Impossible de lancer l'accueil"
  }
}
