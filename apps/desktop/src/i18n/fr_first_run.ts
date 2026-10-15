import type { FirstRunCopy } from './types_first_run'

export const frFirstRun: FirstRunCopy = {
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
  localModel: {
    title: 'Cet ordinateur peut exécuter un modèle en local',
    text: (model, size) =>
      `${model} (${size}) convient à votre matériel. Il fonctionne gratuitement et vos conversations restent sur cet ordinateur.`,
    setUp: 'Le configurer',
    notNow: 'Pas maintenant'
  },
  intro: {
    greeting: 'Bonjour, je suis Hermes.',
    plan: 'Configurons tout pour vous. Ensuite, nous ferons quelque chose de génial.'
  },
  rerun: {
    title: "Relancer l'accueil",
    description: 'Ouvre un nouveau chat de configuration qui reprend les premières étapes avec votre modèle actuel.',
    action: 'Relancer',
    failed: "Impossible de lancer l'accueil"
  }
}
