import type { Translations } from './types'

export const frAppTour: Translations['appTour'] = {
  sessions: { title: 'Vos conversations', text: 'Chaque conversation se trouve ici. Recherchez, épinglez ou rouvrez n\' laquelle d\'entre elles.' },
  composer: { title: 'Demandez ici', text: 'Dites ce que vous voulez faire. Tapez @ pour apporter un fichier.' },
  newSession: { title: 'Commencez neuf', text: 'Une nouvelle session obtient son propre contexte. Utilisez-en une par travail.' },
  model: { title: 'Sélecteur de modèle', text: 'Choisit quel modèle répondra.' },
  modelLocal: 'Cet ordinateur peut l\'exécuter localement : Paramètres > Fournisseurs > Modèles locaux.',
  capabilities: { title: 'Capacités', text: 'Compétences, outils et extensions que Hermes peut utiliser. Ajoutez-en davantage ici.' },
  messaging: { title: 'Messagerie', text: 'Contactez Hermes depuis Telegram, Slack, Discord et plus encore.' },
  rightPane: { title: 'Le volet de travail', text: 'Ouvre les fichiers, le terminal, la revue et le navigateur intégré sur la droite.' }
}