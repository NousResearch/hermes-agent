import type { Translations } from './types'

export const frAppTour: Translations['appTour'] = {
  sessions: {
    title: 'Vos conversations',
    text: 'Toutes les conversations se trouvent ici. Recherchez, épinglez ou rouvrez-en une.'
  },
  composer: { title: 'Demandez ici', text: 'Dites ce que vous voulez faire. Tapez @ pour importer un fichier.' },
  newSession: {
    title: 'Repartez de zéro',
    text: 'Une nouvelle session dispose de son propre contexte. Utilisez-en une par tâche.'
  },
  model: { title: 'Sélecteur de modèle', text: 'Choisit le modèle qui vous répond.' },
  modelLocal: 'Cet ordinateur peut en exécuter un localement : Paramètres > Fournisseurs > Modèles locaux.',
  capabilities: { title: 'Skills', text: 'Skills, outils et plugins que Hermes peut utiliser. Ajoutez-en ici.' },
  messaging: { title: 'Messagerie', text: 'Joignez Hermes depuis Telegram, Slack, Discord et plus encore.' },
  rightPane: {
    title: 'Le volet de travail',
    text: 'Ouvre les fichiers, le terminal, la revue et le navigateur intégré à droite.'
  }
}
