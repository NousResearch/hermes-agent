import type { FirstRunCopy } from './types_first_run'

export const itFirstRun: FirstRunCopy = {
  beats: {
    name: 'Nome',
    theme: 'Colore d’accento',
    layout: 'Layout',
    connectors: 'App',
    plugins: 'Plugin',
    tour: 'Tour',
    build: 'Prima attività'
  },
  namePlaceholder: 'Il tuo nome',
  useName: name => `Usa “${name}”`,
  install: 'Installa',
  installed: 'Installato',
  loading: 'Caricamento…',
  noPlugins: 'Al momento non ci sono plugin da mostrare.',
  noneChosen: 'Nessuno',
  noAnswer: 'Senza risposta',
  answerRefused: 'Risposta non accettata',
  tours: {
    quick: 'Tour rapido',
    full: 'Mostrami tutto',
    skip: 'Salta, costruiamo qualcosa'
  },
  skippedAll: 'Configurazione saltata',
  skip: 'Salta la configurazione',
  skipTip: 'Chiude la configurazione guidata e apre una nuova chat.',
  skipFailed: 'Impossibile saltare la configurazione',
  localModel: {
    title: 'Questo computer può eseguire un modello in locale',
    text: (model, size) =>
      `${model} (${size}) è adatto al tuo hardware. È gratuito e le chat restano su questo computer.`,
    setUp: 'Configuralo',
    notNow: 'Non ora'
  },
  intro: {
    greeting: 'Ciao, sono Hermes.',
    plan: 'Prepariamo tutto per te. Poi facciamo qualcosa di bello.'
  },
  rerun: {
    title: 'Riesegui la configurazione iniziale',
    description:
      'Apre una nuova chat di configurazione che ripercorre i passaggi del primo avvio con il modello attuale.',
    action: 'Esegui di nuovo',
    failed: 'Impossibile avviare la configurazione'
  }
}
