import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into fr.ts.
export const frSimpleMode = {
  interfaceMode: {
    title: 'Mode d’interface',
    hint: 'Modifie ce qui est affiché, pas ce que Hermes peut faire.',
    sessionNote:
      'Défini par le mode Simple. Une modification ici dure le temps de cette session ; passez en mode Avancé pour la conserver.',
    switchToAdvanced: 'Passer en mode Avancé',
    simple: {
      label: 'Simple',
      description:
        'Pour discuter avec Hermes. Barre latérale et conversation ; pas de panneaux terminal, fichiers ou diff.'
    },
    advanced: {
      label: 'Avancé',
      description: 'Pour les développeurs. Terminal, fichiers, diffs, barre d’état et dispositions, selon vos réglages.'
    }
  },
  chatStarters: {
    label: 'Des idées pour commencer',
    intro: 'Demandez-moi ce que vous voulez, ou commencez par une de ces idées.',
    email: {
      title: 'Aide-moi à écrire un e-mail',
      prompt: 'Aide-moi à écrire un e-mail. Voici à qui il est destiné et ce que je veux dire : '
    },
    dinners: {
      title: 'Prévoir les dîners de la semaine',
      prompt: 'Propose cinq dîners faciles pour cette semaine et donne-moi une seule liste de courses pour tous.'
    },
    summarize: {
      title: 'Résumer un document',
      prompt: "Je vais joindre un fichier — donne-m'en les points essentiels."
    },
    lookup: {
      title: 'Faire une recherche pour moi',
      prompt: 'Cherche ceci pour moi et dis-moi ce que tu trouves : '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
