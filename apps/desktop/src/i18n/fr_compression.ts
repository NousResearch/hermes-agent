import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const frCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: 'Moteur de contexte'
  },
  compression: {
    enabled: 'Auto-compression',
    threshold: 'Seuil de compression',
    codexGpt55Autoraise: 'Relèvement automatique de la compression Codex',
    targetRatio: 'Objectif de compression',
    protectLastN: 'Messages récents protégés',
    warmHandoff: 'Transfert à chaud'
  },
  auxiliary: {
    compression: {
      timeout: 'Délai du modèle de compression (s)'
    }
  }
}

export const frCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: 'Stratégie pour gérer les longues conversations proches de la limite de contexte.'
  },
  compression: {
    enabled: 'Résumer le contexte ancien quand les conversations grossissent.',
    codexGpt55Autoraise: 'Relever la compression à 85 % pour les modèles ChatGPT Codex OAuth pris en charge.',
    warmHandoff:
      'Le modèle principal rédige le résumé de compression sur son prompt en cache : un serveur avec cache de prompt ne lit que les nouveaux messages. Plus rapide quand la compression utilise le même modèle. Auto : seulement si la compression utilise le modèle principal et que le serveur signale des jetons en cache. Activé : toujours essayer. Désactivé : toujours utiliser le modèle de compression. En cas d’échec, le résumé normal est utilisé.'
  },
  auxiliary: {
    compression: {
      timeout:
        'Secondes d’attente du modèle de compression auxiliaire par appel (120 par défaut). Augmentez pour les modèles locaux lents.'
    }
  }
}
