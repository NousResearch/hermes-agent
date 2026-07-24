// French (fr) labels and descriptions for the settings fields.
// Mirrors FIELD_LABELS / FIELD_DESCRIPTIONS (constants.ts); used by the `fr` locale.
// Contributed by William MERI.

import { defineFieldCopy } from './field-copy'

export const FIELD_LABELS_FR: Record<string, string> = defineFieldCopy({
  model: 'Modèle par défaut',
  modelContextLength: 'Fenêtre de contexte',
  fallbackProviders: 'Modèles de secours',
  toolsets: "Jeux d'outils activés",
  timezone: 'Fuseau horaire',
  display: {
    personality: 'Personnalité',
    showReasoning: 'Blocs de raisonnement',
  },
  agent: {
    maxTurns: "Étapes max de l'agent",
    imageInputMode: 'Pièces jointes image',
    apiMaxRetries: 'Tentatives API',
    serviceTier: 'Niveau de service',
    toolUseEnforcement: "Contrôle d'usage des outils",
  },
  terminal: {
    cwd: 'Répertoire de travail',
    backend: "Backend d'exécution",
    timeout: 'Délai de commande',
    persistentShell: 'Shell persistant',
    envPassthrough: "Variables d'environnement",
    dockerImage: 'Image Docker',
    singularityImage: 'Image Singularity',
    modalImage: 'Image Modal',
    daytonaImage: 'Image Daytona',
  },
  fileReadMaxChars: 'Limite de lecture de fichier',
  toolOutput: {
    maxBytes: 'Limite de sortie terminal',
    maxLines: 'Limite de pages de fichier',
    maxLineLength: 'Limite de longueur de ligne',
  },
  codeExecution: {
    mode: "Mode d'exécution du code",
  },
  approvals: {
    mode: "Mode d'approbation",
    timeout: "Délai d'approbation",
    mcpReloadConfirm: 'Confirmer les rechargements MCP',
  },
  commandAllowlist: 'Liste de commandes autorisées',
  security: {
    redactSecrets: 'Masquer les secrets',
    allowPrivateUrls: 'Autoriser les URL privées',
  },
  browser: {
    allowPrivateUrls: 'URL privées du navigateur',
    autoLocalForPrivateUrls: 'Navigateur local pour URL privées',
  },
  checkpoints: {
    enabled: 'Points de contrôle de fichiers',
    maxSnapshots: 'Limite de points de contrôle',
  },
  voice: {
    recordKey: 'Raccourci vocal',
    maxRecordingSeconds: "Durée max d'enregistrement",
    autoTts: 'Lire les réponses à voix haute',
  },
  stt: {
    enabled: 'Reconnaissance vocale',
    provider: 'Fournisseur de reconnaissance vocale',
    local: {
      model: 'Modèle de transcription local',
      language: 'Langue de transcription',
    },
    openai: {
      model: 'Modèle STT OpenAI',
    },
    groq: {
      model: 'Modèle STT Groq',
    },
    mistral: {
      model: 'Modèle STT Mistral',
    },
    elevenlabs: {
      modelId: 'Modèle STT ElevenLabs',
      languageCode: 'Langue ElevenLabs',
      tagAudioEvents: 'Marquer les événements audio',
      diarize: 'Distinction des locuteurs',
    },
  },
  tts: {
    provider: 'Fournisseur de synthèse vocale',
    edge: {
      voice: 'Voix Edge',
    },
    openai: {
      model: 'Modèle TTS OpenAI',
      voice: 'Voix OpenAI',
    },
    elevenlabs: {
      voiceId: 'Voix ElevenLabs',
      modelId: 'Modèle ElevenLabs',
    },
    xai: {
      voiceId: 'Voix xAI (Grok)',
      language: 'Langue xAI',
    },
    minimax: {
      model: 'Modèle TTS MiniMax',
      voiceId: 'Voix MiniMax',
    },
    mistral: {
      model: 'Modèle TTS Mistral',
      voiceId: 'Voix Mistral',
    },
    gemini: {
      model: 'Modèle TTS Gemini',
      voice: 'Voix Gemini',
    },
    neutts: {
      model: 'Modèle NeuTTS',
      device: 'Périphérique NeuTTS',
    },
    kittentts: {
      model: 'Modèle KittenTTS',
      voice: 'Voix KittenTTS',
    },
    piper: {
      voice: 'Voix Piper',
    },
  },
  memory: {
    memoryEnabled: 'Mémoire persistante',
    userProfileEnabled: 'Profil utilisateur',
    memoryCharLimit: 'Budget mémoire',
    userCharLimit: 'Budget profil',
    provider: 'Fournisseur de mémoire',
  },
  context: {
    engine: 'Moteur de contexte',
  },
  compression: {
    enabled: 'Compression automatique',
    threshold: 'Seuil de compression',
    targetRatio: 'Cible de compression',
    protectLastN: 'Messages récents protégés',
  },
  delegation: {
    model: 'Modèle de sous-agent',
    provider: 'Fournisseur de sous-agent',
    maxIterations: 'Limite de tours de sous-agent',
    maxConcurrentChildren: 'Sous-agents parallèles',
    childTimeoutSeconds: 'Délai de sous-agent',
    reasoningEffort: 'Effort de raisonnement du sous-agent',
  },
  updates: {
    nonInteractiveLocalChanges: 'Modifications locales lors des MAJ in-app',
  },
})

export const FIELD_DESCRIPTIONS_FR: Record<string, string> = defineFieldCopy({
  model: 'Utilisé pour les nouvelles conversations, sauf si vous choisissez un autre modèle dans le composeur.',
  modelContextLength: 'Laissez à 0 pour utiliser la fenêtre de contexte détectée du modèle sélectionné.',
  fallbackProviders: 'Entrées fournisseur:modèle de secours à essayer si le modèle par défaut échoue.',
  display: {
    personality: "Style par défaut de l'assistant pour les nouvelles sessions.",
    showReasoning: 'Afficher les sections de raisonnement quand le backend les fournit.',
  },
  timezone: 'Utilisé quand Hermes a besoin du contexte horaire local. Vide = fuseau du système.',
  agent: {
    imageInputMode: 'Contrôle la façon dont les pièces jointes image sont envoyées au modèle.',
    maxTurns: "Limite supérieure de tours d'appel d'outils avant que Hermes n'arrête une exécution.",
  },
  terminal: {
    cwd: 'Dossier de projet par défaut pour les outils et le terminal.',
    persistentShell: "Conserver l'état du shell entre les commandes quand le backend le permet.",
    envPassthrough: "Variables d'environnement à transmettre à l'exécution des outils.",
    dockerImage: "Image de conteneur utilisée quand le backend d'exécution est Docker.",
    singularityImage: "Image utilisée quand le backend d'exécution est Singularity.",
    modalImage: "Image utilisée quand le backend d'exécution est Modal.",
    daytonaImage: "Image utilisée quand le backend d'exécution est Daytona.",
  },
  codeExecution: {
    mode: "Niveau de restriction de l'exécution du code au projet courant.",
  },
  fileReadMaxChars: 'Nombre maximal de caractères que Hermes peut lire en une seule requête de fichier.',
  approvals: {
    mode: 'Comment Hermes gère les commandes nécessitant une approbation explicite.',
    timeout: "Durée d'attente des invites d'approbation avant expiration.",
  },
  security: {
    redactSecrets: 'Masquer les secrets détectés du contenu visible par le modèle si possible.',
  },
  checkpoints: {
    enabled: 'Créer des instantanés de restauration avant les modifications de fichiers.',
  },
  memory: {
    memoryEnabled: 'Enregistrer des mémoires durables utiles aux futures sessions.',
    userProfileEnabled: 'Maintenir un profil compact des préférences utilisateur.',
  },
  context: {
    engine: 'Stratégie de gestion des longues conversations proches de la limite de contexte.',
  },
  compression: {
    enabled: 'Résumer le contexte ancien quand les conversations deviennent volumineuses.',
  },
  voice: {
    autoTts: "Énoncer automatiquement les réponses de l'assistant.",
  },
  tts: {
    xai: {
      voiceId: 'ID de voix xAI (ex. eve) ou un ID de voix personnalisé.',
      language: 'Code de langue parlée, ex. en.',
    },
    neutts: {
      device: "Périphérique d'inférence local pour NeuTTS.",
    },
  },
  stt: {
    enabled: 'Activer la transcription vocale locale ou via un fournisseur.',
    elevenlabs: {
      languageCode: 'Code de langue ISO-639-3 optionnel. Vide : ElevenLabs détecte automatiquement.',
    },
  },
  updates: {
    nonInteractiveLocalChanges:
      "Quand Hermes se met à jour depuis l'app (sans invite terminal), conserver les modifications source locales (stash) ou les jeter (discard). Les mises à jour en terminal demandent toujours.",
  },
})
