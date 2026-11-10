import type { TranslationOverrides } from './define-locale'

export const frOnboarding: TranslationOverrides['onboarding'] = {
  headerTitle: 'Configurons Hermes Agent pour vous',
  headerDesc:
    'Connectez un fournisseur de modèles pour commencer à discuter. La plupart des options nécessitent un clic.',
  preparingInstall: "Hermes finalise l'installation. Cela prend généralement moins d'une minute au premier lancement.",
  starting: 'Démarrage de Hermes…',
  setupSlowTitle: 'La configuration prend plus de temps que d’habitude.',
  setupSlowBody: 'Hermes est toujours en cours de démarrage en arrière-plan.',
  continueWithoutSetup: 'Continuer sans configuration',
  lookingUpProviders: 'Recherche des fournisseurs...',
  collapse: 'Réduire',
  otherProviders: 'Autres fournisseurs',
  haveApiKey: 'Vous avez une clé API ?',
  chooseLater: 'Je choisirai un fournisseur plus tard',
  recommended: 'Recommandé',
  connected: 'Connecté',
  featuredPitch: 'Un abonnement, 300+ modèles de pointe — la méthode recommandée pour exécuter Hermes',
  fireworksPitch: 'API de modèles directe — modèles de pointe hébergés par Fireworks',
  localModelsTitle: 'Exécuter des modèles en local',
  localModelsPitch: 'Aucun compte requis — téléchargez un modèle et exécutez-le sur cette machine',
  openRouterPitch: 'Une clé, des centaines de modèles — une valeur par défaut solide',
  apiKeyOptions: {
    fireworks: {
      short: 'API de modèles directe',
      description: 'Accès direct aux modèles hébergés par Fireworks AI.'
    },
    openrouter: {
      short: 'une clé, de nombreux modèles',
      description:
        'Héberge des centaines de modèles derrière une seule clé. Bonne valeur par défaut pour les nouvelles installations.'
    },
    openai: {
      short: 'Modèles de classe GPT',
      description: 'Accès direct aux modèles OpenAI.'
    },
    gemini: {
      short: 'Modèles Gemini',
      description: 'Accès direct aux modèles Google Gemini.'
    },
    xai: {
      short: 'Modèles Grok',
      description: 'Accès direct aux modèles xAI Grok.'
    },
    local: {
      short: 'auto-hébergé',
      description:
        'Pointez Hermes vers un point de terminaison local ou auto-hébergé compatible OpenAI (vLLM, llama.cpp, Ollama, etc).'
    }
  },
  backToSignIn: 'Retour à la connexion',
  getKey: 'Obtenir une clé',
  replaceCurrent: 'Remplacer la valeur actuelle',
  pasteApiKey: 'Collez votre clé API',
  localApiKeyPlaceholder: 'Clé API (facultatif — uniquement si votre point de terminaison en requiert une)',
  localModelNamePlaceholder: 'Nom du modèle (ex. command-a-plus-05-2026)',
  couldNotSave: "Impossible d'enregistrer l'identifiant.",
  connecting: 'Connexion',
  update: 'Mettre à jour',
  flowSubtitles: {
    pkce: 'Ouvre votre navigateur pour vous connecter, puis continue ici',
    device_code: 'Ouvre une page de vérification dans votre navigateur — Hermes se connecte automatiquement',
    external: 'Connectez-vous une fois dans votre terminal, puis revenez discuter'
  },
  startingSignIn: provider => `Démarrage de la connexion pour ${provider}...`,
  verifyingCode: provider => `Vérification de votre code avec ${provider}...`,
  connectedProvider: provider => `${provider} connecté`,
  connectedPicking: provider => `${provider} connecté. Choix d'un modèle par défaut...`,
  signInFailed: 'Échec de la connexion. Réessayez.',
  signInExpired:
    "La connexion a expiré dans l'attente de l'autorisation. Cela signifie généralement que la page de connexion s'est figée dans l'onglet ouvert (problème côté serveur) — terminez la connexion dans cet onglet, puis réessayez. Si le problème persiste, utilisez plutôt une clé API ou la solution de secours en ligne de commande.",
  signInDidNotFinish: provider =>
    `La connexion avec ${provider} ne s'est pas terminée. Vérifiez votre connexion Internet et réessayez, ou choisissez un autre fournisseur.`,
  tryAgain: 'Réessayer',
  useApiKeyInstead: 'Utiliser une clé API',
  errorDetails: 'Détails',
  pickDifferentProvider: 'Choisissez un autre fournisseur',
  signInWith: provider => `Se connecter avec ${provider}`,
  openedBrowser: provider => `Nous avons ouvert ${provider} dans votre navigateur.`,
  authorizeThere: 'Autorisez Hermes là-bas.',
  copyAuthCode: "Copiez le code d'autorisation et collez-le ci-dessous.",
  pasteAuthCode: "Coller le code d'autorisation",
  reopenAuthPage: "Rouvrir la page d'autorisation",
  waitingAuthorize: 'En attente de votre autorisation...',
  externalPending: provider =>
    `${provider} se connecte via sa propre CLI. Exécutez cette commande dans un terminal, puis revenez et choisissez « Je me suis connecté » :`,
  signedIn: 'Je me suis connecté',
  deviceCodeOpened: provider => `Nous avons ouvert ${provider} dans votre navigateur. Entrez ce code là-bas :`,
  reopenVerification: 'Rouvrir la page de vérification',
  copy: 'Copier',
  defaultModel: 'Modèle par défaut',
  freeTier: 'Gratuit',
  pro: 'Pro',
  free: 'Gratuit',
  price: (input, output) => `${input} entrant / ${output} sortant par Mtok`,
  change: 'Modifier',
  startChatting: 'Commencer',
  docs: provider => `Documentation ${provider}`
}
