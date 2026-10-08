import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const frMemoryDiscovery = {
  installed: 'Installés',
  availableToInstall: 'Disponibles à installer',
  installationRequired: 'Installation requise',
  reviewInstall: 'Vérifier et installer',
  exploreAll: 'Tout explorer…',
  missing: 'Manquant',
  installConsent:
    'Installe et active le plugin avec ses dépendances. Le fournisseur mémoire actif reste inchangé jusqu’à votre sélection explicite.',
  builtin: 'Intégré',
  configureElsewhere:
    'Configurez ce fournisseur via son assistant CLI ou mettez Hermes à jour pour enregistrer sans activer.',
  notReady:
    'Terminez la configuration et installez les dépendances. Après une installation, redémarrez le backend puis réessayez.',
  useFailed: 'Impossible d’utiliser ce fournisseur. Vérifiez sa configuration et réessayez.',

  activeProvider: name => `Actif : ${name}`,
  useProvider: 'Utiliser ce fournisseur',
  loadFailed: 'Impossible de charger les fournisseurs de mémoire',
  ownerChanged: 'Revenez à la connexion et au profil utilisés à l’ouverture de cet installateur, puis réessayez.',
  notDiscovered:
    'Le paquet est installé, mais son fournisseur de mémoire n’est pas encore détecté. Revenez aux paramètres de mémoire pour réessayer.',
  installedNotice: 'Fournisseur détecté. Configurez-le, puis choisissez explicitement de l’utiliser.',
  backToMemory: 'Retour aux paramètres de mémoire',
  connect: 'Connecter',
  reconnect: 'Reconnecter',
  connectOAuth: 'Connecter via OAuth',
  apiKeySet: 'Clé API définie',
  oauthSet: 'OAuth connecté',
  waitingConsent: 'En attente du consentement dans le navigateur…',
  stopWaiting: 'Arrêter d’attendre',
  stoppedWaiting: 'Attente arrêtée. L’autorisation est peut-être encore en cours.',
  startFailed: 'Impossible de démarrer la connexion.',
  connectionFailed: 'La connexion a échoué.'
} satisfies Partial<MemoryDiscoveryTranslations>
