import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const frMessagingGateway = {
  hintGatewayStopped: 'Démarrez le gateway de messagerie ici pour vous connecter.',
  restartNeeded: 'Enregistré. Redémarrez le gateway de messagerie pour appliquer les nouveaux paramètres.',
  restartNow: 'Redémarrer maintenant',
  restarting: 'Redémarrage…',
  restartFailedManual: 'Le redémarrage du gateway a échoué — redémarrez-le manuellement et consultez ses journaux.',
  restartFailedManualDetail:
    'Réessayez le redémarrage ; si le problème persiste, ouvrez les journaux et envoyez les diagnostics.',
  startMessagingGateway: 'Démarrer le gateway de messagerie',
  startingMessagingGateway: 'Démarrage du gateway de messagerie…',
  gatewayStartFailed: 'Échec du démarrage du gateway de messagerie.'
} satisfies MessagingGatewayTranslations
