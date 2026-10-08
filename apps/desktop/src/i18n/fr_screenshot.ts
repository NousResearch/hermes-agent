import type { ScreenshotTranslations } from './types_screenshot'

export const frScreenshot: ScreenshotTranslations = {
  leftCommand: 'Gauche ⌘',
  rightCommand: 'Droite ⌘',
  enabledTitle: "Raccourci de capture d'écran",
  enabledDesc:
    "Appuyez simultanément sur les touches Commande (⌘) gauche et droite depuis n'importe quelle application pour capturer sa fenêtre au premier plan et la joindre au brouillon Hermes actuel. Rien n'est envoyé automatiquement. Désactivé par défaut et limité à ce Mac. Le contenu peut être sensible : vérifiez la pièce jointe avant l'envoi.",
  statusTitle: "État du raccourci de capture d'écran",
  checking: "Vérification du raccourci de capture d'écran…",
  disabled: "Le raccourci de capture d'écran est désactivé.",
  starting: "Démarrage de l'écouteur du raccourci ; il n'est pas encore prêt.",
  ready: "Le raccourci est prêt. Les captures d'écran sont jointes au brouillon actuel sans être envoyées.",
  inputPermission:
    "L'autorisation Surveillance de l'entrée permet à Hermes de détecter les touches Commande (⌘) gauche et droite lorsqu'une autre application est active. Autorisez Hermes dans Réglages Système → Confidentialité et sécurité → Surveillance de l'entrée, puis réessayez.",
  screenPermission:
    "L'autorisation Enregistrement de l'écran permet à Hermes de capturer la fenêtre au premier plan. Autorisez Hermes dans Réglages Système → Confidentialité et sécurité → Enregistrement de l'écran, puis réessayez. Redémarrez Hermes si macOS le demande.",
  openSettings: 'Ouvrir les Réglages Système',
  retry: 'Réessayer',
  unavailable: "Le raccourci de capture d'écran est indisponible. Réessayez ou désactivez-le.",
  errorTitle: "Erreur du raccourci de capture d'écran",
  loadFailed: "Impossible de lire l'état du raccourci. Réessayez pour vérifier son réglage actuel.",
  saveFailed: 'Impossible de confirmer la modification du raccourci. Réessayez pour vérifier son réglage actuel.',
  permissionFailed:
    "Impossible d'ouvrir les Réglages Système. Ouvrez manuellement Confidentialité et sécurité, puis réessayez.",
  captureFailed: "Impossible de capturer la fenêtre au premier plan. Rien n'a été joint ni envoyé.",
  contextChanged: "Le brouillon actuel a changé pendant la capture. L'image n'a pas été jointe ni envoyée."
}
