// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines
// (en_desktop_commands.ts is the source).
export const frDesktopCommands = {
  error: (message: string) => `erreur : ${message}`,
  rollbackUsage: 'utilisation : /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled:
    'Les points de restauration ne sont pas activés pour cette session (définissez checkpoints.enabled dans config.yaml).',
  checkpointsEmpty: 'Aucun point de restauration pour ce répertoire de travail.',
  checkpointsHeader: 'Points de restauration (du plus récent au plus ancien) :',
  checkpointsHint:
    'Restaurez avec /rollback <n> [file] (ajoutez --all pour écraser aussi vos propres modifications) ; prévisualisez avec /rollback diff <n>.',
  noDiff: 'Aucun changement depuis ce point de restauration.',
  rollbackFailed: (error: string) => `Échec de la restauration : ${error}`,
  unknownError: 'erreur inconnue',
  restored: (hash: string) => (hash ? `Point de restauration ${hash} restauré.` : 'Point de restauration restauré.'),
  rewound: (count: string) => `${count} message(s) retiré(s) de cette conversation.`,
  reopenForRewind: 'Rouvrez cette discussion pour voir la conversation rembobinée.',
  agentsEmpty: 'Aucun sous-agent actif ni processus en arrière-plan dans cette discussion.',
  subagentsHeader: (count: string) => `Sous-agents (${count}) :`,
  subagentFallback: 'sous-agent',
  statusRunning: 'en cours',
  toolCalls: (count: string) => `${count} appels d’outils`,
  processesHeader: (count: string) => `Processus en arrière-plan (${count}) :`,
  configHeader: 'Configuration (affichage, approbations, collage et voix ; le reste n’est pas affiché) :',
  configDefaults: '(valeurs par défaut)',
  configFooter:
    'Modifiez les réglages dans Réglages, ou éditez config.yaml avec `hermes config edit` sur l’hôte de la passerelle.',
  historyHeader: (count: string) => `Historique de la conversation (${count} messages) :`,
  historyEmpty: 'Aucun message pour l’instant.',
  historyYou: 'Vous',
  historyHermes: 'Hermes',
  saveUsage: 'utilisation : /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'Non enregistré : cette passerelle ne peut pas exporter de copie expurgée, nommée ou autre que JSON. Mettez à jour la passerelle, ou lancez /save sans option pour écrire le JSON brut sur l’hôte de la passerelle.',
  savedOnHost: (path: string) => `Enregistré sur l’hôte de la passerelle : ${path}`,
  unknownPath: '(chemin inconnu)',
  saveUnsupported: 'erreur : cette version de l’application ne peut pas enregistrer de fichiers',
  saveDialogTitle: 'Enregistrer la conversation',
  savedTo: (path: string) => `Conversation enregistrée dans ${path}`,
  saveCancelled: 'Non enregistré (annulé).',
  exportRemote:
    '/export empaquette un profil depuis l’hôte de la passerelle, qui n’est pas cette machine. Lancez `hermes profile export` sur cet hôte.',
  importRemote:
    '/import installe un profil sur l’hôte de la passerelle, qui n’est pas cette machine. Lancez `hermes profile import <archive>` sur cet hôte.',
  timestampsUsage: 'utilisation : /timestamps [on|off|status]',
  timestampsOn: 'Horodatage : activé (cette fenêtre ; définissez display.timestamps dans config.yaml pour le garder)',
  timestampsOff:
    'Horodatage : désactivé (cette fenêtre ; définissez display.timestamps dans config.yaml pour le garder)'
}
