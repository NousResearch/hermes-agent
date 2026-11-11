import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_FR = {
  group: {
    ...GROUP_CHAT_LOCALES.fr,
    newTitle: 'Nouvelle discussion de groupe',
    manageDesc:
      'Un agent peut rejoindre plusieurs discussions de groupe. Les adhésions sont synchronisées sur chaque ordinateur.',
    manageTitle: 'Gérer les groupes',
    settingsTitle: 'Paramètres du groupe',
    settingsDesc: 'Renommez le groupe ou choisissez une image. Les membres et l’historique sont conservés.',
    nameLabel: 'Nom du groupe',
    holdDetection: 'Détecter les demandes d’arrêt',
    holdDetectionHint:
      'Les demandes d’arrêt dans la discussion de groupe mettent les membres concernés en pause jusqu’à leur prochaine mention.',
    compressHistory: 'Compacter l’historique',
    compressHistoryHint: member =>
      `Compacter l’historique masqué de ${member} dans la discussion de groupe pour éviter les réponses vides`,
    compressing: member => `Compactage de l’historique de ${member} dans la discussion de groupe…`,
    compressDone: (member, compressed, detail) =>
      `${compressed} session${compressed > 1 ? 's' : ''} de discussion de groupe compactée${compressed > 1 ? 's' : ''} pour ${member}${detail ? ` — ${detail}` : ''}`,
    compressNothing: member => `Rien à compacter pour ${member} — aucune session de discussion de groupe`,
    compressFailed: (member, error) => `Impossible de compacter l’historique de ${member} : ${error}`,
    searchToAdd: 'Rechercher des agents à ajouter',
    searchToAddPlaceholder: 'Rechercher des agents à ajouter…',
    removeFromSelection: 'Retirer de la sélection',
    disbandTitle: 'Dissoudre la discussion de groupe ?',
    deleteTitle: 'Supprimer la discussion de groupe ?',
    deleteAction: 'Supprimer',
    composerPlaceholder: 'Écrivez un message — tous les agents de cette discussion de groupe le verront.',
    slashCommandsUnsupported:
      'Les commandes slash ne sont pas disponibles dans les discussions de groupe. Ouvrez une discussion individuelle avec un agent pour les utiliser.',
    attachHint: 'Joindre des fichiers — tous les agents qui répondent les verront',
    newThread: 'Nouveau fil',
    reply: 'Répondre',
    replyInThread: 'Répondre dans le fil',
    replyInThreadPlaceholder: 'Répondre dans le fil…',
    openThread: 'Ouvrir ce fil',
    collapseThread: 'Réduire le fil',
    collapseThreadLabel: 'Réduire ce fil',
    activity: 'Activité',
    noActivityYet: 'Aucune activité pour ce tour.',
    showActivity: 'Afficher l’activité de la discussion de groupe',
    hideActivity: 'Masquer l’activité de la discussion de groupe',
    stop: 'Arrêter',
    stopHint: 'Arrêter cette exécution — interrompt le membre actif et met les autres en pause',
    heldMembersStatus: members => `En pause : ${members}`,
    holdReleaseHint: 'Mentionnez un agent en pause ou envoyez @all resume pour reprendre.',
    needsYourInput: 'Un agent de cette discussion de groupe attend votre réponse',
    noMembersToSend: group =>
      `${group} n’a aucun membre à qui envoyer le message — ajoutez un agent ou rouvrez la discussion de groupe si les membres sont encore en cours de chargement.`,
    pictureGenerationFailed: 'Échec de la génération de l’image du groupe',
    nameTaken: name => `Un groupe nommé « ${name} » existe déjà.`,
    you: 'Vous',
    settingsHint: group => `Paramètres du groupe — renommer ${group} ou choisir une image`,
    settingsLabel: group => `Paramètres du groupe ${group}`,
    disbandHint: group => `Dissoudre la discussion de groupe ${group}`,
    disbandLabel: group => `Dissoudre ${group}`,
    disbandAction: 'Dissoudre',
    disbanding: 'Dissolution…',
    disbandDone: 'Groupe dissous',
    disbanded: group => `« ${group} » a été dissous`,
    stopped: group => `${group} arrêté — les tours restants sont en pause jusqu’à la reprise`,
    removeAttachment: 'Retirer la pièce jointe',
    threadFallback: 'Fil',
    replyCount: replies => `${replies} réponse${replies > 1 ? 's' : ''}`,
    dropToThread: 'Déposer pour joindre à cette réponse',
    dropToRoom: 'Déposer pour joindre — tous les agents qui répondent le verront',
    waitingForAnswer: 'En attente de votre réponse…',
    memberThinking: name => `${name} réfléchit…`,
    roomWorking: 'La discussion de groupe est en cours…',
    messageRoom: group => `Envoyer un message à ${group}`,
    newThreadPlaceholder: group => `Nouveau fil dans ${group}… (@nom pour un membre, @everyone pour tous)`,
    everyoneMeta: 'Tous les agents de la discussion de groupe',
    commandApproval: 'approbation de commande',
    answerFailed: (handle, error) => `Impossible d’envoyer la réponse à @${handle} : ${error}`,
    wantsToRunCommand: handle => `@${handle} souhaite exécuter une commande :`,
    asks: handle => `@${handle} demande :`,
    answerTo: member => `Répondre à @${member}`
  },
  roster: {
    ...ROSTER_CHROME.fr,
    search: 'Rechercher des agents et des discussions de groupe',
    searchPlaceholder: 'Rechercher des agents et des discussions de groupe…',
    newBotOrGroup: 'Nouvel agent ou discussion de groupe',
    groupChats: 'Discussions de groupe',
    emptyTitle: 'Aucun agent pour le moment',
    emptyDesc: 'Créez votre premier agent.',
    noMatchQuery: query => `Aucun agent ni discussion de groupe ne correspond à « ${query} »`,
    noMatchQueryOn: (query, gateway) =>
      `Aucun agent ni discussion de groupe ne correspond à « ${query} » sur ${gateway}`,
    noMatchFiltersOn: gateway => `Aucun agent ni discussion de groupe ne correspond à ces filtres sur ${gateway}`,
    noMatchFilters: 'Aucun agent ni discussion de groupe ne correspond à ces filtres.',
    clearFilters: 'Effacer les filtres',
    allHidden: 'Tous les agents sont masqués',
    allHiddenDesc: 'Ils continuent de travailler et conservent leur historique.',
    showHidden: 'Afficher les agents masqués',
    noHiddenMatch: 'Aucun agent masqué ne correspond à ces filtres.',
    hiddenFromRoster: 'Masqué dans la liste',
    pinned: 'Épinglé',
    needsAttention: 'nécessite votre attention',
    needsInput: 'Attend votre réponse',
    botsAndGroups: 'Agents et discussions de groupe',
    botsOnly: 'Agents uniquement',
    groupsOnly: 'Discussions de groupe uniquement',
    anyActivity: 'Toute activité',
    activeNow: 'Actifs maintenant',
    recentlyActive: 'Actifs récemment',
    older: 'Plus anciens',
    gatewayRemoved: 'Machine supprimée',
    onDemand: 'À la demande',
    ready: 'Prêt',
    statusUnknown: 'État inconnu',
    unavailable: 'Indisponible',
    retryNow: 'Réessayer maintenant',
    rosterUnavailable: reason =>
      `Liste indisponible : ${reason}. Si votre version de Hermes ne prend pas en charge profiles.list, mettez Hermes à jour et redémarrez-le.`,
    waitingForGateway:
      'En attente de la connexion à la machine… (les machines distantes peuvent prendre quelques secondes ; nouvelle tentative automatique)'
  },
  sections: {
    newSection: 'Nouvelle section',
    newTitle: 'Nouvelle section',
    renameTitle: 'Renommer la section',
    nameLabel: 'Nom de la section',
    namePlaceholder: 'Par ex. Clients',
    create: 'Créer',
    rename: 'Renommer…',
    moveUp: 'Monter',
    moveDown: 'Descendre',
    unassigned: 'Sans section',
    options: name => `Options de la section ${name}`,
    headingTip: 'Déposez des agents ici · double-cliquez pour renommer',
    emptyHint: 'Glissez des agents ici',
    moveTo: 'Déplacer vers une section',
    newSectionEllipsis: 'Nouvelle section…',
    removeFromSection: 'Retirer de la section',
    deleted: (name, count) =>
      count === 0
        ? `Section « ${name} » supprimée`
        : `Section « ${name} » supprimée — ${count} agent${count === 1 ? '' : 's'} sans section`,
    undo: 'Annuler'
  },
  bot: {
    newChatWith: 'Nouvelle discussion avec cet agent',
    newTitle: 'Nouvel agent',
    thisDevice: 'Cet appareil',
    removeFromAllGroups: 'Retirer de tous les groupes',
    openBotChat: 'Ouvrir la discussion avec l’agent',
    openBotChatInNewTab: 'Ouvrir la discussion avec l’agent dans un nouvel onglet',
    openingChat: 'Ouverture de la discussion…',
    pinToTop: 'Épingler en haut',
    unpin: 'Désépingler',
    pinnedToast: name => `${name} épinglé en haut`,
    unpinnedToast: name => `${name} désépinglé`,
    hide: 'Masquer',
    unhide: 'Afficher',
    hiddenToast: name => `${name} masqué — utilisez le bouton en forme d’œil de la liste des agents pour le retrouver`,
    unhiddenToast: name => `${name} est de retour dans la liste`,
    groupsMenu: groups => `Groupes : ${groups}…`,
    manageGroups: 'Gérer les groupes…',
    metadataLoadFailed: 'Impossible de charger les données de l’agent',
    loadFailed: 'Impossible de charger l’agent',
    groupsLoadFailed: 'Impossible de charger les groupes de l’agent',
    attentionFallback: 'Nécessite votre attention',
    attentionProviderAuth: 'Reconnectez-vous pour cet agent',
    attentionQuota: 'Quota ou solde épuisé',
    attentionMissingConfig: 'Fournisseur non configuré — exécutez hermes model',
    attentionBlocked: 'Agent bloqué — consultez son dernier message',
    editMenu: 'Modifier…',
    duplicate: 'Dupliquer',
    duplicateFailed: 'Échec de la duplication'
  },
  avatar: {
    upload: 'Importer',
    generate: 'Générer',
    generating: 'Génération…'
  },
  screen: {
    autoOpenMenu: 'Ouvrir l’écran lorsque l’agent l’utilise',
    autoOpenOnToast: name => `L’écran de ${name} s’ouvre lorsque l’agent commence à utiliser son bureau`,
    autoOpenOffToast: name => `L’écran de ${name} reste fermé jusqu’à ce que vous l’ouvriez`,
    menu: 'Ouvrir l’écran'
  }
} satisfies Pick<BotsMessages, 'group' | 'roster' | 'sections'> & {
  bot: Pick<
    BotsMessages['bot'],
    | 'newChatWith'
    | 'newTitle'
    | 'thisDevice'
    | 'removeFromAllGroups'
    | 'openBotChat'
    | 'openBotChatInNewTab'
    | 'openingChat'
    | 'pinToTop'
    | 'unpin'
    | 'pinnedToast'
    | 'unpinnedToast'
    | 'hide'
    | 'unhide'
    | 'hiddenToast'
    | 'unhiddenToast'
    | 'groupsMenu'
    | 'manageGroups'
    | 'metadataLoadFailed'
    | 'loadFailed'
    | 'groupsLoadFailed'
    | 'attentionFallback'
    | 'attentionProviderAuth'
    | 'attentionQuota'
    | 'attentionMissingConfig'
    | 'attentionBlocked'
    | 'editMenu'
    | 'duplicate'
    | 'duplicateFailed'
  >
  avatar: Pick<BotsMessages['avatar'], 'upload' | 'generate' | 'generating'>
  screen: Pick<BotsMessages['screen'], 'menu' | 'autoOpenMenu' | 'autoOpenOnToast' | 'autoOpenOffToast'>
}
