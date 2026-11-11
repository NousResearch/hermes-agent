import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_DE = {
  group: {
    ...GROUP_CHAT_LOCALES.de,
    newTitle: 'Neuer Gruppenchat',
    manageDesc:
      'Ein Agent kann mehreren Gruppenchats beitreten. Mitgliedschaften werden auf allen Computern synchronisiert.',
    manageTitle: 'Gruppen verwalten',
    settingsTitle: 'Gruppeneinstellungen',
    settingsDesc: 'Benenne die Gruppe um oder wähle ein Gruppenbild. Mitglieder und Verlauf bleiben erhalten.',
    nameLabel: 'Gruppenname',
    holdDetection: 'Stoppanweisungen erkennen',
    holdDetectionHint:
      'Stoppanweisungen im Gruppenchat pausieren angesprochene Mitglieder, bis sie erneut erwähnt werden.',
    compressHistory: 'Verlauf komprimieren',
    compressHistoryHint: member =>
      `Verborgenen Gruppenchat-Verlauf von ${member} komprimieren, um leere Antworten zu vermeiden`,
    compressing: member => `Gruppenchat-Verlauf von ${member} wird komprimiert…`,
    compressDone: (member, compressed, detail) =>
      `${compressed} Gruppenchat-Sitzung${compressed === 1 ? '' : 'en'} für ${member} komprimiert${detail ? ` — ${detail}` : ''}`,
    compressNothing: member => `Nichts für ${member} zu komprimieren — noch keine Gruppenchat-Sitzung`,
    compressFailed: (member, error) => `Gruppenchat-Verlauf von ${member} konnte nicht komprimiert werden: ${error}`,
    searchToAdd: 'Agenten zum Hinzufügen suchen',
    searchToAddPlaceholder: 'Agenten zum Hinzufügen suchen…',
    removeFromSelection: 'Aus Auswahl entfernen',
    disbandTitle: 'Gruppenchat auflösen?',
    deleteTitle: 'Gruppenchat löschen?',
    deleteAction: 'Löschen',
    composerPlaceholder: 'Schreibe etwas — jeder Agent in diesem Gruppenchat sieht deine Nachricht.',
    slashCommandsUnsupported:
      'Slash-Befehle sind in Gruppenchats nicht verfügbar. Öffne dafür einen einzelnen Agenten-Chat.',
    attachHint: 'Dateien anhängen — jeder antwortende Agent sieht sie',
    newThread: 'Neuer Gesprächsfaden',
    reply: 'Antworten',
    replyInThread: 'Im Gesprächsfaden antworten',
    replyInThreadPlaceholder: 'Im Gesprächsfaden antworten…',
    openThread: 'Diesen Gesprächsfaden öffnen',
    collapseThread: 'Gesprächsfaden einklappen',
    collapseThreadLabel: 'Diesen Gesprächsfaden einklappen',
    activity: 'Aktivität',
    noActivityYet: 'Noch keine Aktivität in diesem Durchgang.',
    showActivity: 'Gruppenchat-Aktivität anzeigen',
    hideActivity: 'Gruppenchat-Aktivität ausblenden',
    stop: 'Stoppen',
    stopHint: 'Diesen Lauf stoppen — unterbricht das aktive Mitglied und pausiert die anderen',
    heldMembersStatus: members => `Pausiert: ${members}`,
    holdReleaseHint: 'Erwähne einen pausierten Agenten oder sende @all resume, um fortzufahren.',
    needsYourInput: 'Ein Agent in diesem Gruppenchat benötigt deine Eingabe',
    noMembersToSend: group =>
      `${group} hat keine Empfänger — füge einen Agenten hinzu oder öffne den Gruppenchat erneut, falls die Mitglieder noch laden.`,
    pictureGenerationFailed: 'Gruppenbild konnte nicht erstellt werden',
    nameTaken: name => `Eine Gruppe namens „${name}“ existiert bereits.`,
    you: 'Du',
    settingsHint: group => `Gruppeneinstellungen — ${group} umbenennen oder ein Gruppenbild wählen`,
    settingsLabel: group => `Einstellungen für ${group}`,
    disbandHint: group => `Den Gruppenchat ${group} auflösen`,
    disbandLabel: group => `${group} auflösen`,
    disbandAction: 'Auflösen',
    disbanding: 'Wird aufgelöst…',
    disbandDone: 'Aufgelöst',
    disbanded: group => `„${group}“ aufgelöst`,
    stopped: group => `${group} gestoppt — verbleibende Durchgänge sind bis zur Fortsetzung pausiert`,
    removeAttachment: 'Anhang entfernen',
    threadFallback: 'Gesprächsfaden',
    replyCount: replies => `${replies} Antwort${replies === 1 ? '' : 'en'}`,
    dropToThread: 'Ablegen, um an diese Antwort anzuhängen',
    dropToRoom: 'Zum Anhängen ablegen — jeder antwortende Agent sieht es',
    waitingForAnswer: 'Warte auf deine Antwort…',
    memberThinking: name => `${name} denkt nach…`,
    roomWorking: 'Der Gruppenchat arbeitet…',
    messageRoom: group => `Nachricht an ${group}`,
    newThreadPlaceholder: group => `Neuer Gesprächsfaden in ${group}… (@name für ein Mitglied, @everyone für alle)`,
    everyoneMeta: 'Jeder Agent im Gruppenchat',
    commandApproval: 'Befehlsfreigabe',
    answerFailed: (handle, error) => `Antwort an @${handle} konnte nicht gesendet werden: ${error}`,
    wantsToRunCommand: handle => `@${handle} möchte einen Befehl ausführen:`,
    asks: handle => `@${handle} fragt:`,
    answerTo: member => `@${member} antworten`
  },
  roster: {
    ...ROSTER_CHROME.de,
    search: 'Agenten und Gruppenchats suchen',
    searchPlaceholder: 'Agenten und Gruppenchats suchen…',
    newBotOrGroup: 'Neuer Agent oder Gruppenchat',
    groupChats: 'Gruppenchats',
    emptyTitle: 'Noch keine Agenten',
    emptyDesc: 'Erstelle deinen ersten Agenten.',
    noMatchQuery: query => `Keine Agenten oder Gruppenchats für „${query}“ gefunden`,
    noMatchQueryOn: (query, gateway) => `Keine Agenten oder Gruppenchats für „${query}“ auf ${gateway} gefunden`,
    noMatchFiltersOn: gateway => `Keine Agenten oder Gruppenchats entsprechen diesen Filtern auf ${gateway}`,
    noMatchFilters: 'Keine Agenten oder Gruppenchats entsprechen diesen Filtern.',
    clearFilters: 'Filter zurücksetzen',
    allHidden: 'Alle Agenten sind ausgeblendet',
    allHiddenDesc: 'Sie arbeiten weiter und behalten ihren Verlauf.',
    showHidden: 'Ausgeblendete Agenten anzeigen',
    noHiddenMatch: 'Keine ausgeblendeten Agenten entsprechen diesen Filtern.',
    hiddenFromRoster: 'In der Liste ausgeblendet',
    pinned: 'Angeheftet',
    needsAttention: 'benötigt Aufmerksamkeit',
    needsInput: 'Benötigt deine Eingabe',
    botsAndGroups: 'Agenten und Gruppenchats',
    botsOnly: 'Nur Agenten',
    groupsOnly: 'Nur Gruppenchats',
    anyActivity: 'Jede Aktivität',
    activeNow: 'Jetzt aktiv',
    recentlyActive: 'Kürzlich aktiv',
    older: 'Älter',
    gatewayRemoved: 'Rechner entfernt',
    onDemand: 'Bei Bedarf',
    ready: 'Bereit',
    statusUnknown: 'Status unbekannt',
    unavailable: 'Nicht verfügbar',
    retryNow: 'Jetzt erneut versuchen',
    rosterUnavailable: reason =>
      `Liste nicht verfügbar: ${reason}. Falls deine Hermes-Version profiles.list noch nicht unterstützt, aktualisiere Hermes und starte es neu.`,
    waitingForGateway:
      'Warte auf die Verbindung zum Rechner… (entfernte Rechner können einige Sekunden benötigen; automatischer neuer Versuch)'
  },
  sections: {
    newSection: 'Neuer Abschnitt',
    newTitle: 'Neuer Abschnitt',
    renameTitle: 'Abschnitt umbenennen',
    nameLabel: 'Abschnittsname',
    namePlaceholder: 'z. B. Kunden',
    create: 'Erstellen',
    rename: 'Umbenennen…',
    moveUp: 'Nach oben',
    moveDown: 'Nach unten',
    unassigned: 'Nicht zugeordnet',
    options: name => `Optionen für den Abschnitt ${name}`,
    headingTip: 'Agenten hier ablegen · zum Umbenennen doppelklicken',
    emptyHint: 'Agenten hierher ziehen',
    moveTo: 'In Abschnitt verschieben',
    newSectionEllipsis: 'Neuer Abschnitt…',
    removeFromSection: 'Aus Abschnitt entfernen',
    deleted: (name, count) =>
      count === 0
        ? `„${name}“ gelöscht`
        : `„${name}“ gelöscht — ${count} Agent${count === 1 ? '' : 'en'} jetzt nicht zugeordnet`,
    undo: 'Rückgängig'
  },
  bot: {
    newChatWith: 'Neuer Chat mit diesem Agenten',
    newTitle: 'Neuer Agent',
    thisDevice: 'Dieses Gerät',
    removeFromAllGroups: 'Aus allen Gruppen entfernen',
    openBotChat: 'Agenten-Chat öffnen',
    openBotChatInNewTab: 'Agenten-Chat in neuem Tab öffnen',
    openingChat: 'Chat wird geöffnet…',
    pinToTop: 'Oben anheften',
    unpin: 'Lösen',
    pinnedToast: name => `${name} oben angeheftet`,
    unpinnedToast: name => `${name} gelöst`,
    hide: 'Ausblenden',
    unhide: 'Einblenden',
    hiddenToast: name => `${name} ausgeblendet — das Augensymbol über der Agenten-Liste zeigt ausgeblendete Agenten`,
    unhiddenToast: name => `${name} ist wieder in der Liste`,
    groupsMenu: groups => `Gruppen: ${groups}…`,
    manageGroups: 'Gruppen verwalten…',
    metadataLoadFailed: 'Agent-Metadaten konnten nicht geladen werden',
    loadFailed: 'Agent konnte nicht geladen werden',
    groupsLoadFailed: 'Agent-Gruppen konnten nicht geladen werden',
    attentionFallback: 'Benötigt Aufmerksamkeit',
    attentionProviderAuth: 'Für diesen Agenten erneut anmelden',
    attentionQuota: 'Kontingent oder Guthaben aufgebraucht',
    attentionMissingConfig: 'Anbieter nicht eingerichtet — hermes model ausführen',
    attentionBlocked: 'Agent blockiert — letzte Nachricht ansehen',
    editMenu: 'Bearbeiten…',
    duplicate: 'Duplizieren',
    duplicateFailed: 'Duplizieren fehlgeschlagen'
  },
  avatar: {
    upload: 'Hochladen',
    generate: 'Erstellen',
    generating: 'Wird erstellt…'
  },
  screen: {
    autoOpenMenu: 'Bildschirm öffnen, wenn der Agent ihn verwendet',
    autoOpenOnToast: name => `Der Bildschirm von ${name} öffnet sich, sobald der Agent seinen Desktop verwendet`,
    autoOpenOffToast: name => `Der Bildschirm von ${name} bleibt geschlossen, bis du ihn öffnest`,
    menu: 'Bildschirm öffnen'
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
