import type { ConnectionsCopy } from './connections-copy'

export const fr: ConnectionsCopy = {
  badges: { destructive: 'Destructif', off: 'Désactivé', read: 'Lecture', write: 'Écriture' },
  bulk: {
    discard: 'Annuler',
    save: 'Enregistrer',
    someOn: 'Certains activés',
    turnAllOff: 'Tout désactiver',
    turnAllOn: 'Tout activer'
  },
  card: {
    add: 'Ajouter',
    connect: 'Connecter',
    connected: 'Connectée',
    disconnect: 'Déconnecter',
    manage: 'Gérer',
    menu: 'Plus d’options',
    notConnected: 'Non connectée',
    reconnect: 'Reconnecter',
    refreshTools: 'Actualiser la liste d’outils',
    remove: 'Supprimer',
    rename: 'Renommer',
    toolsOff: count => `${count} outil${count === 1 ? '' : 's'} désactivé${count === 1 ? '' : 's'}`,
    toolsOn: count => `${count} outil${count === 1 ? '' : 's'} activé${count === 1 ? '' : 's'}`,
    toolsSomeOn: (on, total) => `${on} outils sur ${total} activés`,
    waiting: 'En attente de connexion'
  },
  categories: {
    'customers-growth': {
      name: 'Clients et croissance',
      subtitle: 'Marketing, ventes et assistance.'
    },
    'data-analytics': {
      name: 'Données et analyses',
      subtitle: 'Comprendre votre produit et vos données.'
    },
    'design-content': {
      name: 'Design et contenu',
      subtitle: 'Créer, concevoir et publier.'
    },
    engineering: {
      name: 'Ingénierie',
      subtitle: 'Développer, héberger et surveiller.'
    },
    money: {
      name: 'Argent',
      subtitle: 'Paiements, banque et investissements.'
    },
    reference: {
      name: 'Référence',
      subtitle: 'Trouver des réponses sans quitter la conversation.'
    },
    'travel-fitness': {
      name: 'Voyages et forme',
      subtitle: 'Planifier des trajets et suivre vos entraînements.'
    },
    'work-productivity': {
      name: 'Travail et productivité',
      subtitle: 'Documents, projets, réunions et les applis de votre équipe.'
    }
  },
  disconnect: {
    body: name => `${name} cessera de fonctionner dans la conversation jusqu’à ce que vous le reconnectiez.`,
    cancel: 'Annuler',
    confirm: 'Déconnecter',
    title: name => `Déconnecter ${name} ?`
  },
  errors: {
    connectBody: 'Vérifiez que vous avez terminé la connexion, puis réessayez.',
    connectTitle: name => `Impossible de connecter ${name}`,
    dismiss: 'Ignorer',
    loadBody: 'Vos applis n’ont pas pu être vérifiées. Ce qui était déjà chargé reste affiché.',
    loadTitle: 'Impossible de charger les connexions',
    tryAgain: 'Réessayer'
  },
  facetTooltips: {
    destructive: 'Destructif : peut supprimer des données ou annuler du travail.',
    read: 'Lecture : peut consulter vos données sans les modifier.',
    write: 'Écriture : peut créer et modifier des données.'
  },
  filters: {
    activeFilters: count => `${count} filtre${count === 1 ? '' : 's'}`,
    addMcpServer: 'Ajouter un serveur MCP',
    all: 'Tout',
    allCategories: 'Toutes les catégories',
    apply: 'Afficher les résultats',
    backToConnections: 'Connexions',
    button: 'Filtres',
    categories: 'Catégories',
    categoryChipLabel: 'Catégorie',
    chooseOne: 'Choisissez-en un',
    clear: 'Tout effacer',
    close: 'Fermer',
    findCategory: 'Trouver une catégorie',
    noResultsBody: 'Rien ne correspond à ces filtres. Effacez-les pour voir toutes les applis.',
    noResultsQuery: query => `Rien ne correspond à « ${query} ». Essayez une autre recherche ou effacez vos filtres.`,
    noResultsTitle: 'Aucune connexion ne correspond',
    resultCount: count => `${count} connexion${count === 1 ? '' : 's'}`,
    searchCategory: name => `Rechercher ${name}`,
    selectedCount: count => `${count} sélectionné${count === 1 ? '' : 's'}`,
    showConnections: count => `Afficher ${count} connexion${count === 1 ? '' : 's'}`,
    status: 'Statut',
    statusAny: 'Toutes',
    statusChipLabel: 'Statut',
    statusConnected: 'Connectées',
    statusNotConnected: 'Non connectées',
    title: 'Filtres'
  },
  page: {
    subtitle: 'Les applis et outils que Hermes peut utiliser pour vous.',
    title: 'Connexions'
  },
  removeAccount: {
    body: name => `${name} sera retiré de cette connexion. L’appli reste connectée avec vos autres comptes.`,
    cancel: 'Annuler',
    confirm: 'Supprimer',
    title: 'Supprimer ce compte ?'
  },
  search: { clear: 'Effacer la recherche', placeholder: 'Rechercher des connexions' },
  sections: {
    connected: 'Connectées',
    other: 'Autres applis',
    seeAll: count => `Voir les ${count}`,
    suggested: 'Suggestions pour vous'
  },
  sheet: {
    account: 'Compte',
    accounts: 'Comptes',
    addAnAccount: '+ Ajouter un compte',
    addAnotherAccount: '+ Ajouter un autre compte',
    availableSubtitle: 'En pause pour tous les agents ; restez connecté.',
    availableToAgents: 'Disponible pour les agents',
    cancelConnection: 'Annuler la connexion',
    changeAfterConnect: 'Vous pourrez les modifier après la connexion.',
    close: 'Fermer',
    connected: 'Connectée',
    connectedOn: date => `Connecté le ${date}`,
    disconnectFailed: app => `Impossible de déconnecter ${app}. Vérifiez votre connexion et `,
    facetDescriptions: {
      destructive: 'Suppression définitive',
      read: 'Lecture seule',
      write: 'Créer et modifier'
    },
    finishSignIn: 'Terminer la connexion',
    finishSignInBody: app => `Approuvez l’accès sur la page ${app} dans votre navigateur, puis revenez ici.`,
    notConnected: 'Non connectée',
    permissionsTooltip: app =>
      `${app} définit si chaque outil est en Lecture, Écriture ou Destructif. Vous choisissez ceux qui sont activés.`,
    previewAll: count => `Aperçu des ${count}`,
    removeAccountBody: app => `Les agents ne peuvent plus utiliser ce compte ${app}. Les autres comptes restent connectés.`,
    removeAccountTitle: alias => `Supprimer « ${alias} » ?`,
    removeFailed: alias => `Impossible de supprimer « ${alias} ». Vérifiez votre connexion et `,
    removedSuccess: alias => `« ${alias} » a bien été supprimé`,
    renamePlaceholder: 'Nom du compte',
    reopenSignIn: 'Rouvrir la page de connexion',
    reviewAll: count => `Vérifier les ${count}`,
    reviewTools: 'Voir tous les outils',
    signInBody: app => `Ouvre ${app} dans votre navigateur. Hermes ne voit jamais votre mot de passe.`,
    signInWith: app => `Se connecter avec ${app}`,
    startsOff: 'Désactivé au départ',
    startsOn: 'Activé au départ',
    toolPermissions: 'Autorisations des outils',
    tools: 'Outils',
    toolsCount: count => `${count} outil${count === 1 ? '' : 's'}`,
    waiting: 'En attente de connexion',
    waitingFor: app => `En attente de ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'IA, médias et présentations',
    'automation-ai-apis': 'Automatisation, IA et API',
    'bi-search-warehouses': 'BI, recherche et entrepôts',
    'code-review-ci': 'Revue de code et CI',
    communication: 'Communication',
    'customer-support': 'Assistance client',
    'design-whiteboards': 'Design et tableaux blancs',
    'docs-files': 'Documents et fichiers',
    'docs-knowledge': 'Documents et connaissances',
    'hiring-people': 'Recrutement et équipes',
    'hosting-databases': 'Hébergement et bases de données',
    'marketing-seo': 'Marketing et SEO',
    'meetings-scheduling': 'Réunions et planification',
    'monitoring-incidents': 'Supervision et incidents',
    'outdoors-fitness': 'Plein air et forme',
    'payments-commerce': 'Paiements et commerce',
    'product-analytics': 'Analyse produit',
    'projects-tasks': 'Projets et tâches',
    'sales-crm': 'Ventes et CRM',
    'spend-investing-crypto': 'Dépenses, investissement et crypto',
    travel: 'Voyages',
    'websites-assets': 'Sites et ressources'
  },
  toast: {
    connected: name => `${name} est connectée.`,
    disconnected: name => `${name} est déconnectée.`,
    removed: name => `${name} a été supprimé.`,
    renamed: name => `Renommé en ${name}.`
  },
  toolFilters: {
    all: 'Tout',
    apply: 'Afficher les résultats',
    areas: 'Domaines',
    behavior: 'Comportement',
    clear: 'Tout effacer',
    close: 'Fermer',
    status: 'Statut',
    title: 'Filtrer les outils',
    uncategorised: 'Sans catégorie'
  },
  tools: {
    allAreas: 'Tous les domaines',
    areaChipLabel: 'Domaine',
    back: 'Retour aux connexions',
    emptyBody: 'Cette appli n’a partagé aucun outil.',
    emptyTitle: 'Aucun outil pour l’instant',
    errorBody: 'La liste d’outils n’a pas pu être lue. Réessayez.',
    errorTitle: 'Impossible de charger les outils',
    filter: 'Filtres',
    findArea: 'Trouver un domaine',
    loading: 'Chargement des outils',
    lockedByOrg: 'Désactivé par votre organisation',
    noResultsBody: 'Rien ne correspond à cette recherche ni à ces filtres.',
    noResultsTitle: 'Aucun outil ne correspond',
    off: 'Désactivé',
    on: 'Activé',
    pageSummary: (on, total, app) =>
      `${on} outils sur ${total} activés. Les modifications sont enregistrées ensemble et s’appliquent à chaque agent qui utilise ${app}.`,
    pageTitle: app => `Outils ${app}`,
    permissionChipLabel: 'Autorisation',
    searchAppTools: app => `Rechercher dans les outils ${app}`,
    searchPlaceholder: 'Rechercher des outils',
    showTools: count => `Afficher ${count} outil${count === 1 ? '' : 's'}`,
    subtitle: name => `Choisissez ce que ${name} peut faire.`,
    summary: (on, total) => `${on} sur ${total} activés`,
    title: 'Tous les outils',
    toolNamePrefix: 'Nom de l’outil :',
    unsavedChanges: count => `${count} modification${count === 1 ? '' : 's'} non enregistrée${count === 1 ? '' : 's'}`,
    whatItDoes: 'Ce qu’il fait',
    whatItDoesHint: 'Choisissez-en un ou plusieurs. « Tout » efface les autres choix.'
  }
}


export const de: ConnectionsCopy = {
  badges: { destructive: 'Destruktiv', off: 'Aus', read: 'Lesen', write: 'Schreiben' },
  bulk: {
    discard: 'Verwerfen',
    save: 'Speichern',
    someOn: 'Einige an',
    turnAllOff: 'Alle ausschalten',
    turnAllOn: 'Alle einschalten'
  },
  card: {
    add: 'Hinzufügen',
    connect: 'Verbinden',
    connected: 'Verbunden',
    disconnect: 'Trennen',
    manage: 'Verwalten',
    menu: 'Weitere Optionen',
    notConnected: 'Nicht verbunden',
    reconnect: 'Neu verbinden',
    refreshTools: 'Werkzeugliste aktualisieren',
    remove: 'Entfernen',
    rename: 'Umbenennen',
    toolsOff: count => `${count} Werkzeug${count === 1 ? '' : 'e'} aus`,
    toolsOn: count => `${count} Werkzeug${count === 1 ? '' : 'e'} an`,
    toolsSomeOn: (on, total) => `${on} von ${total} Werkzeugen an`,
    waiting: 'Warten auf Anmeldung'
  },
  categories: {
    'customers-growth': {
      name: 'Kunden & Wachstum',
      subtitle: 'Marketing, Vertrieb und Support.'
    },
    'data-analytics': {
      name: 'Daten & Analysen',
      subtitle: 'Verstehe dein Produkt und deine Daten.'
    },
    'design-content': {
      name: 'Design & Inhalte',
      subtitle: 'Erstellen, gestalten und veröffentlichen.'
    },
    engineering: {
      name: 'Entwicklung',
      subtitle: 'Bauen, hosten und überwachen.'
    },
    money: {
      name: 'Geld',
      subtitle: 'Zahlungen, Banking und Investments.'
    },
    reference: {
      name: 'Nachschlagen',
      subtitle: 'Antworten finden, ohne den Chat zu verlassen.'
    },
    'travel-fitness': {
      name: 'Reisen & Fitness',
      subtitle: 'Reisen planen und Training verfolgen.'
    },
    'work-productivity': {
      name: 'Arbeit & Produktivität',
      subtitle: 'Dokumente, Projekte, Meetings und die Apps deines Teams.'
    }
  },
  disconnect: {
    body: name => `${name} funktioniert nicht mehr im Chat, bis du es erneut verbindest.`,
    cancel: 'Abbrechen',
    confirm: 'Trennen',
    title: name => `${name} trennen?`
  },
  errors: {
    connectBody: 'Prüfe, ob du die Anmeldung abgeschlossen hast, und versuche es erneut.',
    connectTitle: name => `Verbindung zu ${name} fehlgeschlagen`,
    dismiss: 'Schließen',
    loadBody: 'Deine Apps konnten nicht geprüft werden. Bereits Geladenes bleibt sichtbar.',
    loadTitle: 'Verbindungen konnten nicht geladen werden',
    tryAgain: 'Erneut versuchen'
  },
  facetTooltips: {
    destructive: 'Destruktiv: darf Daten löschen oder Arbeit rückgängig machen.',
    read: 'Lesen: darf deine Daten ansehen, aber nicht ändern.',
    write: 'Schreiben: darf Daten erstellen und ändern.'
  },
  filters: {
    activeFilters: count => `${count} Filter`,
    addMcpServer: 'MCP-Server hinzufügen',
    all: 'Alle',
    allCategories: 'Alle Kategorien',
    apply: 'Ergebnisse anzeigen',
    backToConnections: 'Verbindungen',
    button: 'Filter',
    categories: 'Kategorien',
    categoryChipLabel: 'Kategorie',
    chooseOne: 'Eins auswählen',
    clear: 'Alle löschen',
    close: 'Schließen',
    findCategory: 'Kategorie finden',
    noResultsBody: 'Zu diesen Filtern gibt es nichts. Lösche sie, um alle Apps zu sehen.',
    noResultsQuery: query => `Zu „${query}“ gibt es nichts. Versuche eine andere Suche oder lösche die Filter.`,
    noResultsTitle: 'Keine passenden Verbindungen',
    resultCount: count => `${count} Verbindung${count === 1 ? '' : 'en'}`,
    searchCategory: name => `${name} durchsuchen`,
    selectedCount: count => `${count} ausgewählt`,
    showConnections: count => `${count} Verbindung${count === 1 ? '' : 'en'} anzeigen`,
    status: 'Status',
    statusAny: 'Alle',
    statusChipLabel: 'Status',
    statusConnected: 'Verbunden',
    statusNotConnected: 'Nicht verbunden',
    title: 'Filter'
  },
  page: {
    subtitle: 'Apps und Werkzeuge, die Hermes für dich nutzen kann.',
    title: 'Verbindungen'
  },
  removeAccount: {
    body: name => `${name} wird aus dieser Verbindung entfernt. Die App bleibt mit deinen anderen Konten verbunden.`,
    cancel: 'Abbrechen',
    confirm: 'Entfernen',
    title: 'Dieses Konto entfernen?'
  },
  search: { clear: 'Suche löschen', placeholder: 'Verbindungen suchen' },
  sections: {
    connected: 'Verbunden',
    other: 'Weitere Apps',
    seeAll: count => `Alle ${count} ansehen`,
    suggested: 'Für dich vorgeschlagen'
  },
  sheet: {
    account: 'Konto',
    accounts: 'Konten',
    addAnAccount: '+ Konto hinzufügen',
    addAnotherAccount: '+ Weiteres Konto hinzufügen',
    availableSubtitle: 'Für alle Agenten pausiert; angemeldet bleiben.',
    availableToAgents: 'Für Agenten verfügbar',
    cancelConnection: 'Verbindung abbrechen',
    changeAfterConnect: 'Du kannst dies nach dem Verbinden ändern.',
    close: 'Schließen',
    connected: 'Verbunden',
    connectedOn: date => `Verbunden am ${date}`,
    disconnectFailed: app => `Verbindung zu ${app} konnte nicht getrennt werden. Prüfe deine Verbindung und `,
    facetDescriptions: {
      destructive: 'Endgültig löschen',
      read: 'Nur ansehen',
      write: 'Erstellen und bearbeiten'
    },
    finishSignIn: 'Anmeldung abschließen',
    finishSignInBody: app =>
      `Bestätige den Zugriff auf der ${app}-Seite in deinem Browser und komm dann hierher zurück.`,
    notConnected: 'Nicht verbunden',
    permissionsTooltip: app =>
      `${app} legt fest, ob ein Werkzeug Lesen, Schreiben oder Destruktiv ist. Du wählst, welche an sind.`,
    previewAll: count => `Alle ${count} ansehen`,
    removeAccountBody: app =>
      `Agenten können dieses ${app}-Konto nicht mehr nutzen. Andere Konten bleiben verbunden.`,
    removeAccountTitle: alias => `„${alias}“ entfernen?`,
    removeFailed: alias => `„${alias}“ konnte nicht entfernt werden. Prüfe deine Verbindung und `,
    removedSuccess: alias => `„${alias}“ wurde entfernt`,
    renamePlaceholder: 'Kontoname',
    reopenSignIn: 'Anmeldeseite erneut öffnen',
    reviewAll: count => `Alle ${count} prüfen`,
    reviewTools: 'Alle Werkzeuge prüfen',
    signInBody: app => `Öffnet ${app} in deinem Browser. Hermes sieht dein Passwort nie.`,
    signInWith: app => `Mit ${app} anmelden`,
    startsOff: 'Startet aus',
    startsOn: 'Startet an',
    toolPermissions: 'Werkzeugberechtigungen',
    tools: 'Werkzeuge',
    toolsCount: count => `${count} Werkzeug${count === 1 ? '' : 'e'}`,
    waiting: 'Warten auf Anmeldung',
    waitingFor: app => `Warten auf ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'KI, Medien & Präsentationen',
    'automation-ai-apis': 'Automatisierung, KI & APIs',
    'bi-search-warehouses': 'BI, Suche & Data Warehouses',
    'code-review-ci': 'Code-Review & CI',
    communication: 'Kommunikation',
    'customer-support': 'Kundensupport',
    'design-whiteboards': 'Design & Whiteboards',
    'docs-files': 'Dokumente & Dateien',
    'docs-knowledge': 'Dokumente & Wissen',
    'hiring-people': 'Personal & Recruiting',
    'hosting-databases': 'Hosting & Datenbanken',
    'marketing-seo': 'Marketing & SEO',
    'meetings-scheduling': 'Meetings & Termine',
    'monitoring-incidents': 'Monitoring & Vorfälle',
    'outdoors-fitness': 'Outdoor & Fitness',
    'payments-commerce': 'Zahlungen & Handel',
    'product-analytics': 'Produktanalysen',
    'projects-tasks': 'Projekte & Aufgaben',
    'sales-crm': 'Vertrieb & CRM',
    'spend-investing-crypto': 'Ausgaben, Investments & Krypto',
    travel: 'Reisen',
    'websites-assets': 'Websites & Assets'
  },
  toast: {
    connected: name => `${name} ist verbunden.`,
    disconnected: name => `${name} ist getrennt.`,
    removed: name => `${name} wurde entfernt.`,
    renamed: name => `Umbenannt in ${name}.`
  },
  toolFilters: {
    all: 'Alle',
    apply: 'Ergebnisse anzeigen',
    areas: 'Bereiche',
    behavior: 'Verhalten',
    clear: 'Alle löschen',
    close: 'Schließen',
    status: 'Status',
    title: 'Werkzeuge filtern',
    uncategorised: 'Ohne Kategorie'
  },
  tools: {
    allAreas: 'Alle Bereiche',
    areaChipLabel: 'Bereich',
    back: 'Zurück zu den Verbindungen',
    emptyBody: 'Diese App hat keine Werkzeuge freigegeben.',
    emptyTitle: 'Noch keine Werkzeuge',
    errorBody: 'Die Werkzeugliste konnte nicht gelesen werden. Versuche es erneut.',
    errorTitle: 'Werkzeuge konnten nicht geladen werden',
    filter: 'Filter',
    findArea: 'Bereich finden',
    loading: 'Werkzeuge werden geladen',
    lockedByOrg: 'Von deiner Organisation deaktiviert',
    noResultsBody: 'Zu dieser Suche oder diesen Filtern gibt es nichts.',
    noResultsTitle: 'Keine passenden Werkzeuge',
    off: 'Aus',
    on: 'An',
    pageSummary: (on, total, app) =>
      `${on} von ${total} Werkzeugen an. Änderungen werden zusammen gespeichert und gelten für jeden Agenten, der ${app} nutzt.`,
    pageTitle: app => `${app}-Werkzeuge`,
    permissionChipLabel: 'Berechtigung',
    searchAppTools: app => `${app}-Werkzeuge durchsuchen`,
    searchPlaceholder: 'Werkzeuge suchen',
    showTools: count => `${count} Werkzeug${count === 1 ? '' : 'e'} anzeigen`,
    subtitle: name => `Lege fest, was ${name} tun darf.`,
    summary: (on, total) => `${on} von ${total} an`,
    title: 'Alle Werkzeuge',
    toolNamePrefix: 'Werkzeugname:',
    unsavedChanges: count => `${count} nicht gespeicherte Änderung${count === 1 ? '' : 'en'}`,
    whatItDoes: 'Was es tut',
    whatItDoesHint: 'Wähle eines oder mehrere. „Alle“ löscht andere Auswahl.'
  }
}


export const es: ConnectionsCopy = {
  badges: { destructive: 'Destructiva', off: 'Desactivada', read: 'Lectura', write: 'Escritura' },
  bulk: {
    discard: 'Descartar',
    save: 'Guardar',
    someOn: 'Algunas activas',
    turnAllOff: 'Desactivar todo',
    turnAllOn: 'Activar todo'
  },
  card: {
    add: 'Añadir',
    connect: 'Conectar',
    connected: 'Conectada',
    disconnect: 'Desconectar',
    manage: 'Gestionar',
    menu: 'Más opciones',
    notConnected: 'No conectada',
    reconnect: 'Reconectar',
    refreshTools: 'Actualizar lista de herramientas',
    remove: 'Quitar',
    rename: 'Renombrar',
    toolsOff: count => `${count} herramienta${count === 1 ? '' : 's'} desactivada${count === 1 ? '' : 's'}`,
    toolsOn: count => `${count} herramienta${count === 1 ? '' : 's'} activada${count === 1 ? '' : 's'}`,
    toolsSomeOn: (on, total) => `${on} de ${total} herramientas activas`,
    waiting: 'Esperando el inicio de sesión'
  },
  categories: {
    'customers-growth': {
      name: 'Clientes y crecimiento',
      subtitle: 'Marketing, ventas y soporte.'
    },
    'data-analytics': {
      name: 'Datos y analítica',
      subtitle: 'Entiende tu producto y tus datos.'
    },
    'design-content': {
      name: 'Diseño y contenido',
      subtitle: 'Crea, diseña y publica.'
    },
    engineering: {
      name: 'Ingeniería',
      subtitle: 'Crea, aloja y supervisa.'
    },
    money: {
      name: 'Dinero',
      subtitle: 'Pagos, banca e inversiones.'
    },
    reference: {
      name: 'Referencia',
      subtitle: 'Consulta respuestas sin salir del chat.'
    },
    'travel-fitness': {
      name: 'Viajes y deporte',
      subtitle: 'Planifica viajes y sigue tus entrenamientos.'
    },
    'work-productivity': {
      name: 'Trabajo y productividad',
      subtitle: 'Documentos, proyectos, reuniones y las apps de tu equipo.'
    }
  },
  disconnect: {
    body: name => `${name} dejará de funcionar en el chat hasta que la vuelvas a conectar.`,
    cancel: 'Cancelar',
    confirm: 'Desconectar',
    title: name => `¿Desconectar ${name}?`
  },
  errors: {
    connectBody: 'Comprueba que terminaste de iniciar sesión y vuelve a intentarlo.',
    connectTitle: name => `No se pudo conectar ${name}`,
    dismiss: 'Descartar',
    loadBody: 'No se pudieron comprobar tus apps. Lo que ya estaba cargado sigue aquí.',
    loadTitle: 'No se pudieron cargar las conexiones',
    tryAgain: 'Reintentar'
  },
  facetTooltips: {
    destructive: 'Destructiva: puede borrar datos o deshacer trabajo.',
    read: 'Lectura: puede ver tus datos, pero no cambiarlos.',
    write: 'Escritura: puede crear y cambiar datos.'
  },
  filters: {
    activeFilters: count => `${count} filtro${count === 1 ? '' : 's'}`,
    addMcpServer: 'Añadir servidor MCP',
    all: 'Todas',
    allCategories: 'Todas las categorías',
    apply: 'Mostrar resultados',
    backToConnections: 'Conexiones',
    button: 'Filtros',
    categories: 'Categorías',
    categoryChipLabel: 'Categoría',
    chooseOne: 'Elige una',
    clear: 'Borrar todo',
    close: 'Cerrar',
    findCategory: 'Buscar una categoría',
    noResultsBody: 'Nada coincide con estos filtros. Bórralos para ver todas las apps.',
    noResultsQuery: query => `Nada coincide con «${query}». Prueba otra búsqueda o borra los filtros.`,
    noResultsTitle: 'No hay conexiones que coincidan',
    resultCount: count => `${count} conexión${count === 1 ? '' : 'es'}`,
    searchCategory: name => `Buscar ${name}`,
    selectedCount: count => `${count} seleccionada${count === 1 ? '' : 's'}`,
    showConnections: count => `Mostrar ${count} conexión${count === 1 ? '' : 'es'}`,
    status: 'Estado',
    statusAny: 'Todas',
    statusChipLabel: 'Estado',
    statusConnected: 'Conectadas',
    statusNotConnected: 'No conectadas',
    title: 'Filtros'
  },
  page: {
    subtitle: 'Las apps y herramientas que Hermes puede usar por ti.',
    title: 'Conexiones'
  },
  removeAccount: {
    body: name => `${name} se quitará de esta conexión. La app seguirá conectada con tus otras cuentas.`,
    cancel: 'Cancelar',
    confirm: 'Quitar',
    title: '¿Quitar esta cuenta?'
  },
  search: { clear: 'Borrar búsqueda', placeholder: 'Buscar conexiones' },
  sections: {
    connected: 'Conectadas',
    other: 'Otras apps',
    seeAll: count => `Ver las ${count}`,
    suggested: 'Sugerencias para ti'
  },
  sheet: {
    account: 'Cuenta',
    accounts: 'Cuentas',
    addAnAccount: '+ Añadir una cuenta',
    addAnotherAccount: '+ Añadir otra cuenta',
    availableSubtitle: 'En pausa para todos los agentes; sigue con la sesión iniciada.',
    availableToAgents: 'Disponible para los agentes',
    cancelConnection: 'Cancelar conexión',
    changeAfterConnect: 'Puedes cambiarlas después de conectar.',
    close: 'Cerrar',
    connected: 'Conectada',
    connectedOn: date => `Conectada el ${date}`,
    disconnectFailed: app => `No se pudo desconectar ${app}. Comprueba tu conexión y `,
    facetDescriptions: {
      destructive: 'Eliminación permanente',
      read: 'Solo lectura',
      write: 'Crear y editar'
    },
    finishSignIn: 'Terminar de iniciar sesión',
    finishSignInBody: app =>
      `Aprueba el acceso en la página de ${app} en tu navegador y luego vuelve aquí.`,
    notConnected: 'No conectada',
    permissionsTooltip: app =>
      `${app} decide si cada herramienta es de Lectura, Escritura o Destructiva. Tú eliges cuáles están activas.`,
    previewAll: count => `Ver las ${count}`,
    removeAccountBody: app =>
      `Los agentes ya no podrán usar esta cuenta de ${app}. Las demás cuentas seguirán conectadas.`,
    removeAccountTitle: alias => `¿Quitar «${alias}»?`,
    removeFailed: alias => `No se pudo quitar «${alias}». Comprueba tu conexión y `,
    removedSuccess: alias => `Se quitó «${alias}» correctamente`,
    renamePlaceholder: 'Nombre de la cuenta',
    reopenSignIn: 'Reabrir la página de inicio de sesión',
    reviewAll: count => `Revisar las ${count}`,
    reviewTools: 'Revisar todas las herramientas',
    signInBody: app => `Abre ${app} en tu navegador. Hermes nunca ve tu contraseña.`,
    signInWith: app => `Iniciar sesión con ${app}`,
    startsOff: 'Empieza desactivada',
    startsOn: 'Empieza activada',
    toolPermissions: 'Permisos de las herramientas',
    tools: 'Herramientas',
    toolsCount: count => `${count} herramienta${count === 1 ? '' : 's'}`,
    waiting: 'Esperando el inicio de sesión',
    waitingFor: app => `Esperando a ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'IA, medios y presentaciones',
    'automation-ai-apis': 'Automatización, IA y API',
    'bi-search-warehouses': 'BI, búsqueda y almacenes',
    'code-review-ci': 'Revisión de código e IC',
    communication: 'Comunicación',
    'customer-support': 'Atención al cliente',
    'design-whiteboards': 'Diseño y pizarras',
    'docs-files': 'Documentos y archivos',
    'docs-knowledge': 'Documentos y conocimiento',
    'hiring-people': 'Contratación y personas',
    'hosting-databases': 'Alojamiento y bases de datos',
    'marketing-seo': 'Marketing y SEO',
    'meetings-scheduling': 'Reuniones y agenda',
    'monitoring-incidents': 'Monitorización e incidencias',
    'outdoors-fitness': 'Aire libre y deporte',
    'payments-commerce': 'Pagos y comercio',
    'product-analytics': 'Analítica de producto',
    'projects-tasks': 'Proyectos y tareas',
    'sales-crm': 'Ventas y CRM',
    'spend-investing-crypto': 'Gastos, inversión y cripto',
    travel: 'Viajes',
    'websites-assets': 'Webs y recursos'
  },
  toast: {
    connected: name => `${name} está conectada.`,
    disconnected: name => `${name} está desconectada.`,
    removed: name => `Se quitó ${name}.`,
    renamed: name => `Renombrado a ${name}.`
  },
  toolFilters: {
    all: 'Todas',
    apply: 'Mostrar resultados',
    areas: 'Áreas',
    behavior: 'Comportamiento',
    clear: 'Borrar todo',
    close: 'Cerrar',
    status: 'Estado',
    title: 'Filtrar herramientas',
    uncategorised: 'Sin categoría'
  },
  tools: {
    allAreas: 'Todas las áreas',
    areaChipLabel: 'Área',
    back: 'Volver a conexiones',
    emptyBody: 'Esta app no ha compartido ninguna herramienta.',
    emptyTitle: 'Aún no hay herramientas',
    errorBody: 'No se pudo leer la lista de herramientas. Vuelve a intentarlo.',
    errorTitle: 'No se pudieron cargar las herramientas',
    filter: 'Filtros',
    findArea: 'Buscar un área',
    loading: 'Cargando herramientas',
    lockedByOrg: 'Desactivada por tu organización',
    noResultsBody: 'Nada coincide con esta búsqueda ni con estos filtros.',
    noResultsTitle: 'No hay herramientas que coincidan',
    off: 'Desactivada',
    on: 'Activada',
    pageSummary: (on, total, app) =>
      `${on} de ${total} herramientas activas. Los cambios se guardan juntos y se aplican a cada agente que use ${app}.`,
    pageTitle: app => `Herramientas de ${app}`,
    permissionChipLabel: 'Permiso',
    searchAppTools: app => `Buscar en las herramientas de ${app}`,
    searchPlaceholder: 'Buscar herramientas',
    showTools: count => `Mostrar ${count} herramienta${count === 1 ? '' : 's'}`,
    subtitle: name => `Elige qué puede hacer ${name}.`,
    summary: (on, total) => `${on} de ${total} activas`,
    title: 'Todas las herramientas',
    toolNamePrefix: 'Nombre de la herramienta:',
    unsavedChanges: count => `${count} cambio${count === 1 ? '' : 's'} sin guardar`,
    whatItDoes: 'Qué hace',
    whatItDoesHint: 'Elige una o varias. «Todas» borra las demás opciones.'
  }
}

