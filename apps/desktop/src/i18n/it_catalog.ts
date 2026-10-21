import type { CatalogTranslations } from './types_catalog'

export const itCatalog: CatalogTranslations = {
  about: 'Informazioni',
  add: 'Aggiungi',
  added: 'Aggiunto',
  addedDate: 'Aggiunto il',
  allCategories: 'Tutte le categorie',
  allSources: 'Tutte le fonti',
  alphabetical: 'Nome',
  author: 'Autore',
  back: 'Torna ai risultati',
  cardView: 'Vista schede',
  category: 'Categoria',
  clearFilters: 'Cancella i filtri',
  commands: 'Comandi',
  destinationChanged: 'La destinazione è cambiata. Chiudi questa finestra e apri di nuovo il link di installazione.',
  developer: {
    docs: 'Documentazione per sviluppatori',
    heading: 'Modalità sviluppatore',
    pitch: 'Hai costruito uno strumento, un connettore, un plugin o una Mod?',
    pitchFollowUp: 'Vuoi farlo ma non sai come?',
    submit: 'Invia una Mod',
    title: 'Crea per Hermes'
  },
  discover: 'Scopri',
  discoverKind: kind => `Scopri ${kind}`,
  documentation: 'Documentazione',
  explorePlugins: 'Esplora i plugin',
  exploreSkills: 'Esplora le skill',
  featured: 'In evidenza',
  filters: 'Filtri',
  hooks: 'Hook',
  installComplete: name => `“${name}” installato`,
  installDescription: 'Questa skill sarà disponibile nelle nuove sessioni. Installa solo da fonti affidabili.',
  installHint: 'Controlla il sorgente prima di installare. Le modifiche si applicano alle nuove sessioni.',
  installTitle: name => `Installare “${name}”?`,
  installTo: 'Installa in',
  installed: 'Installato',
  installing: 'Installazione…',
  license: 'Licenza',
  listView: 'Vista elenco',
  loadFailed: 'Impossibile caricare il catalogo',
  middleware: 'Middleware',
  more: 'Mostra di più',
  moreToExplore: 'Altro da esplorare',
  moreToExploreBlurb: 'Gli angoli più piccoli del catalogo, tutti in un solo posto.',
  mostStarred: 'Più stellati',
  newest: 'Più recenti',
  nextSlide: 'Successivo in evidenza',
  noResults: 'Nessuna corrispondenza',
  pinned: 'Commit verificato',
  platforms: 'Piattaforme',
  previousSlide: 'Precedente in evidenza',
  recentlyUpdated: 'Aggiornati di recente',
  related: 'Simili a questo',
  repository: 'Repository',
  requires: 'Richiede',
  results: count => `${count.toLocaleString()} ${count === 1 ? 'risultato' : 'risultati'}`,
  retry: 'Riprova',
  screenshots: 'Screenshot',
  searchPlugins: 'Cerca plugin',
  searchSkills: 'Cerca skill',
  seeAll: 'Vedi tutti',
  skillsFor: category => `Skill per ${category.toLowerCase()}.`,
  slideOf: (index, count) => `${index} di ${count}`,
  snapshotHint: 'Dal catalogo di Hermes. Esplorare non contatta mai i repository sorgente.',
  sortBy: 'Ordina per',
  source: 'Fonte',
  tags: 'Tag',
  thisComputer: 'Questo computer',
  tier: 'Livello',
  tierLabels: {
    community: 'Community',
    official: 'Ufficiale',
    verified: 'Verificato'
  },
  tiers: {
    official: {
      blurb: 'Costruiti e mantenuti da Nous Research.',
      title: 'Plugin ufficiali'
    },
    verified: {
      blurb: 'Dall’azienda che sta dietro all’integrazione, verificati da Nous.',
      title: 'Plugin verificati'
    }
  },
  tools: 'Strumenti',
  toolsets: {
    callCount: (count, calls) => `${count} ${calls === 1 ? 'chiamata' : 'chiamate'}`,
    essentials: 'Essenziali',
    essentialsBlurb: 'Su cosa gira la maggior parte del lavoro con Hermes.',
    mostUsed: 'Più usati',
    mostUsedBlurb: 'I set di strumenti che Hermes usa di più su questo profilo.',
    themes: {
      connect: {
        blurb: 'Account, altri agenti, la casa intelligente e la ricerca oltre il web.',
        label: 'Connetti e raggiungi'
      },
      create: {
        blurb: 'Guarda immagini e video, trascrivi la voce e genera contenuti.',
        label: 'Vedi, ascolta e crea'
      },
      more: {
        blurb: 'Tutto il resto che questo profilo può attivare.',
        label: 'Altri strumenti'
      },
      recall: {
        blurb: 'Ciò che Hermes ricorda e le skill e la cronologia che può cercare.',
        label: 'Memoria e conoscenza'
      },
      work: {
        blurb: 'Guida browser e desktop, modifica file, esegui codice e pianifica lavori.',
        label: 'Sfoglia, lavora e automatizza'
      }
    },
    toolCount: count => `${count} ${count === 1 ? 'strumento' : 'strumenti'}`
  },
  tryAnother: 'Prova un’altra ricerca o cancella i filtri.',
  updatedDate: 'Aggiornato',
  version: 'Versione'
}
