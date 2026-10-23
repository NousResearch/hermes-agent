import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const itDelegationRouting: DelegationRoutingTranslations = {
  title: 'Routing',
  settingsTitle: 'Predefiniti dei subagenti',
  intro: 'Come ogni subagente ottiene il proprio modello.',
  modes: {
    default: { label: 'Predefinito', description: 'Ogni subagente usa il modello subagente qui sotto.' },
    model_map: {
      label: 'Route di modelli',
      description: 'Nomina alcuni modelli. L’agente ne sceglie uno per ogni subagente.'
    },
    profile_routes: {
      label: 'Route di profili',
      description: 'I subagenti girano come uno dei tuoi profili, con il suo modello, le sue skill e le sue chiavi.'
    }
  },
  inUse: 'In uso',
  plugin: 'Plugin',
  pluginMissing: 'Le route di modelli e le route di profili arrivano dal plugin Advanced Delegation.',
  pluginDisabled: 'Advanced Delegation è installato ma disattivato per questo profilo.',
  pluginNotLoaded:
    'Advanced Delegation è attivo, ma Hermes non lo ha ancora caricato. Riavvia Hermes per completare.',
  installPlugin: 'Installa plugin',
  installUnavailable:
    'Advanced Delegation non è ancora nel catalogo dei plugin. Installalo da Capacità → Plugin.',
  enablePlugin: 'Attiva',
  enableFailed: 'Impossibile attivare Advanced Delegation',
  modeFailed: 'Impossibile cambiare il routing dei subagenti',
  loadFailed: 'Impossibile caricare il routing dei subagenti',
  blockedPill: 'Non in esecuzione',
  blockedDetail:
    'Questa modalità è salvata, quindi Hermes rifiuta ogni subagente invece di usare in silenzio Predefinito.',
  blockedMissing: 'I subagenti non si avvieranno finché Advanced Delegation non viene installato.',
  blockedDisabled: 'I subagenti non si avvieranno finché Advanced Delegation non viene attivato per questo profilo.',
  blockedNotLoaded:
    'I subagenti non si avvieranno finché Hermes non carica Advanced Delegation. Riavvia Hermes per completare.',
  blockedUnavailable: label =>
    `I subagenti non si avvieranno finché ${label} non potrà girare qui. Il suo plugin segnala che non può.`,
  blockedUnregistered: name =>
    `I subagenti non si avvieranno finché non viene installato e attivato un plugin che fornisce “${name}”.`,
  thirdPartyDescription: 'Da un plugin. Le sue impostazioni stanno con quel plugin.',

  modelRoutesTitle: 'Route di modelli',
  modelRoutesEmpty:
    'Ancora nessuna route. Finché non ne aggiungi una, i subagenti usano il modello dell’agente principale.',
  addRoute: 'Aggiungi route',
  editRoute: 'Modifica',
  makeDefault: 'Rendi predefinita',
  deleteRoute: 'Elimina',
  defaultBadge: 'Predefinita',
  routeActions: name => `Azioni per ${name}`,
  deleteConfirmTitle: name => `Eliminare la route “${name}”?`,
  deleteConfirmBody: 'I subagenti non potranno più sceglierla.',
  deleteConfirmNextDefault: name => `È la route predefinita, quindi “${name}” diventa la predefinita.`,
  deleteConfirmAction: 'Elimina route',
  saveFailed: 'Impossibile salvare le route',
  baseUrlRoute: url => `endpoint ${url}`,

  dialogNew: 'Nuova route di modelli',
  dialogEdit: 'Modifica route di modelli',
  dialogDesc: 'L’agente legge nome e descrizione per decidere quale route prende un subagente.',
  nameLabel: 'Nome',
  namePlaceholder: 'es. veloce',
  nameHint: 'Breve, senza spazi. L’agente sceglie le route da questo nome.',
  descriptionLabel: 'Quando usarla',
  descriptionPlaceholder: 'es. Economico e rapido: ricerche, riassunti, modifiche semplici.',
  descriptionHint: 'Una riga che l’agente legge prima di scegliere.',
  providerLabel: 'Provider',
  providerPlaceholder: 'Scegli un provider',
  modelLabel: 'Modello',
  effortLabel: 'Sforzo di ragionamento',
  effortInherit: 'Predefinito del subagente',
  tierLabel: 'Velocità',
  tierNormal: 'Normale',
  tierFast: 'Veloce',
  tierUltrafast: 'Ultrarapida',
  tierHint: 'Solo sui modelli che la offrono.',
  optional: 'opzionale',
  errors: { required: 'Obbligatorio', spaces: 'Niente spazi', duplicate: 'Un’altra route usa già questo nome' },
  saveRoute: 'Salva route',
  createRoute: 'Aggiungi route',

  profileRoutesTitle: 'Profili in cui i subagenti possono girare',
  profileRoutesHint: 'Ogni profilo spuntato porta con sé il proprio modello, skill, personaggio e chiavi.',
  profileDescriptionPlaceholder: 'Quando usarlo (opzionale)',
  profileDefault: 'Predefinito',
  profileEmpty: 'Ancora nessun altro profilo. Creane uno dalla barra dei profili.',
  profileMissing: name => `${name} (il profilo non esiste più)`,

  inactiveTitle: 'Non usato in questa modalità',
  inactiveModelMap: route =>
    `Le route di modelli scelgono il modello di ogni subagente. I subagenti che non scelgono una route usano “${route}”.`,
  inactiveProfiles: profile =>
    `Le route di profili scelgono il profilo di ogni subagente. I subagenti che non scelgono un profilo girano come “${profile}”.`,
  inactiveModelMapNoDefault:
    'Le route di modelli scelgono il modello di ogni subagente. I subagenti che non scelgono una route usano il modello dell’agente principale.',
  inactiveProfilesNoDefault:
    'Le route di profili scelgono il profilo di ogni subagente. I subagenti che non scelgono un profilo usano il modello dell’agente principale.',
  inactivePlugin: label => `${label} decide il modello di ogni subagente, quindi questa impostazione non si applica.`,
  useDefault: 'Passa a Predefinito'
}
