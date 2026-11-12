import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_ES = {
  group: {
    ...GROUP_CHAT_LOCALES.es,
    newTitle: 'Nuevo chat de grupo',
    manageDesc: 'Un bot puede unirse a varios chats de grupo. Las membresías se sincronizan en todos los equipos.',
    manageTitle: 'Administrar grupos',
    settingsTitle: 'Ajustes del grupo',
    settingsDesc: 'Cambia el nombre o la imagen del grupo. Se conservan los miembros y el historial.',
    nameLabel: 'Nombre del grupo',
    holdDetection: 'Detectar instrucciones de parada',
    holdDetectionHint:
      'Las instrucciones de parada en el chat de grupo pausan a los miembros indicados hasta que se los vuelva a mencionar.',
    compressHistory: 'Comprimir historial',
    compressHistoryHint: member =>
      `Comprimir el historial oculto de ${member} en el chat de grupo para evitar respuestas vacías`,
    compressing: member => `Comprimiendo el historial de ${member} en el chat de grupo…`,
    compressDone: (member, compressed, detail) =>
      `${compressed} ${compressed === 1 ? 'sesión comprimida' : 'sesiones comprimidas'} del chat de grupo para ${member}${detail ? ` — ${detail}` : ''}`,
    compressNothing: member => `Nada que comprimir para ${member} — aún no hay sesiones de chat de grupo`,
    compressFailed: (member, error) => `No se pudo comprimir el historial de ${member}: ${error}`,
    searchToAdd: 'Buscar bots para añadir',
    searchToAddPlaceholder: 'Buscar bots para añadir…',
    removeFromSelection: 'Quitar de la selección',
    disbandTitle: '¿Disolver el chat de grupo?',
    deleteTitle: '¿Eliminar el chat de grupo?',
    deleteAction: 'Eliminar',
    composerPlaceholder: 'Escribe algo — todos los bots de este chat de grupo verán tu mensaje.',
    slashCommandsUnsupported:
      'Los comandos con barra no están disponibles en los chats de grupo. Abre un chat individual con un bot para usarlos.',
    attachHint: 'Adjuntar archivos — todos los bots que respondan los verán',
    newThread: 'Nuevo hilo',
    reply: 'Responder',
    replyInThread: 'Responder en el hilo',
    replyInThreadPlaceholder: 'Responder en el hilo…',
    openThread: 'Abrir este hilo',
    collapseThread: 'Contraer hilo',
    collapseThreadLabel: 'Contraer este hilo',
    activity: 'Actividad',
    noActivityYet: 'Aún no hay actividad en este turno.',
    showActivity: 'Mostrar la actividad del chat de grupo',
    hideActivity: 'Ocultar la actividad del chat de grupo',
    stop: 'Detener',
    stopHint: 'Detener esta ejecución — interrumpe al miembro activo y pausa al resto',
    heldMembersStatus: members => `En pausa: ${members}`,
    holdReleaseHint: 'Menciona a un bot en pausa o envía @all resume para continuar.',
    needsYourInput: 'Un bot de este chat de grupo necesita tu respuesta',
    noMembersToSend: group =>
      `${group} no tiene destinatarios — añade un bot o vuelve a abrir el chat de grupo si aún se están cargando los miembros.`,
    pictureGenerationFailed: 'No se pudo generar la imagen del grupo',
    nameTaken: name => `Ya existe un grupo llamado «${name}».`,
    you: 'Tú',
    settingsHint: group => `Ajustes del grupo — cambia el nombre de ${group} o la imagen del grupo`,
    settingsLabel: group => `Ajustes de ${group}`,
    disbandHint: group => `Disolver el chat de grupo ${group}`,
    disbandLabel: group => `Disolver ${group}`,
    disbandAction: 'Disolver',
    disbanding: 'Disolviendo…',
    disbandDone: 'Disuelto',
    disbanded: group => `«${group}» disuelto`,
    stopped: group => `${group} detenido — los turnos restantes quedan en pausa hasta que reanudes`,
    removeAttachment: 'Quitar adjunto',
    threadFallback: 'Hilo',
    replyCount: replies => `${replies} respuesta${replies === 1 ? '' : 's'}`,
    dropToThread: 'Suelta para adjuntar a esta respuesta',
    dropToRoom: 'Suelta para adjuntar — todos los bots que respondan lo verán',
    waitingForAnswer: 'Esperando tu respuesta…',
    memberThinking: name => `${name} está pensando…`,
    roomWorking: 'El chat de grupo está trabajando…',
    messageRoom: group => `Enviar un mensaje a ${group}`,
    newThreadPlaceholder: group => `Nuevo hilo en ${group}… (@nombre para un miembro, @everyone para todos)`,
    everyoneMeta: 'Todos los bots del chat de grupo',
    commandApproval: 'aprobación de comando',
    answerFailed: (handle, error) => `No se pudo enviar la respuesta a @${handle}: ${error}`,
    wantsToRunCommand: handle => `@${handle} quiere ejecutar un comando:`,
    asks: handle => `@${handle} pregunta:`,
    answerTo: member => `Responder a @${member}`
  },
  roster: {
    ...ROSTER_CHROME.es,
    search: 'Buscar bots y chats de grupo',
    searchPlaceholder: 'Buscar bots y chats de grupo…',
    newBotOrGroup: 'Nuevo bot o chat de grupo',
    groupChats: 'Chats de grupo',
    emptyTitle: 'Aún no hay bots',
    emptyDesc: 'Crea tu primer bot.',
    noMatchQuery: query => `Ningún bot ni chat de grupo coincide con «${query}»`,
    noMatchQueryOn: (query, gateway) => `Ningún bot ni chat de grupo coincide con «${query}» en ${gateway}`,
    noMatchFiltersOn: gateway => `Ningún bot ni chat de grupo coincide con estos filtros en ${gateway}`,
    noMatchFilters: 'Ningún bot ni chat de grupo coincide con estos filtros.',
    clearFilters: 'Borrar filtros',
    allHidden: 'Todos los bots están ocultos',
    allHiddenDesc: 'Siguen trabajando y conservan su historial.',
    showHidden: 'Mostrar bots ocultos',
    noHiddenMatch: 'Ningún bot oculto coincide con estos filtros.',
    hiddenFromRoster: 'Oculto en la lista',
    pinned: 'Fijado',
    needsAttention: 'necesita atención',
    needsInput: 'Necesita tu respuesta',
    botsAndGroups: 'Bots y chats de grupo',
    botsOnly: 'Solo bots',
    groupsOnly: 'Solo chats de grupo',
    anyActivity: 'Cualquier actividad',
    activeNow: 'Activos ahora',
    recentlyActive: 'Activos recientemente',
    older: 'Anteriores',
    gatewayRemoved: 'Pasarela eliminada',
    onDemand: 'Bajo demanda',
    ready: 'Listo',
    statusUnknown: 'Estado desconocido',
    unavailable: 'No disponible',
    retryNow: 'Reintentar ahora',
    rosterUnavailable: reason =>
      `Lista no disponible: ${reason}. Si tu pasarela no admite profiles.list, actualiza Hermes y reinicia la pasarela.`,
    waitingForGateway:
      'Esperando la conexión con la pasarela… (las pasarelas remotas pueden tardar unos segundos; se reintenta automáticamente)'
  },
  sections: {
    newSection: 'Nueva sección',
    newTitle: 'Nueva sección',
    renameTitle: 'Renombrar sección',
    nameLabel: 'Nombre de la sección',
    namePlaceholder: 'P. ej., Clientes',
    create: 'Crear',
    rename: 'Renombrar…',
    moveUp: 'Subir',
    moveDown: 'Bajar',
    unassigned: 'Sin sección',
    options: name => `Opciones de la sección ${name}`,
    headingTip: 'Suelta bots aquí · doble clic para renombrar',
    emptyHint: 'Arrastra bots aquí',
    moveTo: 'Mover a una sección',
    newSectionEllipsis: 'Nueva sección…',
    removeFromSection: 'Quitar de la sección',
    deleted: (name, count) =>
      count === 0
        ? `Sección «${name}» eliminada`
        : `Sección «${name}» eliminada — ${count} bot${count === 1 ? '' : 's'} sin sección`,
    undo: 'Deshacer'
  },
  bot: {
    newChatWith: 'Nuevo chat con este bot',
    newTitle: 'Nuevo bot',
    thisDevice: 'Este dispositivo',
    removeFromAllGroups: 'Quitar de todos los grupos',
    openBotChat: 'Abrir chat con el bot',
    openBotChatInNewTab: 'Abrir chat con el bot en una pestaña nueva',
    openingChat: 'Abriendo chat…',
    pinToTop: 'Fijar arriba',
    unpin: 'Desfijar',
    pinnedToast: name => `${name} fijado arriba`,
    unpinnedToast: name => `${name} desfijado`,
    hide: 'Ocultar',
    unhide: 'Mostrar',
    hiddenToast: name => `${name} oculto — usa el botón del ojo en la lista de bots para verlo`,
    unhiddenToast: name => `${name} vuelve a la lista`,
    groupsMenu: groups => `Grupos: ${groups}…`,
    manageGroups: 'Administrar grupos…',
    metadataLoadFailed: 'No se pudieron cargar los datos del bot',
    loadFailed: 'No se pudo cargar el bot',
    groupsLoadFailed: 'No se pudieron cargar los grupos del bot',
    attentionFallback: 'Necesita atención',
    attentionProviderAuth: 'Vuelve a iniciar sesión con este perfil',
    attentionQuota: 'Cuota o saldo agotado',
    attentionMissingConfig: 'Proveedor sin configurar — ejecuta hermes model',
    attentionBlocked: 'Bot bloqueado — consulta su último mensaje',
    editMenu: 'Editar…',
    duplicate: 'Duplicar',
    duplicateFailed: 'Error al duplicar'
  },
  avatar: {
    upload: 'Subir',
    generate: 'Generar',
    generating: 'Generando…'
  },
  screen: {
    autoOpenMenu: 'Abrir la pantalla cuando el bot la use',
    autoOpenOnToast: name => `La pantalla de ${name} se abre cuando el bot empieza a usar su escritorio`,
    autoOpenOffToast: name => `La pantalla de ${name} permanece cerrada hasta que la abras`,
    menu: 'Abrir pantalla'
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
