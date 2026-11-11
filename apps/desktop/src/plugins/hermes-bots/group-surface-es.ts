import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_ES = {
  group: {
    ...GROUP_CHAT_LOCALES.es,
    newTitle: 'Nuevo chat de grupo',
    manageDesc: 'Un agente puede unirse a varios chats de grupo. Las membresías se sincronizan en todos los equipos.',
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
    searchToAdd: 'Buscar agentes para añadir',
    searchToAddPlaceholder: 'Buscar agentes para añadir…',
    removeFromSelection: 'Quitar de la selección',
    disbandTitle: '¿Disolver el chat de grupo?',
    deleteTitle: '¿Eliminar el chat de grupo?',
    deleteAction: 'Eliminar',
    composerPlaceholder: 'Escribe algo — todos los agentes de este chat de grupo verán tu mensaje.',
    slashCommandsUnsupported:
      'Los comandos con barra no están disponibles en los chats de grupo. Abre un chat individual con un agente para usarlos.',
    attachHint: 'Adjuntar archivos — todos los agentes que respondan los verán',
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
    holdReleaseHint: 'Menciona a un agente en pausa o envía @all resume para continuar.',
    needsYourInput: 'Un agente de este chat de grupo necesita tu respuesta',
    noMembersToSend: group =>
      `${group} no tiene destinatarios — añade un agente o vuelve a abrir el chat de grupo si aún se están cargando los miembros.`,
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
    dropToRoom: 'Suelta para adjuntar — todos los agentes que respondan lo verán',
    waitingForAnswer: 'Esperando tu respuesta…',
    memberThinking: name => `${name} está pensando…`,
    roomWorking: 'El chat de grupo está trabajando…',
    messageRoom: group => `Enviar un mensaje a ${group}`,
    newThreadPlaceholder: group => `Nuevo hilo en ${group}… (@nombre para un miembro, @everyone para todos)`,
    everyoneMeta: 'Todos los agentes del chat de grupo',
    commandApproval: 'aprobación de comando',
    answerFailed: (handle, error) => `No se pudo enviar la respuesta a @${handle}: ${error}`,
    wantsToRunCommand: handle => `@${handle} quiere ejecutar un comando:`,
    asks: handle => `@${handle} pregunta:`,
    answerTo: member => `Responder a @${member}`
  },
  roster: {
    ...ROSTER_CHROME.es,
    search: 'Buscar agentes y chats de grupo',
    searchPlaceholder: 'Buscar agentes y chats de grupo…',
    newBotOrGroup: 'Nuevo agente o chat de grupo',
    groupChats: 'Chats de grupo',
    emptyTitle: 'Aún no hay agentes',
    emptyDesc: 'Crea tu primer agente.',
    noMatchQuery: query => `Ningún agente ni chat de grupo coincide con «${query}»`,
    noMatchQueryOn: (query, gateway) => `Ningún agente ni chat de grupo coincide con «${query}» en ${gateway}`,
    noMatchFiltersOn: gateway => `Ningún agente ni chat de grupo coincide con estos filtros en ${gateway}`,
    noMatchFilters: 'Ningún agente ni chat de grupo coincide con estos filtros.',
    clearFilters: 'Borrar filtros',
    allHidden: 'Todos los agentes están ocultos',
    allHiddenDesc: 'Siguen trabajando y conservan su historial.',
    showHidden: 'Mostrar agentes ocultos',
    noHiddenMatch: 'Ningún agente oculto coincide con estos filtros.',
    hiddenFromRoster: 'Oculto en la lista',
    pinned: 'Fijado',
    needsAttention: 'necesita atención',
    needsInput: 'Necesita tu respuesta',
    botsAndGroups: 'Agentes y chats de grupo',
    botsOnly: 'Solo agentes',
    groupsOnly: 'Solo chats de grupo',
    anyActivity: 'Cualquier actividad',
    activeNow: 'Activos ahora',
    recentlyActive: 'Activos recientemente',
    older: 'Anteriores',
    gatewayRemoved: 'Máquina eliminada',
    onDemand: 'Bajo demanda',
    ready: 'Listo',
    statusUnknown: 'Estado desconocido',
    unavailable: 'No disponible',
    retryNow: 'Reintentar ahora',
    rosterUnavailable: reason =>
      `Lista no disponible: ${reason}. Si tu versión de Hermes no admite profiles.list, actualiza y reinicia Hermes.`,
    waitingForGateway:
      'Esperando la conexión con la máquina… (las máquinas remotas pueden tardar unos segundos; se reintenta automáticamente)'
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
    headingTip: 'Suelta agentes aquí · doble clic para renombrar',
    emptyHint: 'Arrastra agentes aquí',
    moveTo: 'Mover a una sección',
    newSectionEllipsis: 'Nueva sección…',
    removeFromSection: 'Quitar de la sección',
    deleted: (name, count) =>
      count === 0
        ? `Sección «${name}» eliminada`
        : `Sección «${name}» eliminada — ${count} agente${count === 1 ? '' : 's'} sin sección`,
    undo: 'Deshacer'
  },
  bot: {
    newChatWith: 'Nuevo chat con este agente',
    newTitle: 'Nuevo agente',
    thisDevice: 'Este dispositivo',
    removeFromAllGroups: 'Quitar de todos los grupos',
    openBotChat: 'Abrir chat con el agente',
    openBotChatInNewTab: 'Abrir chat con el agente en una pestaña nueva',
    openingChat: 'Abriendo chat…',
    pinToTop: 'Fijar arriba',
    unpin: 'Desfijar',
    pinnedToast: name => `${name} fijado arriba`,
    unpinnedToast: name => `${name} desfijado`,
    hide: 'Ocultar',
    unhide: 'Mostrar',
    hiddenToast: name => `${name} oculto — usa el botón del ojo en la lista de agentes para verlo`,
    unhiddenToast: name => `${name} vuelve a la lista`,
    groupsMenu: groups => `Grupos: ${groups}…`,
    manageGroups: 'Administrar grupos…',
    metadataLoadFailed: 'No se pudieron cargar los datos del agente',
    loadFailed: 'No se pudo cargar el agente',
    groupsLoadFailed: 'No se pudieron cargar los grupos del agente',
    attentionFallback: 'Necesita atención',
    attentionProviderAuth: 'Vuelve a iniciar sesión con este agente',
    attentionQuota: 'Cuota o saldo agotado',
    attentionMissingConfig: 'Proveedor sin configurar — ejecuta hermes model',
    attentionBlocked: 'Agente bloqueado — consulta su último mensaje',
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
    autoOpenMenu: 'Abrir la pantalla cuando el agente la use',
    autoOpenOnToast: name => `La pantalla de ${name} se abre cuando el agente empieza a usar su escritorio`,
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
