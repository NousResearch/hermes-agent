import type { Translations } from './types'

export const esQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'Omitir configuración',
  back: 'Atrás',
  trailLabel: 'Tus respuestas',
  changeAnswer: 'Cambiar esta respuesta',
  retry: 'Reintentar',
  saveFailed: 'No se pudo marcar la configuración como terminada',
  skipped: 'Omitido',
  otherLabel: 'Otra respuesta',
  kinds: {
    accent: 'Color de acento',
    layout: 'Diseño',
    local: 'Modelo local',
    apps: 'Apps en este equipo',
    connectors: 'Conectores'
  },
  name: {
    greeting: 'Hola, soy Hermes.',
    question: '¿Cómo quieres que te llame?',
    placeholder: 'Tu nombre'
  },
  accent: {
    title: 'Elige un color',
    custom: 'Color personalizado'
  },
  layout: {
    title: '¿Cómo quieres trabajar?',
    basic: 'Básico',
    basicDetail: 'Para hablar con Hermes.',
    elite: 'Elite',
    eliteDetail: 'Para desarrolladores: terminal, archivos, diffs.'
  },
  local: {
    titleSpark: 'Tu Spark puede ejecutar Hermes sin conexión',
    title: kind => `Este ${kind} puede ejecutar un modelo en local`,
    download: model => `Descargar ${model}`,
    downloadDetail: 'Se descarga en segundo plano cuando empiece tu primer chat.',
    notNow: 'Ahora no',
    notNowDetail: 'Queda en el menú de modelos como «Ejecutar en local».',
    chip: 'Chip',
    gpu: 'GPU',
    memory: 'Memoria',
    os: 'SO',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: '¿Usar Hermes dentro de estas apps?',
    plugin: 'Plugin',
    pluginNotRunning: 'Plugin · la app no está abierta',
    approve: 'Apruebas cada instalación en tu primer chat.',
    found: 'Encontradas en este equipo. Las apps no instaladas no aparecen.'
  },
  connectors: {
    title: '¿Conectar tus apps?',
    checking: 'Comprobando qué apps puede conectar tu cuenta…',
    offered: 'Esta es la lista que ofrece tu cuenta ahora mismo. Inicias sesión en cada una en tu primer chat.'
  },
  task: {
    title: '¿Qué hacemos primero?',
    question: 'Elige una o escribe la tuya.',
    blender: 'Crear una escena en Blender',
    nvidia: 'Ajustar mi GPU',
    broadcast: 'Configurar mi micro y cámara',
    brief: 'Resumen del día',
    tidy: 'Ordenar mis Descargas'
  },
  tour: {
    title: '¿Quieres un recorrido?',
    question: '¿Echamos un vistazo después?',
    quick: 'Enséñame',
    quickDetail: 'Unas pocas paradas, menos de un minuto.',
    none: 'Ya me apaño',
    noneDetail: 'Puedes pedirle un recorrido a Hermes más tarde.'
  },
  start: {
    title: name => (name ? `Listo, ${name}.` : 'Listo.'),
    sub: 'Este es el primer mensaje que recibe Hermes. La petición va primero; el resto queda debajo.',
    aboutMe: 'Sobre mí:',
    beforeYouStart: 'Antes de empezar:',
    noTask: 'Sin tarea elegida: Hermes te preguntará qué quieres hacer.',
    action: 'Empezar',
    waiting: 'Preparando tu cuenta gratuita…',
    failed: 'No se pudo iniciar tu primer chat'
  },
  summary: {
    custom: 'Personalizado',
    local: model => `Local: ${model}`,
    localNo: 'Local: ahora no',
    noApps: 'Sin apps',
    noConnectors: 'Sin conectores',
    more: (label, count) => `${label} +${count}`,
    tour: 'Recorrido',
    noTour: 'Sin recorrido'
  },
  status: {
    settingUp: 'Configurando tu cuenta gratuita',
    stillSettingUp: 'Aún configurando tu cuenta gratuita',
    unavailable: 'Cuenta gratuita no disponible',
    ready: 'Nous · plan gratuito',
    downloadingEngine: 'Descargando el motor local',
    downloading: model => `Descargando ${model}`
  },
  settings: {
    title: 'Repetir la configuración',
    description: 'Vuelve a abrir las preguntas iniciales. Tus respuestas se aplican al perfil predeterminado.',
    action: 'Configurar',
    failed: 'No se pudo iniciar la configuración'
  }
}
