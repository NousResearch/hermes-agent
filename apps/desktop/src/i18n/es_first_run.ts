import type { FirstRunCopy } from './types_first_run'

export const esFirstRun: FirstRunCopy = {
  guide: {
    title: 'Configura Hermes',
    resume: 'Reanudar la configuración',
    upgradeRequired:
      'Hermes en esta máquina necesita una actualización para configurar sin sesión. Ejecuta hermes update en esa máquina, reinicia Hermes y vuelve a intentarlo. Tus chats no han cambiado.',
    intro: 'Unas preferencias opcionales y luego tu chat habitual. La configuración nunca crea una conversación.',
    appearance: 'Apariencia',
    model: 'Elige un modelo',
    ready: 'Tu modelo está listo.',
    providerNeeded: 'Conecta un proveedor o usa el modelo gratuito antes de enviar un mensaje.',
    freeModel: 'Usar el modelo gratuito',
    freeUnavailable: 'El modelo gratuito no está disponible aquí. Conecta un proveedor o elige uno más tarde.',
    providers: 'Conectar un proveedor',
    finish: 'Finalizar la configuración',
    skip: 'Omitir la configuración',
    back: 'Atrás',
    check: 'Comprobar disponibilidad',
    checking: 'Comprobando…',
    privacy: 'Compartir métricas de uso es opcional.'
  },
  beats: {
    name: 'Nombre',
    theme: 'Color de acento',
    layout: 'Diseño',
    connectors: 'Apps',
    plugins: 'Plugins',
    tour: 'Recorrido',
    build: 'Primera tarea'
  },
  namePlaceholder: 'Tu nombre',
  useName: name => `Usar «${name}»`,
  install: 'Instalar',
  installed: 'Instalado',
  loading: 'Cargando…',
  noPlugins: 'No hay plugins para mostrar ahora mismo.',
  noneChosen: 'Ninguno',
  noAnswer: 'Sin respuesta',
  answerRefused: 'Respuesta no aceptada',
  tours: {
    quick: 'Recorrido rápido',
    full: 'Muéstramelo todo',
    skip: 'Omitir, construyamos algo'
  },
  skippedAll: 'Configuración omitida',
  skip: 'Omitir configuración',
  skipTip: 'Termina la configuración guiada y abre un chat nuevo.',
  skipFailed: 'No se pudo omitir la configuración',
  slashOffTitle: 'Los comandos con barra no están disponibles durante la configuración',
  slashOffBody: 'Funcionarán con normalidad cuando termine la configuración.',
  localModel: {
    title: 'Este ordenador puede ejecutar un modelo localmente',
    text: (model, size) =>
      `${model} (${size}) encaja con tu hardware. Funciona gratis y tus chats se quedan en este ordenador.`,
    setUp: 'Configurarlo',
    notNow: 'Ahora no'
  },
  intro: {
    greeting: 'Hola, soy Hermes.',
    plan: 'Vamos a dejarlo todo listo para ti. Luego haremos algo genial.'
  },
  rerun: {
    title: 'Repetir la configuración inicial',
    description: 'Abre un nuevo chat de configuración que recorre los primeros pasos con tu modelo actual.',
    action: 'Repetir',
    failed: 'No se pudo iniciar la configuración inicial'
  }
}
