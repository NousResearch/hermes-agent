import type { TranslationOverrides } from './define-locale'

export const esOnboarding: TranslationOverrides['onboarding'] = {
  headerTitle: 'Vamos a configurar Hermes Agent',
  headerDesc: 'Conecta un proveedor de modelo para empezar a chatear. La mayoría de opciones requieren un clic.',
  preparingInstall: 'Hermes está terminando la instalación. En el primer inicio suele tardar menos de un minuto.',
  starting: 'Iniciando Hermes…',
  setupSlowTitle: 'La configuración está tardando más de lo habitual.',
  setupSlowBody: 'Hermes sigue iniciándose en segundo plano.',
  continueWithoutSetup: 'Continuar sin configurar',
  lookingUpProviders: 'Buscando proveedores...',
  collapse: 'Contraer',
  otherProviders: 'Otros proveedores',
  haveApiKey: 'Tengo una clave API',
  chooseLater: 'Elegiré un proveedor más tarde',
  recommended: 'Recomendado',
  connected: 'Conectado',
  featuredPitch: 'Una suscripción, más de 300 modelos frontier: la forma recomendada de usar Hermes',
  fireworksPitch: 'API directa de modelos: modelos frontier alojados en Fireworks',
  localModelsTitle: 'Ejecutar modelos localmente',
  localModelsPitch: 'Sin cuenta: descarga un modelo y ejecútalo en este equipo',
  openRouterPitch: 'Una clave, cientos de modelos: un buen valor predeterminado',
  apiKeyOptions: {
    fireworks: {
      short: 'API de modelo directo',
      description: 'Acceso directo a modelos alojados en Fireworks AI.'
    },
    openrouter: {
      short: 'una clave, muchos modelos',
      description:
        'Aloja cientos de modelos detrás de una sola clave. Buen valor predeterminado para instalaciones nuevas.'
    },
    openai: {
      short: 'modelos tipo GPT',
      description: 'Acceso directo a modelos de OpenAI.'
    },
    gemini: {
      short: 'modelos Gemini',
      description: 'Acceso directo a modelos de Google Gemini.'
    },
    xai: {
      short: 'modelos Grok',
      description: 'Acceso directo a modelos Grok de xAI.'
    },
    local: {
      short: 'autohospedado',
      description:
        'Apunta Hermes a un endpoint local o autohospedado compatible con OpenAI (vLLM, llama.cpp, Ollama, etc.).'
    }
  },
  backToSignIn: 'Volver al inicio de sesión',
  getKey: 'Obtener una clave',
  replaceCurrent: 'Reemplazar valor actual',
  pasteApiKey: 'Pegar clave API',
  localApiKeyPlaceholder: 'Clave API (opcional; solo si tu endpoint la requiere)',
  localModelNamePlaceholder: 'Nombre del modelo (p. ej. command-a-plus-05-2026)',
  couldNotSave: 'No se pudo guardar la credencial.',
  connecting: 'Conectando',
  update: 'Actualizar',
  flowSubtitles: {
    pkce: 'Abre tu navegador para iniciar sesión y luego continúa aquí',
    device_code: 'Abre una página de verificación en tu navegador; Hermes se conecta automáticamente',
    external: 'Inicia sesión una vez en tu terminal y vuelve para chatear'
  },
  startingSignIn: provider => `Iniciando sesión con ${provider}...`,
  verifyingCode: provider => `Verificando tu código con ${provider}...`,
  connectedProvider: provider => `${provider} conectado`,
  connectedPicking: provider => `${provider} conectado. Eligiendo un modelo predeterminado...`,
  signInFailed: 'No se pudo iniciar sesión. Inténtalo de nuevo.',
  signInExpired:
    'La página de inicio de sesión caducó antes de que terminaras. Vuelve a intentarlo y completa el paso del navegador en unos minutos, o usa una clave API.',
  signInDidNotFinish: (provider: string) =>
    `No se completó el inicio de sesión con ${provider}. Comprueba tu conexión a internet y vuelve a intentarlo, o elige otro proveedor.`,
  tryAgain: 'Reintentar',
  useApiKeyInstead: 'Usar una clave API',
  errorDetails: 'Detalles',
  pickDifferentProvider: 'Elegir otro proveedor',
  signInWith: provider => `Iniciar sesión con ${provider}`,
  openedBrowser: provider => `Abrimos ${provider} en tu navegador.`,
  authorizeThere: 'Autoriza Hermes allí.',
  copyAuthCode: 'Copia el código de autorización y pégalo abajo.',
  pasteAuthCode: 'Pegar código de autorización',
  reopenAuthPage: 'Volver a abrir página de autorización',
  waitingAuthorize: 'Esperando tu autorización...',
  externalPending: provider =>
    `${provider} inicia sesión con su propia CLI. Ejecuta este comando en una terminal y luego vuelve y elige "Ya inicié sesión":`,
  signedIn: 'Ya inicié sesión',
  deviceCodeOpened: provider => `Abrimos ${provider} en tu navegador. Introduce este código allí:`,
  reopenVerification: 'Volver a abrir página de verificación',
  copy: 'Copiar',
  defaultModel: 'Modelo predeterminado',
  freeTier: 'Nivel gratis',
  pro: 'Pro',
  free: 'Gratis',
  price: (input, output) => `${input} entrada / ${output} salida por Mtok`,
  change: 'Cambiar',
  startChatting: 'Empezar',
  docs: provider => `Docs de ${provider}`
}
