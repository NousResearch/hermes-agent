import type { RedesignOverlay } from './types'

// Spanish wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const esRedesign: RedesignOverlay = {
  sharedMetrics: {
    sending:
      'Si compartes, Hermes envía las estadísticas a Nous una vez al día con un ID aleatorio para este agente. Salvo un aviso único de que Hermes se instaló, las estadísticas de antes de aceptar nunca se envían. Puedes cambiarlo cuando quieras en Configuración.',
    sendDesc:
      'Solo recuentos, enviados una vez al día con un ID aleatorio para este agente. Nunca tus mensajes, archivos, rutas ni textos de error.',
    consentBody:
      'Hermes puede contar cómo lo usas: duración de los chats, qué modelos y herramientas se ejecutan y cuándo algo falla. Nunca registra tus mensajes, archivos, rutas ni textos de error.',
    collectedActivity: 'Chats: duración, resultado, tipo de error, tiempo activo por día',
    collectedMilestones:
      'Configuración: qué pasos se completaron, conexiones con proveedores, cuántos skills, plugins y rutinas'
  },
  connectorsPage: {
    dialog: {
      disconnectBody:
        'Esto desconecta la cuenta para todos los agentes que usan la misma cuenta de Nous, no solo para este agente. Puedes volver a conectarla cuando quieras.',
      nousLine: 'Las apps de Nous siguen a tu cuenta, no al agente.',
      rulesSignIn: 'Inicia sesión para cambiar lo que el agente puede hacer aquí.',
      appSwitch: (name: string) => `El agente puede usar ${name}`,
      bothOn: (name: string) => `Ambos están activados, así que el agente ve cada herramienta de ${name} dos veces.`
    },
    accounts: {
      removeBody:
        'Esto elimina la cuenta conectada para todos los agentes que usan la misma cuenta de Nous, no solo para este agente. Tus otras cuentas siguen conectadas.'
    },
    page: {
      noMatchBody: 'No hay coincidencias. Añade tu propio servidor MCP para dar más herramientas al agente.'
    },
    tools: {
      goneBody: 'El agente ya no puede llamarlo. La fila se queda hasta que la quites, así que nada desaparece.',
      summaryTitle: (name: string) => `Lo que el agente puede hacer con ${name}`,
      summaryPreviewTitle: (name: string) => `Lo que el agente podría hacer con ${name} cuando lo conectes`
    }
  },
  common: {
    bots: 'Agentes'
  },
  boot: {
    steps: {
      connectingGateway: 'Conectando con la máquina',
      loadingSessions: 'Cargando chats recientes'
    },
    failure: {
      remoteTitle: 'Se requiere iniciar sesión en la máquina remota',
      remoteDescription:
        'Tu sesión de la máquina remota caducó. Inicia sesión de nuevo para reconectar. Esto no elimina tus chats ni tu configuración.',
      useLocalGateway: 'Usar máquina local',
      gatewaySettings: 'Configuración de la máquina',
      remoteSignInHint: signInLabel =>
        `Cierra la sesión guardada del navegador remoto y abre ${signInLabel}. Usa la máquina local para cambiar al backend incluido.`,
      remoteFailureHint: 'Revisa la URL e inicia sesión en Configuración de la máquina, o cambia a la máquina local.',
      cloudDownDescription:
        'El agente en la nube administrado por Nous al que se conecta esta app devuelve un error de servidor. No se puede reiniciar desde aquí: revisa su estado, cambia a la máquina local o pide ayuda.',
      signedInMessage: 'Reconectando con la máquina remota…',
      signInToRemoteGateway: 'Iniciar sesión en la máquina remota'
    }
  },
  notifications: {
    errors: {
      gatewayAuthFailed:
        'Este Hermes ya no acepta tu inicio de sesión guardado. Abre Máquinas y vuelve a iniciar sesión (o pega un nuevo token de acceso) e inténtalo otra vez.'
    },
    actions: {
      openGateways: 'Abrir Máquinas'
    },
    voice: {
      couldNotStartSession: 'No se pudo iniciar el chat de voz',
      liveEnded: 'Chat de voz en vivo finalizado',
      liveEndedConnectionLost: 'El chat de voz en vivo perdió la conexión.',
      liveEndedClosed: 'El servicio cerró el chat de voz en vivo.',
      liveDelegationFailed: 'No se pudo pasar la solicitud al agente'
    },
    native: {
      inputBody: 'El agente espera tu respuesta.',
      turnDoneTitle: 'El agente terminó'
    }
  },
  promptNotices: {
    desktopCommands: {
      configFooter:
        'Cambia los ajustes en Ajustes, o edita config.yaml con `hermes config edit` en la máquina conectada.',
      saveLegacyRefused:
        'No guardado: Hermes en esta máquina no puede exportar una copia censurada, con nombre o que no sea JSON. Actualiza Hermes o ejecuta /save sin opciones para escribir el JSON sin procesar en la máquina conectada.',
      savedOnHost: (path: string) => `Guardado en la máquina conectada: ${path}`,
      exportRemote:
        '/export empaqueta un agente desde la máquina conectada, que no es este equipo. Ejecuta `hermes profile export` en esa máquina.',
      importRemote:
        '/import instala un agente en la máquina conectada, que no es este equipo. Ejecuta `hermes profile import <archive>` en esa máquina.',
      historyHermes: 'Agente',
      checkpointsDisabled:
        'Los puntos de control no están activados para este chat (configura checkpoints.enabled en config.yaml).'
    },
    sessionCommands: {
      busy: (command: string) =>
        `chat ocupado — el agente sigue respondiendo. Detén primero la respuesta actual (botón Detener o Ctrl+C en una terminal) y luego ejecuta /${command}.`,
      busyCompressing: (command: string) =>
        `chat ocupado — se está ejecutando un /compress; ejecuta /${command} cuando termine.`,
      titleSet: (title, queued) =>
        `Título del chat establecido: ${title}${queued ? ' (en cola mientras se inicia el chat)' : ''}`,
      titleCleared: 'Título del chat borrado.'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      'Esto sube un paquete de depuración a un almacenamiento interno de Nous (no a un sitio público). Incluye información del sistema (SO, versiones, proveedor y qué claves API están configuradas, nunca las claves en sí) y los registros completos del agente, del servicio Hermes y de la app de escritorio (hasta 512 KB cada uno), que probablemente contengan contenido de conversaciones, salidas de herramientas y rutas de archivos. Los secretos se ocultan antes de subirlo. Solo el personal de Nous y los moderadores autorizados de Discord pueden ver el paquete, que se elimina automáticamente a los 14 días.'
  },
  keybinds: {
    categories: {
      profiles: 'Agentes',
      session: 'Chat'
    },
    actions: {
      'nav.profiles': 'Abrir agentes',
      'nav.cron': 'Abrir rutinas',
      'view.toggleProfileRail': 'Mostrar u ocultar la barra de agentes',
      'profile.default': 'Cambiar al agente predeterminado',
      'profile.switch.1': 'Cambiar al agente 1',
      'profile.switch.2': 'Cambiar al agente 2',
      'profile.switch.3': 'Cambiar al agente 3',
      'profile.switch.4': 'Cambiar al agente 4',
      'profile.switch.5': 'Cambiar al agente 5',
      'profile.switch.6': 'Cambiar al agente 6',
      'profile.switch.7': 'Cambiar al agente 7',
      'profile.switch.8': 'Cambiar al agente 8',
      'profile.switch.9': 'Cambiar al agente 9',
      'profile.switch.10': 'Cambiar al agente 10',
      'profile.switch.11': 'Cambiar al agente 11',
      'profile.switch.12': 'Cambiar al agente 12',
      'profile.switch.13': 'Cambiar al agente 13',
      'profile.switch.14': 'Cambiar al agente 14',
      'profile.switch.15': 'Cambiar al agente 15',
      'profile.switch.16': 'Cambiar al agente 16',
      'profile.switch.17': 'Cambiar al agente 17',
      'profile.switch.18': 'Cambiar al agente 18',
      'profile.next': 'Siguiente agente',
      'profile.prev': 'Agente anterior',
      'profile.toggleAll': 'Alternar vista de todos los agentes',
      'profile.create': 'Crear agente',
      'agents.new': 'Nuevo agente (página Agentes)',
      'nav.artifacts': 'Abrir biblioteca',
      'session.new': 'Nuevo chat',
      'session.newTab': 'Nueva pestaña de chat',
      'session.next': 'Siguiente chat',
      'session.prev': 'Chat anterior',
      'session.slot.1': 'Cambiar al chat reciente 1',
      'session.slot.2': 'Cambiar al chat reciente 2',
      'session.slot.3': 'Cambiar al chat reciente 3',
      'session.slot.4': 'Cambiar al chat reciente 4',
      'session.slot.5': 'Cambiar al chat reciente 5',
      'session.slot.6': 'Cambiar al chat reciente 6',
      'session.slot.7': 'Cambiar al chat reciente 7',
      'session.slot.8': 'Cambiar al chat reciente 8',
      'session.slot.9': 'Cambiar al chat reciente 9',
      'session.focusSearch': 'Buscar chats',
      'session.togglePin': 'Fijar / desfijar chat actual',
      'session.archive': 'Archivar el chat actual',
      'view.toggleSidebar': 'Alternar barra lateral de chats',
      'view.cycleSidebarGrouping': 'Cambiar la agrupación de chats'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: 'Máquina principal'
    },
    nav: {
      gateway: 'Máquinas',
      sessions: 'Chats'
    },
    plugins: {
      installModal: {
        profileLabel: 'Instalar para el agente',
        desktopTargetFromPackage: 'Se carga en esta app desde el paquete de arriba; es igual para todos los agentes'
      }
    },
    searchPlaceholder: {
      gateway: 'Conexión de la máquina...',
      sessions: 'Buscar chats archivados...'
    },
    appearance: {
      vibeHeartsDesc:
        'Corazones flotantes cuando dices gracias, te quiero, buen agente o envías un corazón. Independiente de las reacciones a mensajes de arriba.',
      themeProfileNote: profile => `Guardado para el agente ${profile}; cada agente conserva su propio tema.`,
      sessionDensityTitle: 'Densidad de la lista de chats',
      sessionDensityDesc: 'Elige cuánto contexto aparece bajo los títulos de los chats en la barra lateral.',
      reactionsDesc:
        'Reacciones emoji estilo iMessage — reacciona a los mensajes, y el agente puede reaccionar a los tuyos.',
      tipsDesc:
        'Sugerencias ocasionales de la app y de tu agente. Cada consejo aparece una vez. Se desactiva automáticamente tras tus primeros 30 días; puedes volver a activarlo.',
      toursDesc:
        'Deja que el agente resalte cada paso mientras te guía por la app. Se desactiva automáticamente tras tus primeros 30 días; puedes volver a activarlo.',
      pet: {
        intro:
          'Adopta una mascota animada de petdex que flota sobre la app y reacciona a lo que hace el agente: corre mientras se ejecutan herramientas, celebra los éxitos y se entristece con los errores.'
      }
    },
    uninstallSection: {
      options: {
        full: {
          description:
            'Quita la app, el agente y todos los datos de usuario: configuración, chats, rutinas, secretos y registros.'
        }
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `Esta versión es más reciente que la versión estable ${v}, y tus chats ya usan un entorno de ejecución de Hermes que ${v} no puede abrir. Hermes se queda en esta versión hasta que salga una versión estable más reciente.`
      }
    },
    connections: {
      title: 'Máquinas registradas',
      intro:
        'Gestiona este dispositivo y cada máquina con Hermes a la que puede llegar mediante conexiones remotas, SSH o Cloud.',
      stagedNote:
        'Todas las máquinas permanecen conectadas. Los agentes, chats, mensajería y rutinas se quedan en su máquina, y Chats puede mostrarlos todos a la vez.',
      launchModeTitle: 'Al iniciar, volver a Chats en la última máquina usada',
      launchModeDesc: 'Si está desactivado, Chats se abre en la máquina principal.',
      searchPlaceholder: 'Buscar máquinas…',
      noSearchResults: 'Ninguna máquina coincide con tu búsqueda.',
      kindRemote: 'Máquina remota',
      kindRemoteDesc: 'Una máquina con Hermes accesible por HTTP(S): LAN, Tailscale o internet.',
      urlTitle: 'URL de la máquina',
      headersTitle: 'Encabezados de conexión adicionales',
      headersDesc:
        'Se envían con cada solicitud HTTP y WebSocket a esta máquina, para proxies de acceso como Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Los valores se guardan cifrados. Se ignoran los encabezados que gestiona Hermes (Authorization, Cookie, Host…).',
      duplicateUrl: (label: string) => `Ya existe una conexión a la URL de esta máquina (“${label}”).`
    },
    managedUpdates: {
      intro:
        'Actualiza de forma transaccional las instalaciones SSH administradas por la app: los chats en curso terminan, la copia remota se actualiza y cada agente se restaura con un comprobante correlacionado.',
      progress: 'Terminando los chats en curso, actualizando la instalación remota y restaurando agentes…',
      scopesRestored: (profiles: string) => `Agentes restaurados: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `No se restauró el agente “${profile}”: ${error}`
    },
    gateway: {
      loading: 'Cargando ajustes de la máquina...',
      unavailableTitle: 'Ajustes de la máquina no disponibles',
      webappHostDesc:
        'La Webapp siempre se ejecuta en la máquina que la sirve. Las conexiones guardadas de abajo son la lista de esta máquina, compartida con la app Hermes Desktop; las demás máquinas se alcanzan a través de esta.',
      webappOauthPaste: {
        description:
          'Si la pestaña de inicio de sesión terminó en una dirección que no carga (este navegador no está en la máquina que ejecuta Hermes), pega esa dirección aquí.'
      },
      webAccess: {
        pendingOn: 'Guardado. Hermes servirá la aplicación web cuando se reinicie.',
        pendingOff: 'Desactivado, pero Hermes sigue sirviendo la aplicación web hasta que se reinicie.',
        unavailableDetail: reason => `Hermes no pudo servir la aplicación web: ${reason}`,
        restartBusy:
          'Guardado. Hay un chat en curso, así que Hermes no se reinició. Elige «Reiniciar Hermes» cuando termine.',
        restartFailed: 'Guardado, pero Hermes no pudo reiniciarse. Elige «Reiniciar Hermes» para volver a intentarlo.',
        restartFailedBecause: reason => `Hermes no pudo reiniciarse: ${reason}`,
        startTimedOut: 'Hermes no sirvió la aplicación web en 90 segundos.',
        stopTimedOut: 'Hermes no dejó de servir la aplicación web en 90 segundos.',
        startStages: {
          restarting: 'Reiniciando Hermes para servir la aplicación web.'
        },
        restartGateway: 'Reiniciar Hermes'
      },
      title: 'Conexión de la máquina',
      intro:
        'Local por predeterminado. Usa remoto cuando esta app deba ejecutar Hermes en otro lugar. Conecta una máquina para descubrir sus agentes.',
      localTitle: 'Máquina local',
      remoteTitle: 'Máquina remota',
      remoteAuthHint:
        'Las máquinas alojadas usan OAuth o usuario y contraseña; las autoalojadas pueden usar un token de sesión.',
      cloudSavedTitle: 'Máquinas de Cloud guardadas',
      cloudSavedDesc:
        'Usa una máquina guardada sin cambiar la predeterminada. Inicia sesión abajo para añadir instancias. Gestiona los nombres y el inicio de sesión en la lista de conexiones guardadas.',
      cloudUseSaved: 'Usar máquina',
      probing: 'Comprobando cómo se autentica esta máquina…',
      authSignedInPassword:
        'Esta máquina usa usuario y contraseña. Ya iniciaste sesión; la sesión se actualiza automáticamente.',
      authSignedInOauth: 'Esta máquina usa OAuth. Ya iniciaste sesión; la sesión se actualiza automáticamente.',
      authNeedsPassword: 'Esta máquina usa usuario y contraseña. Inicia sesión para autorizar esta app de escritorio.',
      authNeedsOauth: provider => `Esta máquina usa OAuth. Inicia sesión con ${provider} para autorizar esta app.`,
      plainTextConfirmTitle: '¿Guardar el token de conexión en texto plano?',
      keychainEncryptionDesc:
        'Desactivado por defecto. Si está activado, los tokens de conexión y las credenciales de inicio de sesión se cifran con el llavero del sistema (Acceso a Llaveros, GNOME Keyring o DPAPI de Windows); es posible que el sistema te pida permiso o una contraseña. Si está desactivado, se guardan como archivos normales que solo tu cuenta de usuario puede leer.',
      diagnosticsDesc: 'Muestra desktop.log en tu gestor de archivos; útil cuando Hermes no arranca.',
      incompleteTitle: 'Configuración de la máquina remota incompleta',
      restartingTitle: 'Restableciendo la conexión con la máquina',
      savedTitle: 'Ajustes de la máquina guardados',
      reachableTitle: 'Máquina remota accesible',
      signedOutMessage: 'Se borró la sesión de la máquina remota.',
      failedLoad: 'No se pudieron cargar los ajustes de la máquina',
      testFailed: 'Falló la prueba de conexión con la máquina remota',
      applyFailed: 'No se pudieron aplicar los ajustes de la máquina',
      saveFailed: 'No se pudieron guardar los ajustes de la máquina',
      webappHostTitle: 'Máquina de Hermes'
    },
    profileScope: {
      editsProfile: profile => `Los cambios de esta página se aplican al agente “${profile}”.`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'Descriptor de agentes',
          hint: 'Descripciones automáticas de agentes'
        },
        title_generation: {
          hint: 'Títulos de chats'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'Rutas de agente',
              description:
                'Los subagentes se ejecutan como uno de tus agentes, con su propio modelo, habilidades y claves.'
            }
          },
          pluginMissing: 'Rutas de modelo y Rutas de agente provienen del plugin Advanced Delegation.',
          pluginDisabled: 'Advanced Delegation está instalado, pero desactivado para este agente.',
          blockedDisabled:
            'Los subagentes no se iniciarán hasta que Advanced Delegation esté activado para este agente.',
          profileRoutesTitle: 'Agentes que pueden actuar como subagentes',
          profileRoutesHint: 'Cada agente marcado aporta su propio modelo, habilidades, personalidad y claves.',
          profileEmpty: 'Aún no hay otros agentes. Crea uno desde la barra de agentes.',
          profileMissing: name => `${name} (el agente ya no existe)`,
          inactiveProfiles: profile =>
            `Las rutas de agente eligen qué agente ejecuta cada subagente. Los subagentes que no eligen ninguno se ejecutan como “${profile}”.`,
          inactiveProfilesNoDefault:
            'Las rutas de agente eligen qué agente ejecuta cada subagente. Los subagentes que no eligen ninguno usan el modelo del agente principal.'
        }
      },
      appliesDesc: 'Se aplica a chats nuevos. Usa el selector de modelo en el compositor para cambiar el chat activo.',
      mainAppliedMessage: model => `Los chats nuevos usarán ${model}.`
    },
    billing: {
      errors: {
        remoteSpendingReconnect: (who: string) =>
          `${who} Vuelve a conectarte desde Configuración -> Máquinas para autorizar de nuevo este dispositivo.`,
        sessionRevoked: {
          message: 'Se cerró tu sesión. Vuelve a iniciar sesión desde Configuración → Máquinas.'
        },
        transport: {
          message: 'La solicitud de facturación falló antes de llegar a la máquina.'
        }
      }
    },
    vault: {
      otpHint:
        'La “clave de configuración” que muestra el sitio al activar la 2FA. Si la guardas, el agente genera los códigos por sí mismo.'
    },
    notifications: {
      kinds: {
        input: {
          description: 'El agente hizo una pregunta o necesita una contraseña o un secreto.'
        }
      }
    },
    fieldDescriptions: {
      'display.personality': 'Estilo predeterminado del asistente para chats nuevos.',
      'memory.memoryEnabled': 'Guarda memorias duraderas que pueden ayudar en chats futuros.'
    },
    memoryLayers: {
      noProviderDesc:
        'El agente usa solo la capa integrada de abajo. Elige un proveedor externo en Ajustes › Memoria y contexto para añadir uno.',
      builtinDesc:
        'MEMORY.md y USER.md son la memoria propia del agente, independiente de cualquier proveedor externo. Ambos se cargan en los chats por defecto; los interruptores de memoria y perfil de usuario en Ajustes › Memoria y contexto desactivan cada uno. Un proveedor se suma a esta capa; nunca la reemplaza.',
      selectedDesc: current =>
        `Aún no está activo: los chats siguen usando ${current} hasta que se guarde esta elección.`,
      providerDesc: name => `${name} añade su propio recuerdo a cada chat, además de la capa integrada.`
    },
    sessions: {
      loading: 'Cargando chats archivados…',
      archivedTitle: 'Chats archivados',
      defaultDirDesc:
        'Los chats nuevos empiezan en esta carpeta salvo que elijas otra. Déjala sin definir para usar tu directorio de inicio.',
      failedLoad: 'No se pudieron cargar los chats archivados'
    },
    toolsets: {
      terminalBackend: {
        needsSetupConfirmDescriptionGeneric:
          'Este backend todavía no está configurado. Los chats que empiecen tras este cambio no tendrán herramientas de terminal ni de archivos hasta que termine la configuración.',
        switchedToLocal: 'Los comandos de terminal ahora se ejecutan localmente. Se aplica a los chats nuevos.',
        selectedMessage: backend =>
          `Los comandos del terminal ahora se ejecutan mediante ${backend}. Se aplica a los chats nuevos.`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Los chats que empiecen tras este cambio no tendrán herramientas de terminal ni de archivos hasta que termine la configuración.`
      },
      browserRealProfile: {
        description:
          'Copia los inicios de sesión y las cookies de tu navegador predeterminado en una instantánea administrada con la que navega el agente, y también inicia sesión con ellos en el navegador integrado. Tu perfil en uso nunca se abre directamente. Se aplica a los chats nuevos.',
        enabledMessage: 'Los chats nuevos navegarán con una instantánea de tu perfil predeterminado del navegador.',
        disabledMessage:
          'Se eliminará la instantánea del perfil y se cerrará su sesión en el navegador integrado; los chats nuevos usan un navegador limpio.',
        prompt: {
          body: 'Deja que el agente navegue con una instantánea de tu perfil predeterminado del navegador, para que los sitios se abran con la sesión ya iniciada, para el agente y en el navegador integrado.'
        }
      },
      modelSelectedMessage: model => `${model} se aplica a los chats nuevos.`
    }
  },
  skills: {
    plugins: {
      agentBlurb: 'Amplían el agente seleccionado: herramientas, hooks, proveedores. Se aplican tras reiniciar Hermes.',
      halfDesktopHint: 'esta app, igual para todos los agentes',
      halfAgentIn: (profile: string) => `Agente: ${profile}`,
      defaultProfile: 'Agente predeterminado',
      installAgentHereNoOrigin:
        'La mitad del agente no está instalada para este agente y este paquete se copió a mano (sin entrada de catálogo ni remoto de git), así que no se puede instalar desde aquí. Copia su carpeta en la del agente o reinstálalo desde Git.',
      empty: 'No hay plugins instalados para este agente.',
      alreadyInstalled: (name: string) => `${name} ya está instalado para este agente.`,
      updated: (name: string) =>
        `${name} se actualizó al commit fijado actual del catálogo. Reinicia Hermes para aplicarlo.`,
      autoUpdateTip: (hours: number) =>
        `Aplicar las actualizaciones de plugins automáticamente cuando Hermes compruebe (cada ${hours} h). El código nuevo se carga en los chats nuevos.`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `Esto elimina los archivos del plugin del agente ${profile}. Cualquier mitad de escritorio que incluyera se quita con él. Puedes reinstalarlo desde el catálogo o desde Git cuando quieras.`,
      uninstalled: (name: string) => `${name} desinstalado. Reinicia Hermes para descargarlo.`,
      settingsForm: {
        secretStoredAs: (env: string) =>
          `Se guarda en el .env del agente como ${env}, nunca en config.yaml; déjalo en blanco para conservar el valor actual.`
      }
    },
    remotePluginHint:
      'Abre este agente en su máquina para gestionar sus plugins. La instalación de plugins usa la máquina conectada.',
    openAgentSettings: 'Ajustes del agente',
    changesApplyNewSessions: 'Los cambios se aplican a los chats nuevos.',
    appliesToNewSessions: name => `${name} se aplica a los chats nuevos.`,
    bulkUpdated: count =>
      `${count === 1 ? 'Se actualizó' : 'Se actualizaron'} ${count} ${count === 1 ? 'elemento' : 'elementos'} para los chats nuevos.`,
    openToolLibrary: 'Explorar el catálogo de herramientas'
  },
  gatewayGroupOpen: {
    lookupFailed: 'No se pudo encontrar este grupo en la máquina. Inténtalo de nuevo.'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'Estado de la máquina, registros, reinicio/actualización'
      },
      sessions: {
        title: 'Panel de chats',
        detail: 'Busca, fija y gestiona chats'
      }
    },
    gatewayRunning: 'Mensajería en ejecución',
    gatewayStopped: 'Mensajería detenida',
    restartGateway: 'Reiniciar Hermes',
    gatewayRestartFailed: 'No se pudo reiniciar Hermes.',
    sharedGatewayRestartTitle: '¿Reiniciar Hermes para todos los agentes?',
    sharedGatewayRestartDescription: (bots: string) => `Todos los agentes de este dispositivo se reconectan: ${bots}`,
    sharedGatewayRestarted: (count: number) => `Hermes reiniciado (${count} ${count === 1 ? 'agente' : 'agentes'})`,
    searchPlaceholder: 'Buscar chats, vistas y acciones',
    goToSession: 'Ir al chat',
    sections: {
      sessions: 'Chats'
    },
    nav: {
      newChat: {
        title: 'Nuevo chat',
        detail: 'Inicia un chat nuevo'
      },
      artifacts: {
        title: 'Biblioteca'
      }
    },
    providerSessions: 'Chats',
    pinSession: 'Fijar chat',
    unpinSession: 'Desfijar chat',
    exportSession: 'Exportar chat',
    deleteSession: 'Eliminar chat',
    noSessions: 'Aún no hay chats.',
    statSessions: 'Chats',
    maintenance: {
      backupDesc: 'Comprimir la configuración, memorias, skills y chats en un archivo ZIP'
    },
    newSessionInProject: project => `Nuevo chat en ${project}`,
    hermesActiveSessions: (version, count) => `Hermes ${version} · Chats activos ${count}`
  },
  messaging: {
    states: {
      gateway_stopped: 'Mensajería detenida'
    },
    hintPendingRestart: 'Reinicia Hermes desde la barra de estado para aplicar este cambio.',
    sharedListenerUrl: 'Disponible en',
    hintGatewayStopped: 'Inicia Hermes desde la barra de estado para conectar.',
    gatewayStopped: 'Mensajería detenida',
    restartToApply: 'Reinicia Hermes para que este cambio surta efecto.',
    restartToReconnect: 'Reinicia Hermes para reconectar con las credenciales nuevas.',
    appliedLive: 'Aplicado a Hermes en ejecución.',
    connectingLive: 'Hermes se está conectando con las credenciales nuevas.',
    restartNeeded: 'Guardado. Reinicia Hermes para que la nueva configuración surta efecto.',
    relayLink: {
      managedNote: (platform: string) =>
        `Team Gateway gestiona la mensajería en esta instancia, así que Hermes desactiva la conexión directa de ${platform} en cada inicio. Para usarla junto al relay, define GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true en el entorno de Hermes (una conexión directa impide que la instancia escale a cero y puede entregar mensajes dos veces).`
    },
    telegramQr: {
      savedRestarting: 'Telegram guardado; reiniciando Hermes…',
      savedRestartFailed: (detail: string) => `Telegram guardado; falló el reinicio de Hermes${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'Canal donde el bot envía mensajes proactivos (resultados de rutinas, recordatorios).'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'Canal o grupo predeterminado para entregar resultados de rutinas.'
      }
    }
  },
  webhooks: {
    disabledBody:
      'Los webhooks funcionan con independencia de los canales de chat. Actívalos aquí para aceptar eventos HTTP entrantes; los canales de chat solo son necesarios cuando una suscripción entrega contenido a Telegram, Discord, Slack u otro canal.',
    restarting: 'Reiniciando Hermes…',
    restartNeeded: 'Los webhooks están activados. Reinicia Hermes para empezar a recibir eventos.',
    restartGateway: 'Reiniciar Hermes',
    restartFailed: detail => `Falló el reinicio de Hermes${detail}`,
    enabledRestarting: 'Webhooks activados; reiniciando Hermes…'
  },
  profiles: {
    close: 'Cerrar agentes',
    title: 'Agentes',
    count: count => `${count} ${count === 1 ? 'agente' : 'agentes'}`,
    search: 'Buscar agentes…',
    loading: 'Cargando agentes...',
    newProfile: 'Nuevo agente',
    importProfile: 'Importar agente…',
    exportProfile: 'Exportar agente…',
    imported: 'Agente importado',
    exported: 'Agente exportado',
    failedImport: 'No se pudo importar el agente',
    failedExport: 'No se pudo exportar el agente',
    exportNeedsHost: name =>
      `El navegador no puede guardar un archivo de agente. En la máquina que ejecuta Hermes, ejecuta: hermes profile export ${name}`,
    allProfiles: 'Todos los agentes',
    showAllProfiles: 'Mostrar todos los agentes',
    manageProfiles: 'Gestionar agentes...',
    connectGateway: 'Gestionar máquinas…',
    fleet: {
      allOnGateway: 'Todos los agentes de esta máquina',
      gateway: (gateway: string) => `Agentes en ${gateway}`,
      localDevice: 'Este dispositivo (backend local: instala Hermes si falta; si no, abre un chat nuevo)',
      installDeviceDesc:
        'Esto instalará Hermes localmente y luego abrirá un chat nuevo en este equipo. No se instala nada hasta que confirmes.'
    },
    remoteOverride: {
      description:
        'Los chats de este agente se ejecutarán en el Hermes remoto que indiques, en lugar de en este equipo.',
      collisionWarning: (label: string) =>
        `Ya existe una máquina llamada “${label}” en Configuración. Esta conexión del agente es independiente y no la cambiará.`,
      confirmTitle: '¿Conectar este agente a una máquina remota?',
      savedTitle: 'Agente conectado',
      menuItem: 'Conectar a una máquina remota…',
      authFailedTitle: 'La máquina remota rechazó el token guardado',
      title: (profile: string) => `Conectar ${profile} a una máquina remota`,
      confirmNote: (profile: string, host: string) =>
        `Los chats nuevos de ${profile} se ejecutarán en ${host}. Ese equipo ejecutará comandos y leerá archivos allí, no en este. Conéctate solo a una máquina de confianza.`
    },
    defaultProfile: 'Agente predeterminado',
    defaultDescription:
      'Se usa al abrir Hermes y para los chats nuevos. Los chats existentes se quedan con sus agentes.',
    failedSetDefault: 'No se pudo establecer el agente predeterminado',
    noProfiles: 'Aún no hay agentes.',
    selectPrompt: 'Selecciona un agente para ver sus detalles.',
    refresh: 'Actualizar agentes',
    refreshing: 'Actualizando agentes',
    soulDesc: 'Las instrucciones y la personalidad de este agente.',
    soulPlaceholder: mode =>
      `Las instrucciones y la personalidad de este agente.\nDéjalo en blanco para conservar el ${mode} predeterminado.`,
    deleteTitle: '¿Eliminar agente?',
    createDesc: 'Cada agente tiene sus propias instrucciones, ajustes y chats.',
    cloneFromDesc: 'Copia la configuración, las skills y SOUL.md del agente de origen seleccionado.',
    cloneFromDefaultDesc: 'Copia configuración, skills y SOUL.md desde tu agente predeterminado.',
    createAction: 'Crear agente',
    renameTitle: 'Renombrar agente',
    renameDescPrefix: 'Renombrar actualiza el directorio del agente y cualquier script wrapper en ',
    displayNameDesc:
      'Define un nombre visible que se muestra en toda la app. El ID interno del agente sigue siendo "default".',
    created: 'Agente creado',
    renamed: 'Agente renombrado',
    deleted: 'Agente eliminado',
    failedLoad: 'No se pudieron cargar los agentes',
    failedDelete: 'No se pudo eliminar el agente',
    failedCreate: 'No se pudo crear el agente',
    failedRename: 'No se pudo renombrar el agente',
    status: {
      unread: (count: number) => (count === 1 ? '1 chat sin leer' : `${count} chats sin leer`),
      needsInput: (count: number) =>
        count === 1 ? '1 chat espera tu respuesta' : `${count} chats esperan tu respuesta`,
      working: (count: number) => (count === 1 ? '1 chat en ejecución' : `${count} chats en ejecución`)
    }
  },
  cron: {
    close: 'Cerrar rutinas',
    title: 'Rutinas',
    count: count => `${count} ${count === 1 ? 'rutina' : 'rutinas'}`,
    search: 'Buscar rutinas...',
    loading: 'Cargando rutinas...',
    editJob: 'Editar rutina',
    newCron: 'Nueva rutina',
    emptyDescNew:
      'Las rutinas ejecutan tus instrucciones automáticamente y envían los resultados al destino que elijas.',
    emptyTitleNew: 'Aún no hay rutinas',
    actionsTitle: 'Acciones de la rutina',
    resume: 'Reanudar rutina',
    pause: 'Pausar rutina',
    edit: 'Editar rutina',
    deleteTitle: '¿Eliminar rutina?',
    resumed: 'Rutina reanudada',
    paused: 'Rutina pausada',
    triggered: 'Rutina ejecutada',
    deleted: 'Rutina eliminada',
    created: 'Rutina creada',
    updated: 'Rutina actualizada',
    failedLoad: 'No se pudieron cargar las rutinas',
    failedUpdate: 'No se pudo actualizar la rutina',
    failedTrigger: 'No se pudo ejecutar la rutina',
    failedDelete: 'No se pudo eliminar la rutina',
    failedSave: 'No se pudo guardar la rutina',
    editTitle: 'Editar rutina',
    createTitle: 'Nueva rutina',
    editDesc:
      'Actualiza la frecuencia, las instrucciones o el destino. Los cambios se aplican en la próxima ejecución.',
    createDesc:
      'Ejecuta instrucciones automáticamente cuando elijas. Usa sintaxis cron o una frase como "cada 15 minutos".',
    scriptOnlyEditHint: 'Rutina solo con script (sin prompt de IA). ID de la rutina:',
    createAction: 'Crear rutina',
    tabs: {
      jobs: 'Rutinas'
    },
    blueprints: {
      subtitle: 'Rutinas listas para usar',
      scheduled: 'Rutina creada desde la plantilla',
      emptyDesc: 'Este backend no ofrece plantillas de rutinas.'
    }
  },
  sidebar: {
    filter: {
      profile: 'Agente'
    },
    gatewayGroups: {
      grouping: 'Máquina y agente',
      aliasHint: 'Solo el nombre visible; los nombres de la máquina y del agente no cambian.'
    },
    profileRail: 'Barra de agentes',
    nav: {
      cron: 'Rutinas',
      'new-session': 'Nuevo chat',
      artifacts: 'Biblioteca'
    },
    cronJobs: 'Rutinas',
    storageCorrupt: {
      action:
        'Sal de Hermes para este agente y luego inspecciona el archivo sin modificarlo, o restaura una instantánea:',
      title: 'La base de datos de chats está dañada',
      body: (profiles: string) =>
        `Hermes no puede leer todo el historial de chats de ${profiles}. Los chats que faltan en esta lista no se eliminaron; el archivo donde se guardan está dañado.`
    },
    projects: {
      createdInPreviousContext:
        'El proyecto se creó en la conexión o el agente anterior. Vuelve allí; no se escribió IDEA.md.',
      showAllSessions: 'Mostrar todos los chats',
      moveFailed: 'No se pudo mover el chat',
      toggle: (label, open) => `${open ? 'Mostrar' : 'Ocultar'} chats de ${label}`,
      showAllCount: (count: number) => `Mostrar los ${count} chats`
    },
    row: {
      ownedByProfile: profile => `Agente: ${profile}`,
      copyIdFailed: 'No se pudo copiar el ID del chat',
      sessionActions: 'Acciones del chat',
      sessionRunning: 'Chat en ejecución',
      renameTitle: 'Renombrar chat',
      untitledPlaceholder: 'Chat sin título',
      deleteTitle: '¿Eliminar el chat?',
      deleted: 'Chat eliminado'
    },
    searchAria: 'Buscar chats',
    searchPlaceholder: 'Buscar chats…',
    sessions: 'Chats',
    groupAriaGrouped: 'Mostrar los chats como una sola lista',
    groupAriaUngrouped: 'Agrupar chats por espacio de trabajo',
    showSessions: 'Mostrar chats',
    groupTitleGrouped: 'Desagrupar chats',
    projectEmpty: 'Aún no hay chats',
    projectLoadFailed: 'No se pudieron cargar los chats',
    noSessions: 'Aún no hay chats',
    noFilterMatches: 'Ningún chat coincide con estos filtros',
    noMatch: query => `Ningún chat coincide con “${query}”.`,
    newSessionIn: label => `Nuevo chat en ${label}`
  },
  composer: {
    commandDescs: {
      '/export': 'Exportar un agente (configuración, skills, tema) a un archivo compartible',
      '/import': 'Importar un archivo de agente compartido como agente nuevo',
      '/profile': 'Cambiar el agente activo',
      '/suggestions': 'Revisar las rutinas sugeridas (aceptar/descartar)',
      '/blueprint': 'Configurar una rutina a partir de una plantilla',
      '/platform': 'Pausar, reanudar o listar una plataforma de mensajería que falla',
      '/clear': 'iniciar un chat nuevo',
      '/resume': 'Reanudar un chat guardado',
      '/topic': 'Activar o inspeccionar los chats por tema de los MD de Telegram',
      '/title': 'Renombrar el chat actual',
      '/handoff': 'Pasar este chat a una plataforma de mensajería',
      '/bg': 'Ejecutar un prompt en un chat independiente en segundo plano',
      '/goal': 'Fijar un objetivo permanente en el que el agente trabaja durante varios turnos hasta cumplirlo',
      '/heartbeat': 'Configurar un prompt recurrente que vuelve a este chat cuando está inactivo',
      '/loop': 'Volver a ejecutar un prompt a intervalos regulares en este chat',
      '/status': 'Mostrar el estado del chat actual',
      '/model': 'Cambiar el modelo de este chat'
    },
    queueLostDiscardTip:
      'Hermes se reinició a mitad del turno y no puede terminarlo. Descártalo para que se ejecuten los turnos en cola que esperan detrás.',
    newSessionPlaceholders: [
      '¿Qué vamos a construir?',
      'Dale una tarea al agente',
      '¿Qué tienes en mente?',
      'Describe lo que necesitas',
      '¿Qué abordamos?',
      'Pregunta lo que quieras',
      'Empieza con un objetivo'
    ],
    voiceEngineChained: 'Voz a texto + voz del agente',
    voiceEngineLive: 'GPT-Live (full-duplex, delega en el agente)',
    attachUrlDesc: 'El agente obtendrá la página y la incluirá como contexto para este turno.',
    queueDroppedBody:
      'Se descartó esta entrada en segundo plano porque su chat no se pudo reanudar tras varios intentos. El resto de la cola no se ha visto afectado.',
    browseAllSessions: 'Ver todos los chats…'
  },
  updates: {
    bundleOutOfSyncDesc:
      'El entorno de ejecución de Hermes se actualizó, pero la app de escritorio sigue siendo una compilación anterior: faltarán funciones nuevas de la interfaz (como el modo Agente) hasta que se actualice. Ejecuta la actualización de abajo para recompilar la app. Si eso no elimina este aviso, reinstala desde el instalador de escritorio más reciente.',
    applyStatus: {
      noReturn:
        'El backend no volvió a estar disponible. Puede que la actualización no se haya completado; revisa la máquina que ejecuta Hermes.'
    }
  },
  install: {
    setupChoiceDesc:
      'Conecta esta app con una máquina que ya ejecute Hermes o instala Hermes localmente en este equipo.',
    remoteSetupDesc:
      'Introduce la URL de tu máquina. Hermes Desktop detectará si necesita un token o iniciar sesión en el navegador.',
    remoteUrlTitle: 'URL de la máquina',
    remoteUrlDesc: 'Usa la URL base de Hermes en la máquina e incluye https:// si es remota.',
    probing: 'Detectando la autenticación de la máquina…',
    authNeedsOauth: provider => `Inicia sesión con ${provider} antes de probar la conexión con esta máquina.`,
    enterUrlFirst: 'Introduce primero la URL de una máquina.',
    tokenDesc: 'Pega el token de sesión del archivo .env de Hermes en la máquina remota.',
    incompleteSignInTest: 'Inicia sesión antes de probar la conexión protegida con OAuth a esta máquina.',
    incompleteTokenTest: 'Introduce un token de sesión antes de probar la conexión con esta máquina.',
    connectExistingDesc:
      'Usa una máquina remota con un token de sesión o inicio de sesión en el navegador. No se iniciará ninguna instalación local.'
  },
  shell: {
    gatewayMenu: {
      gateway: 'Máquina',
      reconnectGateway: 'Reconectar máquina',
      machineSettings: 'Máquinas',
      restartTitle: '¿Reiniciar Hermes en esta máquina?'
    },
    statusbar: {
      gateway: 'Máquina',
      gatewayReady: 'lista',
      gatewayRestarting: 'reiniciando Hermes…',
      gatewayTitle: 'Máquina',
      cron: 'Rutinas',
      openCron: 'Abrir rutinas',
      toggleSessionTimer: 'Temporizador del chat',
      cacheHitRateTitle:
        'Tasa de aciertos de la caché de prompts en este chat: los tokens en caché cuestan menos, así que cuanto más alta, más barato'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: 'Máquina eliminada',
    terminalGatewayRemovedBody:
      'Se eliminó la máquina donde se ejecutaba esta terminal, así que no puede reabrirse. Cierra esta pestaña.',
    terminalGatewayNoShellBody:
      'Esta máquina no ofrece acceso a shell, así que esta terminal no puede reabrirse. Cierra esta pestaña.'
  },
  preview: {
    saveScopeChanged: 'Vuelve a la conexión y al agente originales para guardar este borrador.',
    web: {
      remoteLoopback:
        'Esta dirección apunta a la máquina que ejecuta tu agente, no a esta. El panel del navegador carga las páginas localmente, así que un servidor de desarrollo remoto necesita un reenvío de puertos o un nombre de host accesible.',
      restarting: 'El agente está reiniciando el servidor...',
      askRestart: 'Pedir al agente que reinicie el servidor',
      restartingMessage:
        'El agente está trabajando en segundo plano. Mira la consola de vista previa para ver el progreso.',
      blankPageBody: 'Escribe una dirección arriba para navegar o pide al agente que abra una página.',
      restartFailedMessage: 'El agente no pudo reiniciar el servidor.',
      stillWorking:
        'El agente sigue trabajando, pero aún no llegó ningún resultado de reinicio. Puede que el comando del servidor siga en primer plano.',
      lookingRestart: taskId => `El agente está buscando un servidor de vista previa para reiniciar (${taskId})`,
      finishedRestarting: message =>
        `El agente terminó de reiniciar el servidor de vista previa${message ? `: ${message}` : ''}`
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: 'Este chat pertenece a una rutina',
          body: 'Solo ejecuta esa rutina, así que tu mensaje no se ejecutó. Empieza un chat nuevo para enviarlo.'
        },
        no_reply: {
          body: 'El agente terminó este turno, pero no llegó ninguna respuesta a esta ventana. Reinténtalo para enviarla de nuevo.'
        },
        loop_error: {
          title: 'El agente no pudo terminar este paso'
        },
        launch_credentials_unavailable: {
          body: 'Este chat se inició con una clave de API que desde entonces cambió o se eliminó, así que tu mensaje no se envió. Un chat abierto nunca cambia por sí solo a otra clave. Inicia un chat nuevo para usar tu clave actual.'
        }
      },
      loadingSession: 'Cargando chat',
      loadingResponse: 'El agente está cargando una respuesta',
      errorLayers: {
        generic: 'El agente no pudo terminar esta respuesta',
        runtime: 'El agente tuvo un problema'
      },
      errorLayerBodies: {
        generic: 'Algo salió mal mientras el agente respondía. Reinténtalo o copia los detalles si sigue ocurriendo.'
      },
      errorToastTitle: 'El agente no pudo terminar la respuesta',
      errorStartNewSession: 'Iniciar chat nuevo'
    },
    catalogInstall: {
      targetProfile: (profile: string) => `Se instala para tu agente ${profile}`
    },
    tool: {
      titles: {
        cronjob: {
          done: 'Rutina',
          pending: 'Configurando rutina',
          pendingAction: 'Configurando'
        },
        session_search_recall: {
          done: 'Se buscó en el historial de chats',
          pending: 'Buscando en el historial de chats'
        }
      }
    },
    approval: {
      timedOutSystemLine:
        'Se agotó el tiempo de aprobación: el comando no se ejecutó. Pide al agente que lo intente de nuevo o sube el límite en Configuración → Seguridad → Tiempo de aprobación.',
      allowSession: 'Permitir en este chat',
      alwaysDescription: pattern =>
        `Esto añade el patrón “${pattern}” a tu allowlist permanente (~/.hermes/config.yaml). El agente no volverá a preguntar por comandos como este, ni en este chat ni en futuros.`
    },
    themePicker: {
      basicDetail: 'Para hablar con tu agente.'
    },
    mcpSetup: {
      reloadFailed: 'Servidor guardado, pero falló la recarga de herramientas MCP — se cargarán en el próximo chat'
    }
  },
  prompts: {
    sudoInstallDesc:
      'El agente necesita tu contraseña de sudo para instalar los paquetes de la pantalla del agente (TigerVNC + Xfce) en su máquina. Solo se envía a esa máquina.',
    sudoDesc:
      'Revisa el comando antes de introducir tu contraseña de sudo. La contraseña se envía al agente que lo ejecuta y se guarda en caché durante este chat.',
    secretDesc: 'El agente necesita una credencial para continuar.',
    vaultCodeFootnote:
      'Consejo: guarda la clave del autenticador con este acceso en Configuración → Contraseñas e inicios de sesión y el agente introducirá los códigos por ti.',
    vaultSaveDesc: (origin: string) =>
      `El agente llegó a una página de inicio de sesión en ${origin} y no tiene un acceso para ella. Introdúcelo una vez aquí; se cifra en este equipo y se rellena en la página sin que el modelo vea nunca la contraseña.`,
    vaultCodeDesc: (site: string) =>
      `${site} pide un código de un solo uso (SMS, correo o app de autenticación). Introdúcelo aquí y el agente lo escribe en la página; el modelo nunca lo ve.`
  },
  desktop: {
    profileStatus: current =>
      `Agente: ${current}. Usa /profile <nombre> o el selector de "Nuevo chat" para iniciar un chat con otro agente.`,
    unknownProfile: 'Agente desconocido',
    noProfileNamed: (target, available) => `No hay ningún agente llamado "${target}". Disponibles: ${available}`,
    newChatsProfile: name => `Los chats nuevos usarán el agente ${name}.`,
    setProfileFailed: 'No se pudo definir el agente',
    resumeStrandedBody:
      'No se pudo conectar con este chat y se agotaron los reintentos automáticos. Comprueba que Hermes esté en ejecución en la máquina y vuelve a intentarlo.',
    poolSlotTimeoutBody:
      'Hay demasiados agentes en ejecución a la vez para el límite de este equipo. Sube el límite en Configuración → Avanzado o espera a que termine uno y reinténtalo.',
    sessionUnavailable: 'Chat no disponible',
    createSessionFailed: 'No se pudo crear un chat nuevo',
    resumeStrandedTitle: 'No se pudo cargar este chat',
    sessionBusy: 'Chat ocupado',
    cwdStagedMessage: 'Reinicia el backend de escritorio para aplicar cambios de cwd a este chat activo.',
    sessionExported: 'Chat exportado',
    sessionExportFailed: 'No se pudo exportar el chat',
    yoloSystem: active => `YOLO ${active ? 'activado' : 'desactivado'} para este chat`,
    sessionOpenFailed: 'No se pudo abrir este chat'
  },
  tips: {
    items: {
      cron: {
        title: 'Rutinas automáticas',
        text: 'Las rutinas ejecutan una instrucción cada hora, cada noche o con una expresión cron.'
      },
      profiles: {
        title: 'Los agentes son independientes',
        text: 'Cada agente tiene sus propias instrucciones, ajustes y chats.'
      },
      skills: {
        text: 'Las skills son carpetas de instrucciones que el agente carga cuando el trabajo las necesita.'
      },
      messaging: {
        title: 'Tu agente lejos de tu escritorio'
      },
      artifacts: {
        title: 'Todo lo que ha creado tu agente',
        text: 'Imágenes, archivos y enlaces de cada chat, reunidos en tu biblioteca.'
      },
      'command-palette': {
        text: 'Chats, configuración, skills y comandos responden a la paleta.'
      }
    }
  },
  intro: {
    stock: {
      none: [
        'Haz una pregunta, pega un error o indícame un repositorio. Puedo leer código, usar herramientas y ayudarte a entregar.',
        'Describe la tarea con tus palabras. Elegiré las herramientas adecuadas, explicaré mi plan y te consultaré antes de los pasos arriesgados.',
        'Suelta una ruta de archivo, un traceback o una idea en bruto. Investigaré, sugeriré los siguientes pasos y lo mantendré todo reversible.',
        'Busca en el repositorio, edita archivos, ejecuta pruebas, abre PR. Dime el objetivo y yo me ocupo de la parte mecánica.',
        'Escribe una tarea, una pregunta o un fragmento. Recuerdo el chat, cito mis fuentes y me detengo a preguntar cuando tengo dudas.'
      ]
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        'Hermes en esta máquina necesita una actualización para configurar sin chat. Ejecuta hermes update en esa máquina, reinicia Hermes y vuelve a intentarlo. Tus chats no han cambiado.'
    },
    intro: {
      greeting: 'Hola, soy tu agente.'
    }
  },
  connectors: {
    unavailable: 'Los conectores no están disponibles en este chat.',
    disclaimer: 'Conectar es opcional. Autoriza solo las apps que quieras que use el agente.',
    namedAccount: (app: string, alias: string) => `El agente llamó ${alias} a esta cuenta de ${app}.`
  },
  sessionImport: {
    action: 'Importar chat',
    search: 'Buscar en los chats cargados',
    scanError: 'No se pudieron encontrar chats',
    emptyHelp: 'Aquí aparecerán los chats de Claude Code y Codex de esta máquina.',
    searchHelp: 'Prueba con otro título o carpeta, o carga más chats.',
    more: 'Cargar más chats',
    chooseHelp: 'Elige un chat para leer su historial antes de traerlo a Hermes.'
  },
  titlebar: {
    searchTitle: 'Buscar chats, vistas y acciones',
    unreadSessions: count => (count === 1 ? '1 chat sin leer' : `${count} chats sin leer`)
  },
  starmap: {
    emptyDesc: 'A medida que tu agente crea skills y memorias para tu trabajo, aparecerán aquí.'
  },
  artifacts: {
    search: 'Buscar en la biblioteca...',
    refresh: 'Actualizar biblioteca',
    refreshing: 'Actualizando biblioteca',
    indexing: 'Indexando salidas recientes de los chats',
    noArtifactsTitle: 'Aún no hay nada en tu biblioteca',
    noArtifactsDesc: 'Las imágenes generadas y las salidas de archivo aparecerán aquí cuando los chats las produzcan.',
    failedLoad: 'No se pudo cargar la biblioteca',
    colSession: 'Chat',
    partialIndex: (failed, total) => `Se omitieron ${failed} de ${total} chats recientes al indexar la biblioteca.`
  },
  artifactPreview: {
    missingTitle: 'Elemento no disponible',
    missingBody: 'Este elemento ya no está en la biblioteca.'
  },
  statusStack: {
    control: {
      loopDeferredNotice: 'Un objetivo activo controla el chat en este momento.',
      continuationBusy:
        'Objetivo reanudado: el chat está ocupado; detén primero la respuesta actual (botón Stop o Esc) para continuar',
      waitSession: (target: string) => `Esperando al chat ${target}`,
      controlUnavailable: (msg: string) => `Controles del chat no disponibles: ${msg}`
    },
    coding: {
      agentShip: 'Pedir al agente que abra un PR'
    }
  },
  onboarding: {
    headerTitle: 'Vamos a configurar tu agente'
  },
  freeTier: {
    unreachableBody:
      'Hermes no pudo llegar al servicio de Nous para terminar de iniciar tu sesión. Comprueba tu conexión a internet y vuelve a intentarlo. Tu chat sigue aquí.',
    setupFailed: {
      locked:
        'Este chat no puede continuar sin iniciar sesión. Inicia sesión o crea una cuenta gratuita de Nous para seguir.'
    },
    busyBody: (wait: string) =>
      `Hermes no pudo terminar de iniciar tu sesión porque el servicio de Nous está ocupado. Vuelve a intentarlo en ${wait}. Mientras tanto, tu chat sigue aquí.`
  },
  interfaceMode: {
    hint: 'Cambia lo que se muestra, no lo que puede hacer el agente.',
    simple: {
      description: 'Para hablar con tu agente. Barra lateral y chat; sin paneles de terminal, archivos ni diferencias.'
    }
  },
  zones: {
    closeRunningBody:
      'Este chat sigue trabajando o espera tu respuesta. Cerrar la pestaña lo oculta; el chat conserva su progreso y puede volver a abrirse desde la barra lateral.',
    newSessionTab: 'Nueva pestaña de chat'
  }
}
