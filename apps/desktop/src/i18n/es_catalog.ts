import type { TranslationOverrides } from './define-locale'

// Provider-source copy for Capabilities > Skills, wired into es.ts as `catalog`.
export const esCatalog = {
  provider: {
    review: (action: string) => `Revisar ${action}`,
    confirm: (action: string) => `Confirmar ${action}`,
    actions: { install: 'la instalación', update: 'la actualización', uninstall: 'la desinstalación' },
    managed: 'Gestionado por esta fuente. Revisa los cambios aquí en lugar de editar o eliminar archivos locales.',
    loading: 'Cargando la fuente…',
    empty: 'Esta fuente no tiene skills que coincidan.',
    signIn: 'Inicia sesión a través del plugin proveedor para explorar esta fuente.',
    unavailable: 'Esta fuente no está disponible. Tus skills locales no han cambiado.',
    incomplete: 'La revisión está incompleta o la política del proveedor no permite confirmar.',
    changed: 'Esta revisión ya no está vigente. Ciérrala y solicita una nueva revisión.',
    unknown: 'Aún no se conoce el resultado. Recupera la confirmación original antes de iniciar otra acción.',
    recover: 'Recuperar la operación original',
    finalizing: 'Finalizando el estado local…',
    setup: 'Solo indicaciones de configuración. Confirmar no ejecuta estas instrucciones.',
    more: (source: string) => `Cargar más de ${source}`,
    preview: 'Vista previa de la skill',
    status: {
      not_installed: 'No instalada', installed: 'Instalada', removed: 'Eliminada fuera de esta fuente',
      unmanaged_local: 'Copia local no gestionada', pending: 'Operación pendiente', unavailable: 'Fuente no disponible',
      unknown: 'Resultado desconocido', completed: 'Completada', failed: 'Operación rechazada'
    }
  }
} satisfies TranslationOverrides['catalog']
