import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const esErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'Este chat pertenece a una tarea programada',
    body: 'Solo ejecuta esa tarea, así que tu mensaje no se ejecutó. Empieza un chat nuevo para enviarlo.'
  },
  session_read_only: {
    title: 'Este chat es de solo lectura',
    body: 'Es historial guardado que Hermes puede mostrar pero no continuar, así que tu mensaje no se ejecutó. Empieza un chat nuevo para enviarlo.'
  },
  workspace_unavailable: {
    title: 'La carpeta de este chat ya no existe',
    body: 'La carpeta en la que trabaja este chat se movió o se eliminó, así que tu mensaje no se ejecutó. Vuelve a crear la carpeta o usa «Mover a proyecto» para darle otra al chat, y vuelve a intentarlo.'
  },
  launch_credentials_unavailable: {
    title: 'La clave de API de este chat cambió',
    body: 'Este chat se inició con una clave de API que desde entonces cambió o se eliminó, así que tu mensaje no se envió. Un chat abierto nunca cambia por sí solo a otra clave. Usa «Iniciar sesión nueva» para usar tu clave actual.'
  },
  local_runtime_unavailable: {
    title: 'No se puede acceder al servidor de modelos local',
    body: 'El servidor de modelos local aún no está en ejecución: puede que todavía se esté iniciando. Reintenta en un momento o cambia a otro modelo.',
    bodyNoRetry:
      'El servidor de modelos local está apagado. Inícialo desde la app de escritorio de Hermes o con `hermes model` en una terminal, o cambia a otro modelo.'
  }
}
