import { translateNow } from '@/i18n'
import { hostHas } from '@/lib/host-capabilities'

/**
 * Boot/loading copy for where this renderer runs.
 *
 * The Webapp (a browser tab the gateway serves) runs the same cold-boot flow as native Desktop
 * but is not the Desktop app, so its loading steps and ready message never name one — it takes
 * `boot.webapp`. Native Desktop (Electron) keeps the shipped `boot.steps`/`boot.ready` copy.
 *
 * Read per call (never cached): the browser bridge writes the host marker after some modules
 * evaluate, so `hostHas` must be asked at the moment the step is produced.
 */

type BootStepKey = 'startingDesktopConnection' | 'startingHermesDesktop'

/** The Webapp step that replaces each native step (see `boot.webapp.steps`). */
const WEBAPP_STEP: Record<BootStepKey, string> = {
  startingDesktopConnection: 'boot.webapp.steps.startingConnection',
  startingHermesDesktop: 'boot.webapp.steps.startingHermes'
}

export function bootStepCopy(key: BootStepKey): string {
  return hostHas('webappCopy') ? translateNow(WEBAPP_STEP[key]) : translateNow(`boot.steps.${key}`)
}

export function bootReadyCopy(): string {
  return hostHas('webappCopy') ? translateNow('boot.webapp.ready') : translateNow('boot.ready')
}
