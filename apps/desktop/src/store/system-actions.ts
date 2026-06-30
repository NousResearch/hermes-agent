import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

// True while a gateway restart is in flight — drives the statusbar gateway
// indicator (glyph spinner) so the restart shows up where users already look,
// instead of a toast that vanishes or a generic "Agents running" counter.
export const $gatewayRestarting = atom(false)

// Remote-first Reuben Desktop does not supervise or restart the backend.
// Keep this as an inert compatibility action for any stale caller.
export async function runGatewayRestart(): Promise<void> {
  $gatewayRestarting.set(false)
  notify({
    kind: 'info',
    message: 'Restart the remote backend through its own service manager or deployment channel.',
    title: translateNow('commandCenter.restartGateway')
  })
}
