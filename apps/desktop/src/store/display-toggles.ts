/**
 * Appearance switches the BACKEND also has to know about.
 *
 * Most of what Settings → Appearance controls is the renderer's own business.
 * A few switches aren't: they decide whether a tool is in the model's schema at
 * all (`react_to_message`, `tip`, `tour`), and that decision is made where the
 * agent runs. So the value has to travel to whichever gateway the app is
 * actually talking to — local, SSH, plain URL, or cloud — which is why this is
 * `config.set` on the live connection and not an env var on some process.
 *
 * Two moments matter, and shipping only the first is why the reactions toggle
 * has never worked end to end:
 *
 * - **On change**, because that is the user answering.
 * - **On connect**, because a gateway the app has never spoken to holds no
 *   answer at all — and for a switch that defaults ON, silence reads as consent
 *   to exactly the thing the user turned off.
 *
 * The connect push is deliberately narrow: only settings the user has actually
 * touched (a stored value exists) are re-sent. Otherwise a fresh install would
 * broadcast its defaults over a `config.yaml` somebody hand-edited on the
 * server, which is a worse failure than the one being fixed.
 */

import type { ReadableAtom } from 'nanostores'

import { readKey } from '@/lib/storage'
import { $gateway, activeGateway } from '@/store/gateway'

interface Mirror {
  configKey: string
  read: () => boolean
  storageKey: string
}

const mirrors: Mirror[] = []

function push(mirror: Mirror): void {
  void activeGateway()
    ?.request('config.set', { key: mirror.configKey, value: mirror.read() ? 'true' : 'false' })
    .catch(() => {
      // Not connected, or a gateway too old to know the key. The next toggle
      // and the next connection both try again.
    })
}

/** Keep `display.<configKey>` on the live gateway in step with a renderer atom. */
export function mirrorDisplayToggle(configKey: string, storageKey: string, $enabled: ReadableAtom<boolean>): void {
  if (typeof window === 'undefined') {
    return
  }

  const mirror: Mirror = { configKey, read: () => $enabled.get(), storageKey }

  mirrors.push(mirror)
  // listen, not subscribe: fire on CHANGE only. Module init must not write.
  $enabled.listen(() => push(mirror))
}

if (typeof window !== 'undefined') {
  $gateway.listen(() => {
    for (const mirror of mirrors) {
      if (readKey(mirror.storageKey) !== null) {
        push(mirror)
      }
    }
  })
}
