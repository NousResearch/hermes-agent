import { atom } from 'nanostores'

import type { HermesGateway } from '@/hermes'

// The active gateway instance, exposed for inline message-stream components
// (e.g. inline ClarifyTool) that need to call gateway methods without having
// the instance threaded down through props from `ChatView`.
export const $gateway = atom<HermesGateway | null>(null)

export function setGateway(gateway: HermesGateway | null): void {
  if ($gateway.get() === gateway) {
    return
  }

  $gateway.set(gateway)
}
