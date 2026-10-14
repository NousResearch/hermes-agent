import { vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

/** Install a fake `window.hermesDesktop.api`, the preload bridge every REST helper sends through. */
export function installRestBridge(answer: (request: HermesApiRequest) => object) {
  const api = vi.fn(async (request: HermesApiRequest) => answer(request))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}
