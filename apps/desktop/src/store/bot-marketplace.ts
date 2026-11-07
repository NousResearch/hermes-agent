import { atom } from 'nanostores'

export interface BotMarketplaceRequest {
  catalog: string
  nonce: number
}

export const $botMarketplaceRequest = atom<BotMarketplaceRequest | null>(null)

let requestNonce = 0

/** Opens the native review flow. Only the catalog key crosses this boundary. */
export function openBotMarketplaceRequest(catalog: string): void {
  const key = catalog.trim()

  if (!key) {
    return
  }

  requestNonce += 1
  $botMarketplaceRequest.set({ catalog: key, nonce: requestNonce })
}

export function closeBotMarketplaceRequest(): void {
  $botMarketplaceRequest.set(null)
}
