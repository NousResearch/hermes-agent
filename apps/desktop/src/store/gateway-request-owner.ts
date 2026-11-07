import type { ConnectionOwner } from '@/global'

export type SpawnPriority = 'foreground' | 'background'

export interface GatewayRequestOptions {
  expectedConnectionOwner?: ConnectionOwner
  expectedOwner?: { profile: string; connectionOwner: ConnectionOwner }
  spawnPriority?: SpawnPriority
}

export function gatewayRequestOwnerValidator(
  connectionId: null | string,
  profile: string,
  { expectedConnectionOwner, expectedOwner, spawnPriority = 'background' }: GatewayRequestOptions
): () => Promise<void> {
  return async () => {
    if (!expectedOwner) {return}

    if (!window.hermesDesktop?.getConnectionFor) {
      throw new Error('This Desktop build cannot validate the Settings gateway owner. Update Hermes Desktop.')
    }

    await window.hermesDesktop.getConnectionFor({
      connectionId,
      expectedConnectionOwner,
      expectedOwner,
      profile,
      priority: spawnPriority
    })
  }
}
