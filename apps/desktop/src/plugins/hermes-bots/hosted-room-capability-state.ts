import { atom } from '@hermes/plugin-sdk'

import type { HostedRoomCapability } from './hosted-room-client'

/** Current Desktop inventory evidence, never restored from display mirrors. */
export const $hostedRoomCapabilities = atom<Record<string, HostedRoomCapability>>({})
