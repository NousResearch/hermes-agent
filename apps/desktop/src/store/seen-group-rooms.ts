import { atom } from 'nanostores'

import type { GatewayGroupRoom } from '@/plugins/hermes-bots/canonical-group-registry'

/** The gateway group chats the redesign's sidebar has listed for the backend this window is on.
 *  Kept outside the section because the sidebar is a sheet below wide and unmounts each time it
 *  closes. Gateway-bound: a re-home wipes it (gateway-switch.ts). Not persisted. */
export const $seenGroupRooms = atom<GatewayGroupRoom[]>([])
