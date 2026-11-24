import { atom } from 'nanostores'

import type { GatewayGroupRoom } from '@/plugins/hermes-bots/canonical-group-registry'

/** The gateway group chats the redesign's sidebar has listed in this window, for every machine it
 *  has shown. Kept outside the section because the sidebar is a sheet below wide and unmounts each
 *  time it closes. Not persisted: a reload starts from what the active agent lists. */
export const $seenGroupRooms = atom<GatewayGroupRoom[]>([])
