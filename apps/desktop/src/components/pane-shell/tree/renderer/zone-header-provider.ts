import type { ReactNode } from 'react'

import type { GroupNode } from '../model'

import { tabStripVisibleForZone } from './strip-visibility'

export type ZoneStripInput = Omit<Parameters<typeof tabStripVisibleForZone>[0], 'mode'>

/** What a zone draws above its panes: whether its tab strip shows, and a shell header beside it. */
export interface ZoneHeaderChrome {
  conversationHeader: ReactNode
  stripVisible: boolean
}

export type ZoneHeaderHook = (node: GroupNode, zone: ZoneStripInput) => ZoneHeaderChrome

const useSharedStrip: ZoneHeaderHook = (node, zone) => ({
  conversationHeader: null,
  stripVisible: tabStripVisibleForZone({ ...zone, mode: node.tabStrip })
})

let useProvidedZoneHeader: ZoneHeaderHook = useSharedStrip

/** The app shell supplies its zone header (the redesign's conversation header) without this
 *  generic renderer importing the app. It is a hook: install it once at boot, before the first
 *  zone renders, and never swap it while zones are mounted. */
export function provideZoneHeader(hook: ZoneHeaderHook): void {
  useProvidedZoneHeader = hook
}

export function useZoneHeader(node: GroupNode, zone: ZoneStripInput): ZoneHeaderChrome {
  return useProvidedZoneHeader(node, zone)
}
