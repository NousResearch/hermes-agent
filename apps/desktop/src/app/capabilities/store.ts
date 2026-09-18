import { Codecs, persistentAtom } from '@/lib/persisted'

// One browsing layout across Skills and Plugins; Installed keeps its own list.
// (Each tab's own list state — sort direction — lives in that tab's folder.)
export const $catalogCardView = persistentAtom('hermes.desktop.capabilities.catalogCardView', true, Codecs.bool)
