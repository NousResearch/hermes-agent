import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

// One browsing layout across Skills and Plugins; Installed keeps its own list.
export const $catalogCardView = persistentAtom('hermes.desktop.capabilities.catalogCardView', true, Codecs.bool)

// Community registries (ClawHub, skills.sh, …) are an explicit opt-in fetch:
// the Skills catalog loads the small first-party snapshot by default instead
// of the ~60 MB full feed. Ephemeral on purpose — every visit starts
// first-party; asking is what loads the community registries again.
export const $skillsCommunityFeed = atom(false)
