import { Codecs, persistentAtom } from '@/lib/persisted'

// Window-presentation preference: force the canonical slug subtitle under every
// model row. Off by default — a row only shows its slug when its display name
// collides with a sibling in the same provider's catalog.
export const $showModelSlugs = persistentAtom('hermes.desktop.show-model-slugs', false, Codecs.bool)

export const toggleShowModelSlugs = () => $showModelSlugs.set(!$showModelSlugs.get())
