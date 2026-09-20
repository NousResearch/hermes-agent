import { type Codec, persistentAtom } from '@/lib/persisted'

export const PANE_DISTRIBUTION_MODES = ['preserve', 'equal-flex', 'equal-all'] as const
export type PaneDistributionMode = (typeof PANE_DISTRIBUTION_MODES)[number]

const PANE_DISTRIBUTION_STORAGE_KEY = 'hermes.desktop.paneDistributionMode.v1'

const codec: Codec<PaneDistributionMode> = {
  decode: raw =>
    (PANE_DISTRIBUTION_MODES as readonly string[]).includes(raw) ? (raw as PaneDistributionMode) : 'preserve',
  encode: value => (value === 'preserve' ? null : value)
}

/** How structural pane additions and moves distribute direct split tracks. */
export const $paneDistributionMode = persistentAtom<PaneDistributionMode>(
  PANE_DISTRIBUTION_STORAGE_KEY,
  'preserve',
  codec
)

export function setPaneDistributionMode(mode: PaneDistributionMode) {
  $paneDistributionMode.set(mode)
}
