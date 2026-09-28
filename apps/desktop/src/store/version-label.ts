/**
 * `display.version_label` — how the version pill names the running version.
 *
 * 'release' (the opt-in) strips the `+<distance>` suffix from the short
 * label: on an up-to-date install the distance past the last release tag
 * reads as "N commits behind" exactly where the same row reports actual
 * staleness (#123538). The commit distance stays in the tooltip and the
 * expanded version details, which read the unshortened payload.
 *
 * Default 'release+distance' keeps today's label everywhere.
 */

import { atom } from 'nanostores'

export type VersionLabelMode = 'release' | 'release+distance'

export const $versionLabel = atom<VersionLabelMode>('release+distance')

export function setVersionLabelFromConfig(value: unknown): void {
  $versionLabel.set(value === 'release' ? 'release' : 'release+distance')
}
