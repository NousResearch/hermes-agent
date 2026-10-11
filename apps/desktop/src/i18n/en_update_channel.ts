import type { UpdateChannelCopy } from './types_update_channel'

export const enUpdateChannel: UpdateChannelCopy = {
  title: 'Update channel',
  description: 'Stable installs published releases only. Every commit follows the main branch as changes land.',
  stable: 'Stable releases',
  main: 'Every commit',
  stableConfirmTitle: 'Switch to stable releases?',
  stableConfirmBody:
    'Hermes will move to the latest published release, which can be older than the code you run now. Data written by newer code may not read correctly, so back it up first.',
  stableConfirm: 'Switch to stable',
  failed: 'Could not change the update channel',
  detailsLabel: 'Channel',
  branch: (b: string) => `Branch: ${b}`,
  change: 'Change',
  latestRelease: (v: string) => `You’re on the latest stable release (${v}).`
}
