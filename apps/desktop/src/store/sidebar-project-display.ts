import { type Codec, persistentAtom } from '@/lib/persisted'

export type ProjectLabelMode = 'short' | 'full'

const SHOW_EMPTY_WORKTREE_LANES_KEY = 'hermes.desktop.sidebar.showEmptyWorktreeLanes'
const PROJECT_LABEL_MODE_KEY = 'hermes.desktop.sidebar.projectLabelMode'

const projectLabelModeCodec: Codec<ProjectLabelMode> = {
  decode: raw => (raw === 'full' ? raw : 'short'),
  encode: value => value
}

const showEmptyWorktreeLanesCodec: Codec<boolean> = {
  decode: raw => raw !== 'false',
  encode: value => String(value)
}

// Both defaults preserve the existing sidebar, so upgrading does not rearrange
// a user's projects until they deliberately choose a different presentation.
export const $showEmptyWorktreeLanes = persistentAtom(SHOW_EMPTY_WORKTREE_LANES_KEY, true, showEmptyWorktreeLanesCodec)
export const $projectLabelMode = persistentAtom<ProjectLabelMode>(PROJECT_LABEL_MODE_KEY, 'short', projectLabelModeCodec)

export function setShowEmptyWorktreeLanes(value: boolean) {
  $showEmptyWorktreeLanes.set(value)
}

export function setProjectLabelMode(value: ProjectLabelMode) {
  $projectLabelMode.set(value)
}
