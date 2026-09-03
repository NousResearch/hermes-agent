import { type Codec, persistentAtom } from '@/lib/persisted'

// How many recent sessions each project/profile shows in the sidebar before
// the rest are reachable by scrolling (or the Show More control on workspace
// lanes). 'all' lifts the cap entirely.
export type ProjectSessionPreview = 3 | 4 | 5 | 8 | 'all'

const STORAGE_KEY = 'hermes.desktop.projectSessionPreview'

export const PROJECT_PREVIEW_OPTIONS = [3, 4, 5, 8, 'all'] as const

const previewCodec: Codec<ProjectSessionPreview> = {
  decode: raw => {
    if (raw === 'all') {
      return 'all'
    }

    const parsed = Number(raw)

    return (PROJECT_PREVIEW_OPTIONS as readonly ProjectSessionPreview[]).includes(parsed as 3 | 4 | 5 | 8)
      ? (parsed as 3 | 4 | 5 | 8)
      : 3
  },
  encode: value => String(value)
}

export const $projectSessionPreview = persistentAtom<ProjectSessionPreview>(STORAGE_KEY, 3, previewCodec)

export function setProjectSessionPreview(value: ProjectSessionPreview) {
  $projectSessionPreview.set(value)
}
