import { persistentAtom } from '@/lib/persisted'

export type ReviewDiffLayout = 'split' | 'unified'

// Device-wide presentation preference, like the Review tree/list choice.
export const $reviewDiffLayout = persistentAtom<ReviewDiffLayout>('hermes.desktop.reviewDiffLayout', 'unified', {
  decode: raw => (raw === 'split' ? 'split' : 'unified'),
  encode: value => value
})
