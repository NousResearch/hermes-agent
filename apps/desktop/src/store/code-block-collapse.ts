import { type Codec, persistentAtom } from '@/lib/persisted'

/**
 * How tall a chat code block may grow before it folds behind an expand control.
 *
 * Desktop-local presentation preference (#54712): folding only changes how this
 * window paints, so it lives beside "Collapse thinking by default" rather than
 * in the shared config.yaml. Compact is the pre-setting behaviour, so existing
 * users see no change until they pick something else.
 */
export type CodeBlockCollapse = 'compact' | 'off' | 'tall'

const STORAGE_KEY = 'hermes.desktop.codeBlockCollapse'

/**
 * Fold thresholds per mode. `thresholdPx` is the scrollHeight above which a
 * block folds; the classes are the folded / expanded max-heights. Tailwind
 * only emits classes it can see as literals, so these stay spelled out.
 */
export const CODE_BLOCK_COLLAPSE_LIMITS = {
  compact: { thresholdPx: 121, foldedClass: 'max-h-[7.5rem]', expandedClass: 'max-h-[40dvh]' },
  tall: { thresholdPx: 321, foldedClass: 'max-h-[20rem]', expandedClass: 'max-h-[70dvh]' }
} as const

const collapseCodec: Codec<CodeBlockCollapse> = {
  decode: raw => (raw === 'tall' || raw === 'off' ? raw : 'compact'),
  encode: value => value
}

export const $codeBlockCollapse = persistentAtom<CodeBlockCollapse>(STORAGE_KEY, 'compact', collapseCodec)

export function setCodeBlockCollapse(mode: CodeBlockCollapse) {
  $codeBlockCollapse.set(mode)
}
