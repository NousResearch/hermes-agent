import type { VariantProps } from 'class-variance-authority'

import { DecodeText } from '@/components/ui/decode-text'
import { DiffCount } from '@/components/ui/diff-count'
import { FadeText } from '@/components/ui/fade-text'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { Kbd, KbdCombo, KbdGroup, type kbdVariants } from '@/components/ui/kbd'
import { PathText } from '@/components/ui/path-text'

import { everyOf, type GalleryEntry } from './types'

type KbdVariant = NonNullable<VariantProps<typeof kbdVariants>['variant']>
type KbdSize = NonNullable<VariantProps<typeof kbdVariants>['size']>

const KBD_VARIANTS = everyOf<KbdVariant>()('default', 'ghost', 'capturing', 'inverted')
const KBD_SIZES = everyOf<KbdSize>()('sm', 'md')

export const TEXT: readonly GalleryEntry[] = [
  {
    title: 'Kbd',
    source: 'components/ui/kbd.tsx',
    use: 'A key or a shortcut, as a keycap; KbdCombo spells a keybind from its combo string.',
    group: 'text',
    examples: [
      ...KBD_VARIANTS.map(variant => ({ label: variant, render: () => <Kbd variant={variant}>K</Kbd> })),
      ...KBD_SIZES.map(size => ({ label: `size ${size}`, render: () => <Kbd size={size}>Esc</Kbd> })),
      { label: 'KbdGroup', render: () => <KbdGroup keys={['⌘', '⇧', 'N']} /> },
      { label: 'KbdCombo', render: () => <KbdCombo combo="mod+k" /> }
    ]
  },
  {
    title: 'PathText',
    source: 'components/ui/path-text.tsx',
    use: 'A filesystem path on one line: home-relative, the middle elided, the full path in a tip.',
    group: 'text',
    examples: [{ label: 'default', render: () => <PathText path="~/projects/notes/reading-list.md" /> }]
  },
  {
    title: 'HighlightMatches',
    source: 'components/ui/highlight-matches.tsx',
    use: 'Mark where a search query matched inside a result.',
    group: 'text',
    examples: [{ label: 'query "dig"', render: () => <HighlightMatches query="dig" text="Morning digest" /> }]
  },
  {
    title: 'FadeText',
    source: 'components/ui/fade-text.tsx',
    use: 'One line that fades out at its end instead of an ellipsis, only while it overflows.',
    group: 'text',
    examples: [
      {
        label: 'overflowing',
        render: () => (
          <FadeText className="ds-fade-text">A chat title long enough to run past the end of its row</FadeText>
        )
      }
    ]
  },
  {
    title: 'DecodeText',
    source: 'components/ui/decode-text.tsx',
    use: 'Text that resolves from scrambled glyphs, for a value that just arrived.',
    group: 'text',
    examples: [{ label: 'on mount', render: () => <DecodeText text="Thinking" /> }]
  },
  {
    title: 'DiffCount',
    source: 'components/ui/diff-count.tsx',
    use: "A change's added and removed line counts.",
    group: 'text',
    examples: [{ label: 'default', render: () => <DiffCount added={42} removed={7} /> }]
  }
]
