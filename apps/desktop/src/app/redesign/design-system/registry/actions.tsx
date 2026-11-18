import type { VariantProps } from 'class-variance-authority'
import { useState } from 'react'

import { Button, type buttonVariants } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { GenerateButton } from '@/components/ui/generate-button'
import { SplitButton } from '@/components/ui/split-button'
import { TitleMenuTrigger } from '@/components/ui/title-menu-trigger'
import { Plus } from '@/lib/icons'

import { everyOf, type GalleryEntry } from './types'

type ButtonVariant = NonNullable<VariantProps<typeof buttonVariants>['variant']>
type ButtonSize = NonNullable<VariantProps<typeof buttonVariants>['size']>

const BUTTON_VARIANTS = everyOf<ButtonVariant>()(
  'default',
  'secondary',
  'outline',
  'ghost',
  'chip',
  'floating',
  'destructive',
  'destructiveOutline',
  'link',
  'text',
  'textStrong',
  'grip'
)

const TEXT_SIZES = everyOf<Exclude<ButtonSize, `icon${string}` | 'grip'>>()(
  'xs',
  'sm',
  'default',
  'lg',
  'touch',
  'inline',
  'micro'
)

const ICON_SIZES = everyOf<Extract<ButtonSize, `icon${string}`>>()(
  'icon-xs',
  'icon-compact',
  'icon-sm',
  'icon',
  'icon-lg',
  'icon-titlebar'
)

const SPLIT_ACTIONS = [
  { id: 'commit', label: 'Commit' },
  { id: 'commit-push', label: 'Commit & push' }
]

function SplitButtonDemo() {
  const [value, setValue] = useState(SPLIT_ACTIONS[0].id)

  return <SplitButton actions={SPLIT_ACTIONS} onTrigger={() => undefined} onValueChange={setValue} value={value} />
}

function GenerateButtonDemo() {
  const [generating, setGenerating] = useState(false)

  return (
    <GenerateButton
      generating={generating}
      generatingLabel="Generating"
      label="Generate"
      onCancel={() => setGenerating(false)}
      onGenerate={() => setGenerating(true)}
    />
  )
}

export const ACTIONS: readonly GalleryEntry[] = [
  {
    title: 'Button',
    source: 'components/ui/button.tsx',
    use: 'Every action. Pick a variant and a size; never restyle one with className.',
    group: 'actions',
    examples: [
      ...BUTTON_VARIANTS.map(variant => ({
        label: variant,
        render: () => (
          <Button size={variant === 'grip' ? 'grip' : 'default'} variant={variant}>
            {variant === 'grip' ? null : 'Save changes'}
          </Button>
        )
      })),
      ...TEXT_SIZES.map(size => ({
        label: `size ${size}`,
        render: () => (
          <Button size={size} variant="secondary">
            Save
          </Button>
        )
      })),
      ...ICON_SIZES.map(size => ({
        label: `size ${size}`,
        render: () => (
          <Button aria-label="Add" size={size} variant="secondary">
            <Plus />
          </Button>
        )
      })),
      { label: 'disabled', render: () => <Button disabled>Save changes</Button> },
      { label: 'loading', render: () => <Button loading>Save changes</Button> }
    ]
  },
  {
    title: 'CopyButton',
    source: 'components/ui/copy-button.tsx',
    use: 'Copy a value to the clipboard with its own copied / failed feedback.',
    group: 'actions',
    examples: [
      { label: 'button', render: () => <CopyButton text="hermes" /> },
      { label: 'icon', render: () => <CopyButton appearance="icon" text="hermes" /> },
      { label: 'inline', render: () => <CopyButton appearance="inline" text="hermes" /> }
    ]
  },
  {
    title: 'SplitButton',
    source: 'components/ui/split-button.tsx',
    use: 'A primary action with alternates behind a caret; picking one makes it the default.',
    group: 'actions',
    examples: [{ label: 'default', render: () => <SplitButtonDemo /> }]
  },
  {
    title: 'GenerateButton',
    source: 'components/ui/generate-button.tsx',
    use: 'Start a generation and cancel it from the same place.',
    group: 'actions',
    examples: [{ label: 'idle, then generating', render: () => <GenerateButtonDemo /> }]
  },
  {
    title: 'TitleMenuTrigger',
    source: 'components/ui/title-menu-trigger.tsx',
    use: 'A compact "Label ▾" trigger for a menu that belongs to a title.',
    group: 'actions',
    examples: [{ label: 'default', render: () => <TitleMenuTrigger>All agents</TitleMenuTrigger> }]
  }
]
