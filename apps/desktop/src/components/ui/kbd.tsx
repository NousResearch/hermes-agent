import { cva, type VariantProps } from 'class-variance-authority'
import * as React from 'react'

import { comboTokens } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'

const COMPACT_KEY = /^[\p{L}\p{N}⌘⌥⇧⌃↵⇥⌫↑↓←→@/?]$/u

const kbdSurface = [
  'border-[color-mix(in_srgb,var(--ui-stroke-secondary)_75%,transparent)]',
  'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_94%,var(--dt-foreground)_6%)]',
  'text-[color-mix(in_srgb,var(--dt-foreground)_58%,transparent)]',
  'shadow-[0_1px_0_0_color-mix(in_srgb,var(--ui-stroke-tertiary)_85%,transparent),0_1px_2px_0_color-mix(in_srgb,var(--dt-foreground)_7%,transparent)]'
]

const kbdVariants = cva(
  'inline-flex shrink-0 items-center justify-center border [font-family:var(--dt-font-kbd)] font-normal leading-none select-none',
  {
    variants: {
      variant: {
        default: kbdSurface,
        ghost: [
          ...kbdSurface,
          'text-[color-mix(in_srgb,var(--dt-foreground)_38%,transparent)]',
          'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_72%,var(--dt-foreground)_3%)]',
          'border-[color-mix(in_srgb,var(--ui-stroke-tertiary)_80%,transparent)]'
        ],
        capturing: [
          'border-[color-mix(in_srgb,var(--theme-primary)_50%,var(--ui-stroke-secondary))]',
          'bg-[color-mix(in_srgb,var(--theme-primary)_10%,var(--ui-bg-elevated))]',
          'text-[color-mix(in_srgb,var(--theme-primary)_88%,transparent)]',
          'shadow-none'
        ],
        inverted: [
          'border-[color-mix(in_srgb,currentColor_22%,transparent)]',
          'bg-[color-mix(in_srgb,currentColor_12%,transparent)]',
          'text-[color-mix(in_srgb,currentColor_88%,transparent)]',
          'shadow-[0_1px_0_0_color-mix(in_srgb,currentColor_18%,transparent)]'
        ]
      },
      size: {
        sm: 'rounded-[0.2rem] text-[0.625rem]',
        md: 'rounded-[0.25rem] text-[0.6875rem]'
      }
    },
    defaultVariants: {
      variant: 'default',
      size: 'md'
    }
  }
)

function kbdShapeClass(label: string, size: 'sm' | 'md' | null | undefined): string {
  const compact = COMPACT_KEY.test(label)

  if (size === 'sm') {
    return compact ? 'size-[1.125rem] px-0' : 'h-[1.125rem] min-w-[1.125rem] px-1'
  }

  return compact ? 'size-[1.375rem] px-0' : 'h-[1.375rem] min-w-[1.375rem] px-1.5'
}

interface KbdProps extends React.ComponentProps<'kbd'>, VariantProps<typeof kbdVariants> {}

function Kbd({ children, className, size, variant, ...props }: KbdProps) {
  const label = typeof children === 'string' ? children : ''

  return (
    <kbd
      className={cn(kbdVariants({ size, variant }), kbdShapeClass(label, size), className)}
      data-slot="kbd"
      {...props}
    >
      {children}
    </kbd>
  )
}

interface KbdGroupProps extends Omit<React.ComponentProps<'span'>, 'children'>, VariantProps<typeof kbdVariants> {
  keys: string[]
}

function KbdGroup({ className, keys, size, variant, ...props }: KbdGroupProps) {
  return (
    <span
      aria-label={keys.join(' ')}
      className={cn('inline-flex shrink-0 items-center gap-1', className)}
      data-slot="kbd-group"
      {...props}
    >
      {keys.map((key, index) => (
        <Kbd key={`${key}-${index}`} size={size} variant={variant}>
          {key}
        </Kbd>
      ))}
    </span>
  )
}

interface KbdComboProps extends Omit<KbdGroupProps, 'keys'> {
  combo: string
}

function KbdCombo({ combo, ...props }: KbdComboProps) {
  return <KbdGroup keys={comboTokens(combo)} {...props} />
}

export { Kbd, KbdCombo, KbdGroup, kbdVariants }
