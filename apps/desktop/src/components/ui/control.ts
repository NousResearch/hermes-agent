import { cva, type VariantProps } from 'class-variance-authority'

// Single source of truth for non-composer form-control chrome — Input,
// Textarea, and SelectTrigger all consume this. Mirrors `buttonVariants`:
// 2.5px radius, 12px text, padding-driven sizing (no fixed heights). The visual
// chrome (background, border tint, hover, focus glow, invalid state) comes from
// the `desktop-input-chrome` CSS so every control shares one exact look.
export const controlVariants = cva(
  'w-full min-w-0 rounded-[2.5px] text-xs leading-4 text-foreground outline-none placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
  {
    variants: {
      size: {
        xs: 'text-[0.6875rem] leading-4',
        sm: '',
        default: '',
        lg: 'text-sm leading-5'
      },
      // `default` is a boxed field. `plain` is the same type, no box — an
      // editable heading or an id sitting in chrome that isn't a form row.
      chrome: {
        default: 'desktop-input-chrome border',
        plain: 'border-0 bg-transparent shadow-none'
      }
    },
    compoundVariants: [
      { chrome: 'default', size: 'xs', class: 'px-2 py-0.5' },
      { chrome: 'default', size: 'sm', class: 'px-2 py-1' },
      { chrome: 'default', size: 'default', class: 'px-2.5 py-1.5' },
      { chrome: 'default', size: 'lg', class: 'px-3 py-2' }
    ],
    defaultVariants: {
      size: 'default',
      chrome: 'default'
    }
  }
)

export type ControlVariantProps = VariantProps<typeof controlVariants>
