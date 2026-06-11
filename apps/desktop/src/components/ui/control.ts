import { cva, type VariantProps } from 'class-variance-authority'

// Single source of truth for non-composer form-control chrome — Input,
// Textarea, and SelectTrigger all consume this. Mirrors `buttonVariants`:
// 2.5px radius, 12px text, padding-driven sizing (no fixed heights). The visual
// chrome (background, border tint, hover, focus glow, invalid state) comes from
// the `desktop-input-chrome` CSS so every control shares one exact look.
export const controlVariants = cva(
  'desktop-input-chrome w-full min-w-0 rounded-[2.5px] border text-xs leading-4 text-foreground outline-none placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
  {
    variants: {
      size: {
        xs: 'px-2 py-0.5 text-[0.6875rem] leading-4',
        sm: 'px-2 py-1',
        default: 'px-2.5 py-1.5',
        lg: 'px-3 py-2 text-sm leading-5'
      }
    },
    defaultVariants: {
      size: 'default'
    }
  }
)

export type ControlVariantProps = VariantProps<typeof controlVariants>
