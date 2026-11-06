import type { ComponentPropsWithoutRef, ComponentType, ReactNode, SVGProps } from 'react'
import { forwardRef } from 'react'

import { cn } from '@/lib/utils'

/**
 * The square identity chip: a brand glyph on a 16% tint of its own color, or
 * a letter monogram when we have no mark for the thing.
 *
 * One treatment, three surfaces — messaging platforms, the MCP servers tab,
 * connector cards — which is one more than it took for the copies to drift.
 * They had already diverged on radius, on glyph size, and on what an unknown
 * name looks like. Size and radius are the caller's call (`className`);
 * everything else is fixed here so a Slack row and a Linear card wear the
 * same chip.
 */
export interface AvatarChipBrand {
  Icon?: ComponentType<SVGProps<SVGSVGElement>>
  color: string
  /** Optional surface color for marks that are designed on a solid field. */
  backgroundColor?: string
  /** Optional mark color, used when the glyph is intentionally monochrome. */
  glyphColor?: string
  /** Marks that are black or white by design (GitHub, Notion, Unreal) follow
   *  the surrounding text color instead of vanishing into the theme. */
  monochrome?: boolean
  monogram?: string
}

interface AvatarChipProps extends Omit<ComponentPropsWithoutRef<'span'>, 'children'> {
  brand?: AvatarChipBrand | null
  /** Replaces the mark entirely — a resolved favicon, a spinner. */
  children?: ReactNode
  /** The monogram's source, and what a screen reader gets. */
  name: string
  /** Painted over the mark: a status dot, a badge. */
  overlay?: ReactNode
}

/** Proportional to the chip so one glyph size serves every size the chip is
 *  used at — 14px inside the 24px row chip, matching what it replaced. */
const GLYPH_CLASS = 'size-[58%]'

export const AvatarChip = forwardRef<HTMLSpanElement, AvatarChipProps>(function AvatarChip(
  { brand, children, className, name, overlay, style, ...rest },
  ref
) {
  const Icon = brand?.Icon

  return (
    <span
      className={cn(
        // Not `overflow-hidden`: an overlay dot sits proud of the corner. A
        // caller painting a raster mark clips at its own layer.
        'relative inline-grid size-6 shrink-0 place-items-center rounded-md text-[length:var(--conversation-caption-font-size)] font-medium',
        !brand && 'bg-(--ui-bg-tertiary) text-(--ui-text-tertiary)',
        className
      )}
      ref={ref}
      style={
        brand
          ? {
              backgroundColor: brand.backgroundColor ?? `color-mix(in srgb, ${brand.color} 16%, transparent)`,
              color: brand.glyphColor ?? (brand.monochrome ? undefined : brand.color),
              ...style
            }
          : style
      }
      {...rest}
    >
      {children ?? (Icon ? <Icon aria-hidden className={GLYPH_CLASS} /> : (brand?.monogram ?? monogramFor(name)))}
      {overlay}
    </span>
  )
})

export const monogramFor = (name: string): string => name.trim().charAt(0).toUpperCase()
