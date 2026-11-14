import './pixel-icon.css'

import { forwardRef } from 'react'
import type * as React from 'react'

import { PIXEL_ICONS, type PixelIconName } from '@/lib/pixel-icon-registry'
import { KEEP_CLASSIC, RTL_MIRRORED, TABLER_PIXEL, type TablerAlias } from '@/lib/pixel-icons'
import { usePortalPresentation } from '@/store/shell-presentation'

export interface PixelIconProps extends React.SVGProps<SVGSVGElement> {
  name: PixelIconName
}

/** A Pixelarticons glyph: `currentColor` fill on the set's 24-unit grid. Rendered with the
 *  browser's default anti-aliasing, as Figma draws these at 12–20px: `crispEdges` snaps the
 *  2-unit strokes to uneven 1/2px widths at every size that is not a multiple of 24. */
export function PixelIcon({ name, ...props }: PixelIconProps) {
  const Glyph = PIXEL_ICONS[name]

  return (
    <Glyph
      aria-hidden="true"
      data-icon-set="pixel"
      data-pixel-icon={name}
      data-rtl-mirror={RTL_MIRRORED.has(name) ? '' : undefined}
      focusable="false"
      {...props}
    />
  )
}

/** The props a classic icon component (Tabler, lucide) shares with the pixel glyph. */
interface ClassicIconProps extends Omit<React.SVGProps<SVGSVGElement>, 'ref' | 'stroke'> {
  size?: number | string
  stroke?: number | string
  title?: string
}

type ClassicIcon<P> = React.ForwardRefExoticComponent<P & React.RefAttributes<SVGSVGElement>>

/** An icon that draws its classic glyph in the classic shell and its Pixelarticons twin in the
 *  portal shell, flipping live with the shell toggle. `stroke` and `title` are Tabler's own API and
 *  have no pixel meaning; `size` and `color` carry over. */
export function shellIcon<P extends ClassicIconProps>(classic: ClassicIcon<P>, alias: TablerAlias): ClassicIcon<P> {
  const mapped = TABLER_PIXEL[alias]

  if (mapped === KEEP_CLASSIC) {
    return classic
  }

  const Classic = classic as ClassicIcon<ClassicIconProps>

  const ShellIcon = forwardRef<SVGSVGElement, ClassicIconProps>(function ShellIcon(props, ref) {
    const portal = usePortalPresentation()

    if (!portal) {
      return <Classic ref={ref} {...props} />
    }

    const { color, size = 24, stroke: _stroke, style, title: _title, ...rest } = props

    return (
      <PixelIcon
        height={size}
        ref={ref}
        width={size}
        {...rest}
        name={mapped}
        style={color === undefined ? style : { color, ...style }}
      />
    )
  })

  ShellIcon.displayName = `ShellIcon(${alias})`

  return ShellIcon as ClassicIcon<P>
}
