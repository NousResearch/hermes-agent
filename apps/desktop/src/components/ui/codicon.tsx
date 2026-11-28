import type { Icon } from '@tabler/icons-react'
import type * as React from 'react'

import { pixelForCodicon } from '@/lib/pixel-icons'
import { cn } from '@/lib/utils'
import { usePortalPresentation } from '@/store/shell-presentation'

import { PixelIcon } from './pixel-icon'

export interface CodiconProps extends React.HTMLAttributes<HTMLElement> {
  name: string
  size?: number | string
  spinning?: boolean
}

/** A codicon font glyph — or, under the portal shell, its Pixelarticons twin. The pixel SVG keeps
 *  the `codicon codicon-<name>` classes and a 1em box, so every rule that sizes, colours or spins a
 *  codicon by font-size and class applies to it unchanged. */
export function Codicon({ className, name, size, spinning, style, ...props }: CodiconProps) {
  const pixel = pixelForCodicon(name)
  const portal = usePortalPresentation()
  const classes = cn('codicon', `codicon-${name}`, spinning && 'codicon-modifier-spin', className)
  const sized = { ...(size != null && size !== '' ? { fontSize: size } : {}), ...style }

  if (portal && pixel) {
    return (
      <PixelIcon
        height="1em"
        width="1em"
        {...(props as React.SVGAttributes<SVGSVGElement>)}
        className={classes}
        name={pixel}
        style={sized}
      />
    )
  }

  return <i aria-hidden="true" className={classes} style={sized} {...props} />
}

/** Wrap a codicon as a Tabler-shaped icon for nav rows that expect `IconComponent`. */
export function codiconIcon(name: string): Icon {
  function CodiconIcon({ className }: { className?: string }) {
    return <Codicon aria-hidden className={cn('leading-none', className)} name={name} size="1em" />
  }

  CodiconIcon.displayName = `Codicon(${name})`

  return CodiconIcon as Icon
}
