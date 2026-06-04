import * as React from 'react'
import codiconSprite from '@vscode/codicons/dist/codicon.svg?raw'

import { cn } from '@/lib/utils'

export interface CodiconProps extends React.SVGAttributes<SVGSVGElement> {
  name: string
  size?: number | string
  spinning?: boolean
}

const SPRITE_ID = 'hermes-codicon-sprite'
const SYMBOL_IDS = new Set([...codiconSprite.matchAll(/<symbol\b[^>]*\bid="([^"]+)"/g)].map(match => match[1]))
const useIsomorphicLayoutEffect = typeof window === 'undefined' ? React.useEffect : React.useLayoutEffect

function ensureCodiconSprite() {
  if (typeof document === 'undefined' || document.getElementById(SPRITE_ID)) {
    return
  }

  const container = document.createElement('div')
  container.id = SPRITE_ID
  container.setAttribute('aria-hidden', 'true')
  container.style.display = 'none'
  container.innerHTML = codiconSprite
  document.body.prepend(container)
}

export function Codicon({
  className,
  name,
  size,
  spinning,
  style,
  focusable = false,
  'aria-hidden': ariaHidden = true,
  ...props
}: CodiconProps) {
  useIsomorphicLayoutEffect(() => {
    ensureCodiconSprite()
  }, [])

  const symbolName = SYMBOL_IDS.has(name) ? name : 'blank'
  const hidden = ariaHidden === true || ariaHidden === 'true'

  return (
    <svg
      aria-hidden={ariaHidden}
      className={cn('codicon', `codicon-${symbolName}`, spinning && 'codicon-modifier-spin', className)}
      focusable={focusable}
      role={hidden ? undefined : 'img'}
      style={{ height: size, width: size, ...style }}
      viewBox="0 0 16 16"
      {...props}
    >
      <use href={`#${symbolName}`} xlinkHref={`#${symbolName}`} />
    </svg>
  )
}
