import type { ComponentProps, CSSProperties } from 'react'
import { memo, useCallback, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { cn } from '@/lib/utils'

interface FadeTextProps extends Omit<ComponentProps<'span'>, 'children'> {
  children: React.ReactNode
  /**
   * Width of the fade region on the trailing edge. Accepts any CSS length.
   * Defaults to 3rem so long strings clearly trail off — short enough to
   * preserve readable content, long enough to feel like a deliberate fade
   * rather than a clipped ellipsis.
   */
  fadeWidth?: string
}

/**
 * Single-line text that fades out instead of truncating with an ellipsis.
 *
 * Uses an inline mask-image so the fade resolves against whatever the parent
 * background is — no need to know the surface color, no after-pseudo overlap.
 * The mask is only applied when the text is actually overflowing, so short
 * strings render as plain text without an unnecessary gradient on their tail.
 *
 * Layout reads (`el.scrollWidth`) are forced reflows. To avoid measuring
 * once per parent re-render — which during streaming happens on every token —
 * we only re-measure when the ResizeObserver fires (real size changes), not
 * on every `children` reference change. Wrapped in `memo` with a custom
 * comparator so scalar-string children skip re-render entirely when the text
 * is unchanged but the parent re-rendered.
 */
function FadeTextImpl({ children, className, fadeWidth = '3rem', style, ...rest }: FadeTextProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const [overflowing, setOverflowing] = useState(false)

  const measureOverflow = useCallback(() => {
    const el = ref.current

    if (!el) {
      return
    }

    setOverflowing(el.scrollWidth - el.clientWidth > 1)
  }, [])

  useResizeObserver(measureOverflow, ref)

  const maskStyle: CSSProperties = overflowing
    ? {
        maskImage: `linear-gradient(to right, black calc(100% - ${fadeWidth}), transparent)`,
        WebkitMaskImage: `linear-gradient(to right, black calc(100% - ${fadeWidth}), transparent)`,
        ...style
      }
    : (style ?? {})

  return (
    <span
      {...rest}
      className={cn('block min-w-0 max-w-full overflow-hidden whitespace-nowrap', className)}
      ref={ref}
      style={maskStyle}
    >
      {children}
    </span>
  )
}

function styleEqual(a: CSSProperties | undefined, b: CSSProperties | undefined) {
  if (a === b) {
    return true
  }

  if (!a || !b) {
    return false
  }

  const aKeys = Object.keys(a)

  if (aKeys.length !== Object.keys(b).length) {
    return false
  }

  for (const k of aKeys) {
    if ((a as Record<string, unknown>)[k] !== (b as Record<string, unknown>)[k]) {
      return false
    }
  }

  return true
}

export const FadeText = memo(FadeTextImpl, (prev, next) => {
  if (prev.className !== next.className) {
    return false
  }

  if (prev.fadeWidth !== next.fadeWidth) {
    return false
  }

  if (!styleEqual(prev.style, next.style)) {
    return false
  }

  // Cheap path: the common case is a scalar string/number child. Identity
  // comparison is correct for any other element type (a new JSX node should
  // force a re-render).
  return prev.children === next.children
})
