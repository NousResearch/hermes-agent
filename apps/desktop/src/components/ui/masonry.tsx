import './masonry.css'

import { Children, type ComponentProps, useLayoutEffect, useRef } from 'react'

import { cn } from '@/lib/utils'

/** Native grid lanes when available; measured lanes for older Electron releases.
 * Children stay in source order and never move between React parents. */
export function Masonry({ children, className, ...props }: ComponentProps<'div'>) {
  const root = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const node = root.current

    if (!node || CSS.supports('display', 'grid-lanes')) {
      return
    }

    const items = Array.from(node.children) as HTMLElement[]
    let frame = 0
    let disposed = false

    const layout = () => {
      frame = 0

      if (disposed || !node.clientWidth) {
        return
      }
      const style = getComputedStyle(node)
      const tracks = style.gridTemplateColumns.split(' ').map(Number.parseFloat)

      if (!tracks.length || tracks.some(width => !Number.isFinite(width))) {
        return
      }
      const gap = Number.parseFloat(style.columnGap) || 0
      const width = tracks[0]
      const widthValue = `${width}px`

      // Write widths together before measuring wrapping. No React state updates,
      // window resize listeners, or per-item read/write alternation.
      for (const item of items) {
        if (item.style.width !== widthValue) {
          item.style.width = widthValue
        }
      }

      const heights = items.map(item => item.getBoundingClientRect().height)
      const lanes = tracks.map(() => 0)

      const placements = heights.map(height => {
        const lane = lanes.indexOf(Math.min(...lanes))
        const top = lanes[lane]
        lanes[lane] += height + gap

        return { top, left: lane * (width + gap) }
      })

      node.dataset.measured = ''
      items.forEach((item, index) => {
        item.style.insetInlineStart = `${placements[index].left}px`
        item.style.insetBlockStart = `${placements[index].top}px`
      })
      const height = `${Math.max(0, ...lanes) - (items.length ? gap : 0)}px`

      if (node.style.height !== height) {
        node.style.height = height
      }
    }

    const schedule = () => {
      if (!frame) {
        frame = requestAnimationFrame(layout)
      }
    }

    const observer = new ResizeObserver(schedule)
    observer.observe(node)
    items.forEach(item => observer.observe(item))
    layout()

    return () => {
      disposed = true
      observer.disconnect()
      cancelAnimationFrame(frame)
      delete node.dataset.measured
      node.style.removeProperty('height')
      items.forEach(item => {
        item.style.removeProperty('width')
        item.style.removeProperty('inset-inline-start')
        item.style.removeProperty('inset-block-start')
      })
    }
  }, [children])

  return (
    <div {...props} className={cn('masonry', className)} data-slot="masonry" ref={root}>
      {Children.map(children, child => child && <div data-slot="masonry-item">{child}</div>)}
    </div>
  )
}
