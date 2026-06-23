import { useMemo, useRef } from 'react'

import type { PaneNode } from '@/lib/terminal-store'
import { findLeafIds } from '@/lib/terminal-store'

import { TerminalPane } from './terminal-pane'

interface SplitLayoutProps {
  node: PaneNode
  cwd: string
  focusedLeafId: string | null
  onClosePane: (leafId: string) => void
  onFocusPane: (leafId: string) => void
  onResize: (firstLeafId: string, secondLeafId: string, ratio: number) => void
  onAddSelectionToChat: (text: string, label?: string) => void
}

interface Bounds {
  height: number
  left: number
  top: number
  width: number
}

interface LeafLayout {
  bounds: Bounds
  id: string
}

interface DividerLayout {
  bounds: Bounds
  direction: 'horizontal' | 'vertical'
  firstLeafId: string
  position: number
  secondLeafId: string
}

function flattenLayout(node: PaneNode, bounds: Bounds, leaves: LeafLayout[], dividers: DividerLayout[]): void {
  if (node.type === 'leaf') {
    leaves.push({ bounds, id: node.id })

    return
  }

  const isHorizontal = node.direction === 'horizontal'

  const firstBounds: Bounds = {
    height: isHorizontal ? bounds.height : bounds.height * node.ratio,
    left: bounds.left,
    top: bounds.top,
    width: isHorizontal ? bounds.width * node.ratio : bounds.width
  }

  const secondBounds: Bounds = {
    height: isHorizontal ? bounds.height : bounds.height * (1 - node.ratio),
    left: isHorizontal ? bounds.left + firstBounds.width : bounds.left,
    top: isHorizontal ? bounds.top : bounds.top + firstBounds.height,
    width: isHorizontal ? bounds.width * (1 - node.ratio) : bounds.width
  }

  const firstLeafId = findLeafIds(node.first)[0]
  const secondLeafId = findLeafIds(node.second)[0]

  if (firstLeafId && secondLeafId) {
    dividers.push({
      bounds,
      direction: node.direction,
      firstLeafId,
      position: node.ratio,
      secondLeafId
    })
  }

  flattenLayout(node.first, firstBounds, leaves, dividers)
  flattenLayout(node.second, secondBounds, leaves, dividers)
}

const percent = (value: number) => `${value * 100}%`

export function SplitLayout({ node, cwd, onClosePane, onFocusPane, onResize, onAddSelectionToChat }: SplitLayoutProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const layout = useMemo(() => {
    const leaves: LeafLayout[] = []
    const dividers: DividerLayout[] = []

    flattenLayout(node, { height: 1, left: 0, top: 0, width: 1 }, leaves, dividers)

    return { dividers, leaves }
  }, [node])

  return (
    <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden" ref={containerRef}>
      {layout.leaves.map(({ bounds, id }) => (
        <div
          className="absolute flex min-h-0 min-w-0 flex-col overflow-hidden p-px"
          key={id}
          style={{
            height: percent(bounds.height),
            left: percent(bounds.left),
            top: percent(bounds.top),
            width: percent(bounds.width)
          }}
        >
          <TerminalPane
            cwd={cwd}
            leafId={id}
            onAddSelectionToChat={onAddSelectionToChat}
            onClose={() => onClosePane(id)}
            onFocus={() => onFocusPane(id)}
          />
        </div>
      ))}
      {layout.dividers.map(({ bounds, direction, firstLeafId, position, secondLeafId }) => {
        const horizontal = direction === 'horizontal'
        const dividerLeft = horizontal ? bounds.left + bounds.width * position : bounds.left
        const dividerTop = horizontal ? bounds.top : bounds.top + bounds.height * position

        return (
          <div
            className={
              horizontal
                ? 'absolute z-20 w-1 -translate-x-1/2 cursor-col-resize hover:bg-(--ui-accent)/30'
                : 'absolute z-20 h-1 -translate-y-1/2 cursor-row-resize hover:bg-(--ui-accent)/30'
            }
            key={`${firstLeafId}:${secondLeafId}`}
            onMouseDown={event => {
              event.preventDefault()
              const container = containerRef.current

              if (!container) {
                return
              }

              const onMouseMove = (moveEvent: MouseEvent) => {
                const rect = container.getBoundingClientRect()
                const splitLeft = rect.left + bounds.left * rect.width
                const splitTop = rect.top + bounds.top * rect.height
                const splitWidth = bounds.width * rect.width
                const splitHeight = bounds.height * rect.height
                const position = horizontal ? moveEvent.clientX - splitLeft : moveEvent.clientY - splitTop
                const size = horizontal ? splitWidth : splitHeight

                onResize(firstLeafId, secondLeafId, Math.max(0.1, Math.min(0.9, position / size)))
              }

              const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove)
                document.removeEventListener('mouseup', onMouseUp)
                document.body.style.cursor = ''
                document.body.style.userSelect = ''
              }

              document.addEventListener('mousemove', onMouseMove)
              document.addEventListener('mouseup', onMouseUp)
              document.body.style.cursor = horizontal ? 'col-resize' : 'row-resize'
              document.body.style.userSelect = 'none'
            }}
            style={{
              height: horizontal ? percent(bounds.height) : undefined,
              left: percent(dividerLeft),
              top: percent(dividerTop),
              width: horizontal ? undefined : percent(bounds.width)
            }}
          />
        )
      })}
    </div>
  )
}
