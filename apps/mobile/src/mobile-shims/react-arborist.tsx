import { forwardRef, useImperativeHandle, useMemo, useState } from 'react'

export interface NodeApi<T = any> {
  data: T
  id: string
  isOpen: boolean
  isSelected: boolean
  level: number
  toggle: () => void
}

export interface TreeApi<T = any> {
  get: (id: string) => NodeApi<T> | null
}

export interface NodeRendererProps<T = any> {
  dragHandle: (el: HTMLElement | null) => void
  node: NodeApi<T>
  style: React.CSSProperties
}

interface TreeProps<T = any> {
  children: (props: NodeRendererProps<T>) => React.ReactNode
  childrenAccessor?: (node: T) => T[] | null | undefined
  data?: T[]
  initialOpenState?: Record<string, boolean>
  onActivate?: (node: NodeApi<T>) => void
  onToggle?: (id: string) => void
  rowHeight?: number
}

function getId(node: any): string {
  return String(node?.id ?? node?.path ?? node?.name ?? '')
}

export const Tree = forwardRef<TreeApi<any>, TreeProps<any>>(function Tree(
  { children, childrenAccessor, data = [], initialOpenState = {}, onActivate, onToggle, rowHeight = 22 },
  ref
) {
  const [openState, setOpenState] = useState<Record<string, boolean>>(initialOpenState || {})

  const nodes = useMemo(() => {
    const out: NodeApi<any>[] = []
    const visit = (items: any[], level: number) => {
      for (const item of items || []) {
        const id = getId(item)
        const isOpen = Boolean(openState[id])
        const node: NodeApi<any> = {
          data: item,
          id,
          isOpen,
          isSelected: false,
          level,
          toggle: () => {
            setOpenState(prev => ({ ...prev, [id]: !prev[id] }))
            onToggle?.(id)
          },
        }
        out.push(node)
        const kids = childrenAccessor?.(item) || []
        if (isOpen && kids.length) visit(kids, level + 1)
      }
    }
    visit(data, 0)
    return out
  }, [childrenAccessor, data, onToggle, openState])

  useImperativeHandle(ref, () => ({
    get(id: string) {
      return nodes.find(node => node.id === id) || null
    },
  }), [nodes])

  return (
    <div>
      {nodes.map((node, index) => (
        <div key={`${node.id}:${index}`} onDoubleClick={() => onActivate?.(node)}>
          {children({ dragHandle: () => {}, node, style: { height: rowHeight, paddingLeft: node.level * 10 } })}
        </div>
      ))}
    </div>
  )
})
