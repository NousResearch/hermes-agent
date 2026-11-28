import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'

import type { ComposerStatusItem } from '@/store/composer-status'

import { StatusItemRow } from './status-row'

interface TodoRowsProps {
  items: readonly ComposerStatusItem[]
  /** Render a retained, settled snapshot (same rows, no running state). */
  historical?: boolean
  onDismiss?: (id: string) => void
}

interface Frame {
  children: number
  depth: number
  done: number
  end: number
  /** The subtree still has work the user is waiting on. */
  open: boolean
  parent: number
  total: number
}

const settled = (item: ComposerStatusItem) => item.todoStatus === 'completed' || item.todoStatus === 'cancelled'
const live = (item: ComposerStatusItem) => item.todoStatus === 'pending' || item.todoStatus === 'in_progress'

/**
 * Todo rows with a disclosure per parent. `items` is the depth-ordered DFS the
 * composer feed already produces (`todoTree`), so every subtree is one
 * contiguous range: a parent folds to its own line, a settled phase folds by
 * default, and the phase that still holds the live item stays open. The user's
 * own toggles win over the default for the life of the panel.
 */
export function TodoRows({ historical = false, items, onDismiss }: TodoRowsProps) {
  const [overrides, setOverrides] = useState<Map<string, boolean>>(() => new Map())

  const frames = useMemo(() => {
    const list: Frame[] = items.map(item => ({
      children: 0,
      depth: item.depth ?? 0,
      done: 0,
      end: items.length,
      open: false,
      parent: -1,
      total: 0
    }))

    const stack: number[] = []

    for (let i = 0; i < list.length; i++) {
      while (stack.length > 0 && list[stack[stack.length - 1]].depth >= list[i].depth) {
        list[stack.pop()!].end = i
      }

      if (stack.length > 0) {
        list[i].parent = stack[stack.length - 1]
      }

      stack.push(i)
    }

    // Reverse accumulation: children always sit after their parent.
    for (let i = list.length - 1; i >= 0; i--) {
      const parent = list[i].parent

      if (parent >= 0) {
        list[parent].children += 1
        list[parent].total += 1 + list[i].total
        list[parent].done += (settled(items[i]) ? 1 : 0) + list[i].done
        list[parent].open = list[parent].open || live(items[i]) || list[i].open
      }
    }

    return list
  }, [items])

  const collapsed = (index: number) => {
    const frame = frames[index]

    if (frame.children === 0) {
      return false
    }

    const override = overrides.get(items[index].id)

    return override ?? !frame.open
  }

  const toggle = (id: string) =>
    setOverrides(current => {
      const next = new Map(current)
      const index = items.findIndex(item => item.id === id)

      next.set(id, index >= 0 ? !collapsed(index) : true)

      return next
    })

  const rows: ReactNode[] = []
  let hiddenUntil = -1

  for (let i = 0; i < items.length; i++) {
    if (i <= hiddenUntil) {
      continue
    }

    const frame = frames[i]
    const isCollapsed = collapsed(i)

    if (isCollapsed) {
      hiddenUntil = frame.end - 1
    }

    rows.push(
      <StatusItemRow
        collapsed={frame.children > 0 ? isCollapsed : undefined}
        historical={historical}
        item={items[i]}
        key={items[i].id}
        onDismiss={onDismiss}
        onToggle={frame.children > 0 ? () => toggle(items[i].id) : undefined}
        subtree={frame.children > 0 ? { done: frame.done, total: frame.total } : undefined}
      />
    )
  }

  return <>{rows}</>
}
