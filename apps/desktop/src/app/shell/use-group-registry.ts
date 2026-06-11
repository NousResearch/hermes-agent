import { useCallback, useMemo, useState } from 'react'

type Side = 'left' | 'right'
type Groups<T> = Record<Side, Record<string, readonly T[]>>

export type GroupSetter<T> = (id: string, items: readonly T[], side?: Side) => void

interface GroupRegistry<T> {
  flat: { left: T[]; right: T[] }
  set: GroupSetter<T>
}

export function useGroupRegistry<T>(): GroupRegistry<T> {
  const [groups, setGroups] = useState<Groups<T>>({ left: {}, right: {} })

  const set = useCallback<GroupSetter<T>>((id, items, side = 'right') => {
    setGroups(current => {
      const next = { ...current, [side]: { ...current[side] } }

      if (items.length === 0) {
        delete next[side][id]
      } else {
        next[side][id] = items
      }

      return next
    })
  }, [])

  const flat = useMemo(
    () => ({
      left: Object.values(groups.left).flat(),
      right: Object.values(groups.right).flat()
    }),
    [groups]
  )

  return { flat, set }
}
