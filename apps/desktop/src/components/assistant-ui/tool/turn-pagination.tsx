'use client'

import {
  createContext,
  type FC,
  type PropsWithChildren,
  useCallback,
  useContext,
  useLayoutEffect,
  useRef,
  useState
} from 'react'

export const TOOL_TURN_PAGE_SIZE = 20

export interface TurnToolRef {
  key: string
  messageId: string
  partIndex: number
}

interface ToolTurnPaginationState {
  expanded: boolean
  oldestVisibleKey: string | null
  turnKey: string
}

interface ToolTurnPaginationValue {
  hiddenCount: number
  isVisible: (key: string) => boolean
  pagerKey: string | null
  revealEarlier: () => void
}

const ToolTurnPaginationContext = createContext<ToolTurnPaginationValue | null>(null)

export function toolPartPageKey(messageId: string, partIndex: number, toolCallId?: string): string {
  return `${messageId}:${toolCallId || `part-${partIndex}`}`
}

export function firstVisibleToolIndex(
  tools: readonly TurnToolRef[],
  oldestVisibleKey: string | null,
  expanded: boolean
): number {
  if (expanded && oldestVisibleKey) {
    const stableIndex = tools.findIndex(tool => tool.key === oldestVisibleKey)

    if (stableIndex >= 0) {
      return stableIndex
    }
  }

  return Math.max(0, tools.length - TOOL_TURN_PAGE_SIZE)
}

export const ToolTurnPaginationProvider: FC<PropsWithChildren<{ tools: readonly TurnToolRef[]; turnKey: string }>> = ({
  children,
  tools,
  turnKey
}) => {
  const [state, setState] = useState<ToolTurnPaginationState>({
    expanded: false,
    oldestVisibleKey: null,
    turnKey
  })

  const rootRef = useRef<HTMLDivElement | null>(null)
  const focusAfterRevealRef = useRef<string | null>(null)

  if (state.turnKey !== turnKey) {
    focusAfterRevealRef.current = null
    setState({ expanded: false, oldestVisibleKey: null, turnKey })
  }

  const currentState = state.turnKey === turnKey ? state : { expanded: false, oldestVisibleKey: null, turnKey }
  const firstVisible = firstVisibleToolIndex(tools, currentState.oldestVisibleKey, currentState.expanded)
  const hiddenCount = firstVisible
  const visibleKeys = new Set(tools.slice(firstVisible).map(tool => tool.key))
  const pagerKey = hiddenCount > 0 ? (tools[firstVisible]?.key ?? null) : null

  const revealEarlier = useCallback(() => {
    setState(current => {
      const normalized = current.turnKey === turnKey ? current : { expanded: false, oldestVisibleKey: null, turnKey }
      const visibleStart = firstVisibleToolIndex(tools, normalized.oldestVisibleKey, normalized.expanded)
      const nextStart = Math.max(0, visibleStart - TOOL_TURN_PAGE_SIZE)
      const nextOldest = tools[nextStart]?.key ?? null

      focusAfterRevealRef.current = nextStart === 0 ? nextOldest : null

      return { expanded: true, oldestVisibleKey: nextOldest, turnKey }
    })
  }, [tools, turnKey])

  useLayoutEffect(() => {
    const focusKey = focusAfterRevealRef.current

    if (!focusKey || hiddenCount > 0) {
      return
    }

    focusAfterRevealRef.current = null

    const target = Array.from(rootRef.current?.querySelectorAll<HTMLElement>('[data-tool-page-key]') ?? []).find(
      element => element.dataset.toolPageKey === focusKey
    )

    const frame = requestAnimationFrame(() => target?.focus({ preventScroll: true }))

    return () => cancelAnimationFrame(frame)
  }, [hiddenCount])

  const value: ToolTurnPaginationValue = {
    hiddenCount,
    isVisible: key => visibleKeys.has(key),
    pagerKey,
    revealEarlier
  }

  return (
    <ToolTurnPaginationContext.Provider value={value}>
      <div className="contents" ref={rootRef}>
        {children}
      </div>
    </ToolTurnPaginationContext.Provider>
  )
}

export function useToolTurnPagination(): ToolTurnPaginationValue | null {
  return useContext(ToolTurnPaginationContext)
}
