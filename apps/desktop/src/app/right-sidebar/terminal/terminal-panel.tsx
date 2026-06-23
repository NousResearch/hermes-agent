import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  $focusedLeafId,
  addPane,
  closePane,
  focusNext,
  focusPrev,
  type PaneNode,
  resizePane,
  splitPane
} from '@/lib/terminal-store'

import { $rightSidebarTab } from '../store'

import { SplitLayout } from './split-layout'

interface TerminalPanelProps {
  active: boolean
  cwd: string
  onAddSelectionToChat: (text: string, label?: string) => void
  tree: PaneNode
}

export function TerminalPanel({ active, cwd, onAddSelectionToChat, tree }: TerminalPanelProps) {
  const { t } = useI18n()
  const focusedLeafId = useStore($focusedLeafId)

  // Keyboard shortcuts — only when terminal tab is active
  useEffect(() => {
    if (!active) {
      return
    }

    const handler = (event: KeyboardEvent) => {
      const mod = event.ctrlKey || event.metaKey

      if (!mod || !event.shiftKey) {
        return
      }

      // Only handle when terminal tab is visible
      if ($rightSidebarTab.get() !== 'terminal') {
        return
      }

      switch (event.key) {
        case 'T':

        case 't':
          event.preventDefault()
          addPane()

          break

        case '\\':
          event.preventDefault()
          splitPane('horizontal')

          break

        case '-':
          event.preventDefault()
          splitPane('vertical')

          break

        case 'W':

        case 'w':
          event.preventDefault()

          if ($focusedLeafId.get()) {
            closePane($focusedLeafId.get()!)
          }

          break

        case ']':
          event.preventDefault()
          focusNext()

          break

        case '[':
          event.preventDefault()
          focusPrev()

          break
      }
    }

    window.addEventListener('keydown', handler)

    return () => window.removeEventListener('keydown', handler)
  }, [active])

  const handleClosePane = (leafId: string) => closePane(leafId)
  const handleFocusPane = (leafId: string) => $focusedLeafId.set(leafId)

  const handleResize = (firstLeafId: string, secondLeafId: string, ratio: number) =>
    resizePane(firstLeafId, secondLeafId, ratio)

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex h-7 shrink-0 items-center gap-1 px-2 bg-[#002736]">
        <span className="text-[0.6rem] font-bold uppercase tracking-wider text-white/50 mr-1">
          {t.rightSidebar.terminal}
        </span>
        <Tip label="New terminal (Ctrl+Shift+T)">
          <Button
            aria-label="New terminal"
            className="size-5 rounded text-white/50 hover:bg-white/10 hover:text-white/80"
            onClick={() => addPane()}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Codicon name="add" size="0.75rem" />
          </Button>
        </Tip>
        <Tip label="Split right (Ctrl+Shift+\)">
          <Button
            aria-label="Split right"
            className="size-5 rounded text-white/50 hover:bg-white/10 hover:text-white/80"
            onClick={() => splitPane('horizontal')}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Codicon name="split-horizontal" size="0.75rem" />
          </Button>
        </Tip>
        <Tip label="Split down (Ctrl+Shift+-)">
          <Button
            aria-label="Split down"
            className="size-5 rounded text-white/50 hover:bg-white/10 hover:text-white/80"
            onClick={() => splitPane('vertical')}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Codicon name="split-vertical" size="0.75rem" />
          </Button>
        </Tip>
        <Tip label="Close pane (Ctrl+Shift+W)">
          <Button
            aria-label="Close pane"
            className="size-5 rounded text-white/50 hover:bg-white/10 hover:text-white/80"
            disabled={!focusedLeafId}
            onClick={() => focusedLeafId && closePane(focusedLeafId)}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Codicon name="close" size="0.75rem" />
          </Button>
        </Tip>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden" style={{ display: 'flex', flexDirection: 'column' }}>
        <SplitLayout
          cwd={cwd}
          focusedLeafId={focusedLeafId}
          node={tree}
          onAddSelectionToChat={onAddSelectionToChat}
          onClosePane={handleClosePane}
          onFocusPane={handleFocusPane}
          onResize={handleResize}
        />
      </div>
    </div>
  )
}
