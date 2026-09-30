import type { CSSProperties } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { Codicon } from '@/components/ui/codicon'
import { PaneStripGlyph } from '@/components/ui/pane-tab'
import { useI18n } from '@/i18n'

import { CanvasPopoutHost, useCanvasWindowTitle } from './canvas-tile'

/**
 * Dedicated shell for `?win=canvas`: one canvas provider's pane, full-window,
 * no session sidebar or layout tree. Same pane the docked tile renders — just
 * in its own OS window. Closing the window (or "Pop in") seats the tile back
 * in the tree that popped it.
 */
export function CanvasPopoutShell() {
  const { t } = useI18n()
  const title = useCanvasWindowTitle()

  return (
    <div
      className="flex h-screen min-h-0 w-screen flex-col bg-(--ui-bg-chrome) text-(--ui-text-primary)"
      data-contrib-shell=""
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <div
        className="relative shrink-0 border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome)"
        style={{ height: TITLEBAR_HEIGHT }}
      >
        {/* Same traffic-light / native-overlay carve-out as the main titlebar:
            a full-bar drag region would eat the window buttons. */}
        <div className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]" />
        <div className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,24px)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+2.25rem)] flex items-center justify-center [-webkit-app-region:drag]">
          <span className="truncate text-[0.8125rem] text-(--ui-text-secondary)">{title}</span>
        </div>
        <div
          className="absolute inset-y-0 flex items-center [-webkit-app-region:no-drag]"
          style={{ right: 'var(--titlebar-tools-right, 0.75rem)' }}
        >
          <PaneStripGlyph
            icon={<Codicon name="screen-normal" size="0.8125rem" />}
            label={t.preview.popIn}
            onSelect={() => window.close()}
          />
        </div>
      </div>
      <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
        <CanvasPopoutHost />
      </div>
    </div>
  )
}
