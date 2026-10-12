import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { BrowserPlus, iconSize } from '@/lib/icons'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $draftTitles, draftTitleIn } from '@/store/composer'
import { $interfaceMode } from '@/store/interface-mode'
import { $selectedStoredSessionId } from '@/store/session'
import { $openWorkspaceSessionInNewTab } from '@/store/workspace-tab-action'

export function WorkspaceTabTitle({ children }: { children: ReactNode }) {
  const interfaceMode = useStore($interfaceMode)
  const selected = useStore($selectedStoredSessionId)
  const keepOpen = useStore($openWorkspaceSessionInNewTab)
  const draftTitle = useStoreSelector($draftTitles, titles => draftTitleIn(titles, selected))
  const { t } = useI18n()

  if (interfaceMode !== 'simple') {
    return <>{children}</>
  }

  return (
    <span
      className="flex min-w-0 items-center gap-1"
      data-empty-preview={!selected && !draftTitle ? '' : undefined}
      onDoubleClick={() => selected && keepOpen?.()}
    >
      <span className="truncate font-normal italic text-(--ui-text-secondary)">{children}</span>
      {selected && keepOpen && (
        <Tip label={t.simpleWorkspace.keepOpen}>
          <Button
            aria-label={t.simpleWorkspace.keepOpen}
            className="shrink-0 text-foreground"
            onClick={event => {
              event.stopPropagation()
              void keepOpen()
            }}
            onPointerDown={event => event.stopPropagation()}
            size="icon-xs"
            variant="ghost"
          >
            <BrowserPlus aria-hidden className={iconSize.md} />
          </Button>
        </Tip>
      )}
    </span>
  )
}
