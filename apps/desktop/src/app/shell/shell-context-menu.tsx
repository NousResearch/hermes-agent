import type * as React from 'react'
import { useNavigate } from 'react-router'

import { hasTextSelection } from '@/components/assistant-ui/thread/user-message'
import { ActionsContextMenu, type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { useI18n } from '@/i18n'
import { isEditableTarget } from '@/lib/keybinds/combo'
import { openCommandPalette } from '@/store/command-palette'
import { toggleStatusbarVisible } from '@/store/statusbar-prefs'
import { requestActiveUpdate } from '@/store/updates'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'

import { navigateToWorkspacePage, NEW_CHAT_ROUTE, SETTINGS_ROUTE } from '../routes'

/**
 * The app's fallback context menu: right-clicking chrome that owns no menu of
 * its own (the titlebar gutter, a pane's empty body, the sidebar background)
 * gets the window-level verbs instead of nothing.
 *
 * It wraps the whole shell, so the guard below is what keeps it a FALLBACK: a
 * right-click that lands inside a surface with its own context menu, on an
 * editable, or on a live selection is stopped before Radix's trigger sees it,
 * leaving that surface's menu — or Electron's native edit menu — in charge.
 * `stopPropagation` (never `preventDefault`) is the mechanism, because
 * preventing the default is what would swallow the native menu.
 */
export function ShellContextMenu({ children }: { children: React.ReactNode }) {
  const { t } = useI18n()
  const navigate = useNavigate()

  const items = (kit: MenuKit) => (
    <>
      {renderActionItem(kit, {
        icon: 'add',
        label: t.commandCenter.nav.newChat.title,
        onSelect: () => navigateToWorkspacePage(navigate, NEW_CHAT_ROUTE)
      })}
      {canOpenNewWindow() &&
        renderActionItem(kit, {
          icon: 'multiple-windows',
          label: t.keybinds.actions['session.newWindow'],
          onSelect: () => void openNewWindow()
        })}
      {renderActionItem(kit, {
        icon: 'search',
        label: t.commandCenter.paletteTitle,
        onSelect: openCommandPalette
      })}
      <kit.Separator />
      {renderActionItem(kit, {
        icon: 'layout-statusbar',
        label: t.keybinds.actions['view.toggleStatusbar'],
        onSelect: toggleStatusbarVisible
      })}
      {renderActionItem(kit, {
        icon: 'settings-gear',
        label: t.commandCenter.settings,
        onSelect: () => navigateToWorkspacePage(navigate, SETTINGS_ROUTE)
      })}
      <kit.Separator />
      {renderActionItem(kit, {
        icon: 'cloud-download',
        label: t.commandCenter.updateHermes,
        onSelect: requestActiveUpdate
      })}
    </>
  )

  return (
    <ActionsContextMenu contentClassName="w-44" items={items}>
      <div className="contents" data-shell-context-menu="">
        {/* The guard sits on an INNER element on purpose: Radix's trigger props
            are merged onto the same node as ours (asChild), so a handler there
            can't stop the trigger's — but a child's stopPropagation never
            reaches it. */}
        <div className="contents" onContextMenu={guard}>
          {children}
        </div>
      </div>
    </ActionsContextMenu>
  )
}

/** Right-clicks that already have an owner keep it: a surface with its own
 *  context menu, an editable, a live selection (Electron's native edit menu),
 *  or an image/media element (Electron's native image menu — Copy Image,
 *  Save Image As...). Never `preventDefault` — that is what would swallow the
 *  native menu. */
function guard(event: React.MouseEvent<HTMLDivElement>) {
  const target = event.target as HTMLElement | null
  const owner = target?.closest('[data-slot="context-menu-trigger"]')

  if (
    (owner && !owner.hasAttribute('data-shell-context-menu')) ||
    target?.closest('img, picture, video, canvas') ||
    isEditableTarget(target) ||
    hasTextSelection()
  ) {
    event.stopPropagation()
  }
}
