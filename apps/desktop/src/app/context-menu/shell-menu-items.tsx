import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'

import { toggleTargetZoneTabStrip } from '@/components/pane-shell/tree/store'
import { type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { type Translations, useI18n } from '@/i18n'
import { openCommandPalette } from '@/store/command-palette'
import { toggleProfileRailVisible } from '@/store/profile-rail-prefs'
import { toggleStatusbarVisible } from '@/store/statusbar-prefs'
import { requestActiveUpdate } from '@/store/updates'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'

import { navigateToWorkspacePage, NEW_CHAT_ROUTE, SETTINGS_ROUTE } from '../routes'

/** Bare right-click on app chrome: the window verbs (the old shell fallback). */
function shellSections(
  kit: MenuKit,
  navigate: ReturnType<typeof useNavigate>,
  t: Translations,
  primaryOnly: boolean
): ReactNode[][] {
  const newChat = renderActionItem(kit, {
    icon: 'add',
    key: 'shell-new-chat',
    label: t.commandCenter.nav.newChat.title,
    onSelect: () => navigateToWorkspacePage(navigate, NEW_CHAT_ROUTE)
  })

  const settings = renderActionItem(kit, {
    icon: 'settings-gear',
    key: 'shell-settings',
    label: t.commandCenter.settings,
    onSelect: () => navigateToWorkspacePage(navigate, SETTINGS_ROUTE)
  })

  const update = renderActionItem(kit, {
    icon: 'cloud-download',
    key: 'shell-update',
    label: t.commandCenter.updateHermes,
    onSelect: requestActiveUpdate
  })

  if (primaryOnly) {
    return [[newChat, settings, update]]
  }

  return [
    [
      newChat,
      canOpenNewWindow()
        ? renderActionItem(kit, {
            icon: 'multiple-windows',
            key: 'shell-new-window',
            label: t.keybinds.actions['session.newWindow'],
            onSelect: () => void openNewWindow()
          })
        : null,
      renderActionItem(kit, {
        icon: 'search',
        key: 'shell-palette',
        label: t.commandCenter.paletteTitle,
        onSelect: openCommandPalette
      })
    ].filter(Boolean),
    [
      renderActionItem(kit, {
        icon: 'layout-statusbar',
        key: 'shell-statusbar',
        label: t.keybinds.actions['view.toggleStatusbar'],
        onSelect: toggleStatusbarVisible
      }),
      renderActionItem(kit, {
        icon: 'organization',
        key: 'shell-profile-rail',
        label: t.keybinds.actions['view.toggleProfileRail'],
        onSelect: toggleProfileRailVisible
      }),
      // The pointer-only way back to a hidden tab strip: right-clicking the
      // shell reaches this menu from anywhere, including a zone that has no
      // chrome left to right-click.
      renderActionItem(kit, {
        icon: 'layout-menubar',
        key: 'shell-tabstrip',
        label: t.keybinds.actions['view.toggleTabStrip'],
        onSelect: () => void toggleTargetZoneTabStrip()
      }),
      settings
    ],
    [update]
  ]
}

/** Shared by bare app chrome and the pane body's fallback section. */
export function ShellMenuItems({ kit, primaryOnly = false }: { kit: MenuKit; primaryOnly?: boolean }) {
  const navigate = useNavigate()
  const { t } = useI18n()

  return shellSections(kit, navigate, t, primaryOnly).map((section, index) => (
    <div className="contents" key={index}>
      {index > 0 && <kit.Separator />}
      {section}
    </div>
  ))
}
