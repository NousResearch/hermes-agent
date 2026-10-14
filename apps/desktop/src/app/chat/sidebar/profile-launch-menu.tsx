import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import type { DesktopProfileRoute } from '@/global'
import { useI18n } from '@/i18n'
import { $defaultProfileRoute, setDefaultProfile } from '@/store/default-profile'
import { notify, notifyError } from '@/store/notifications'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'

interface ProfileLaunchMenuProps extends DesktopProfileRoute {
  label: string
}

export function ProfileLaunchMenuItems({ connectionId, label, profile }: ProfileLaunchMenuProps) {
  const { t } = useI18n()
  const p = t.profiles
  const defaultRoute = useStore($defaultProfileRoute)
  const isDefault = defaultRoute?.connectionId === connectionId && defaultRoute.profile === profile
  const route = { connectionId, profile }

  const makeDefault = async () => {
    try {
      await setDefaultProfile(route)
      notify({ kind: 'success', title: p.defaultSet(label), message: p.defaultDescription })
    } catch (error) {
      notifyError(error, p.failedSetDefault)
    }
  }

  return (
    <>
      {canOpenNewWindow() && (
        <ContextMenuItem onSelect={() => void openNewWindow(route)}>
          <Codicon name="link-external" size="0.875rem" />
          <span>{p.openInNewWindow}</span>
        </ContextMenuItem>
      )}
      <ContextMenuItem
        disabled={isDefault || !window.hermesDesktop?.profile?.setDefault}
        onSelect={() => void makeDefault()}
      >
        <Codicon name={isDefault ? 'check' : 'home'} size="0.875rem" />
        <span>{isDefault ? p.defaultProfile : p.setAsDefault}</span>
      </ContextMenuItem>
    </>
  )
}

export function ProfileLaunchContextMenu({ children, ...route }: ProfileLaunchMenuProps & { children: ReactNode }) {
  const { t } = useI18n()

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <span className="contents">{children}</span>
      </ContextMenuTrigger>
      <ContextMenuContent
        aria-label={t.profiles.actions}
        className="min-w-52"
        collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }}
        onCloseAutoFocus={event => event.preventDefault()}
      >
        <ProfileLaunchMenuItems {...route} />
      </ContextMenuContent>
    </ContextMenu>
  )
}

export function ProfileLaunchMenuSection(props: ProfileLaunchMenuProps) {
  return (
    <>
      <ProfileLaunchMenuItems {...props} />
      <ContextMenuSeparator />
    </>
  )
}
