import { useCallback, useMemo } from 'react'

import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { displayPath } from '@/lib/display-path'

import type { StatusbarMenuItem } from '../statusbar-controls'

interface FocusedTarget {
  focusedRowConnectionId: string
  focusedRuntimeId: null | string
  primaryFocused: boolean
}

/**
 * The workspace menu's "Change working directory" item: re-homes the FOCUSED
 * chat, the recovery path for a chat that started in the wrong folder, Bot
 * Chats included (they never reach the sidebar's move-to-project). The picker
 * is remote-aware, so a remote gateway browses its own filesystem. A cold tile
 * has no runtime to re-home, and a tile on another gateway would be browsed on
 * the wrong machine, so the item is hidden for both. Disabled while a turn
 * runs: the backend refuses `session.cwd.set` then.
 */
export function useChangeWorkspaceItem(
  changeSessionCwd: (cwd: string, sessionId?: string) => Promise<void>,
  currentCwd: string,
  busy: boolean,
  { focusedRowConnectionId, focusedRuntimeId, primaryFocused }: FocusedTarget
): StatusbarMenuItem {
  const title = useI18n().t.rightSidebar.changeCwdTitle
  const targetId = primaryFocused ? undefined : focusedRuntimeId || null

  const changeWorkspace = useCallback(async () => {
    const [dir] = await selectDesktopPaths({ defaultPath: currentCwd, directories: true, multiple: false, title })

    if (dir) {
      await changeSessionCwd(dir, targetId ?? undefined)
    }
  }, [changeSessionCwd, currentCwd, targetId, title])

  return useMemo(
    () => ({
      disabled: busy,
      hidden: targetId === null || Boolean(focusedRowConnectionId),
      id: 'change-workspace-path',
      label: title,
      onSelect: () => void changeWorkspace(),
      title: displayPath(currentCwd)
    }),
    [busy, changeWorkspace, currentCwd, focusedRowConnectionId, targetId, title]
  )
}
