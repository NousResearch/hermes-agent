import { useStore } from '@nanostores/react'
import type * as React from 'react'

import { type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $projectTree, pickProjectFolder } from '@/store/projects'

import { projectTreeCwd } from './projects'

// Items for the ungrouped sidebar "+": plain click keeps starting a detached
// draft (null is load-bearing in workspace-session-target.ts — Home's "+" and
// the resolveNewSessionCwd fallthrough depend on it), while this right-click
// menu routes a chosen folder into the same onNewSessionInWorkspace(path) the
// grouped view's per-project "+" buttons use. Only projects with a real cwd
// are listed, so the path-less Home bucket can never sneak a null through
// onPick — onPick receives a real folder path or nothing at all.
export function useNewSessionWorkspaceMenuItems(onPick: (path: string) => void) {
  const { t } = useI18n()
  const s = t.sidebar
  const p = s.projects
  const tree = useStore($projectTree)

  const pickFolder = async () => {
    try {
      const dir = (await pickProjectFolder())?.trim()

      if (dir) {
        onPick(dir)
      }
    } catch (err) {
      notifyError(err, p.createFailed)
    }
  }

  return (kit: MenuKit): React.ReactNode => {
    const entries = tree.flatMap(project => {
      const cwd = projectTreeCwd(project)

      return cwd ? [{ cwd, id: project.id, label: project.label }] : []
    })

    return (
      <>
        {entries.map(entry =>
          renderActionItem(kit, {
            icon: 'folder',
            key: entry.id,
            label: s.newSessionIn(entry.label),
            onSelect: () => onPick(entry.cwd)
          })
        )}
        {entries.length > 0 && <kit.Separator />}
        {renderActionItem(kit, {
          icon: 'folder-opened',
          key: 'choose-folder',
          // ponytail: reuses the project dialog's picker label instead of a
          // new i18n key — every locale already carries it, and beside
          // "New session in X" rows "Add folder" unambiguously picks the
          // folder this session starts in. Split it out if the menu grows.
          label: p.addFolder,
          onSelect: () => void pickFolder()
        })}
      </>
    )
  }
}
