import { useStore } from '@nanostores/react'
import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { hiddenPaneProps, usePaneGroup, usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $composerPopout, $composerPopoutGesturesEnabled } from '@/store/composer-popout'
import { isSecondaryWindow } from '@/store/windows'

import { $floatingComposerOwner } from './floating-state'
import { registerFloatingComposer } from './floating-target'
import { useComposerScope, useComposerSurfaceId } from './scope'
import { ComposerVisibleContext } from './visibility'

interface ComposerSurfaceProps {
  children: ReactNode
}

/** Keep each session's editor, drafts and queue alive, but expose only the
 * recipient while floating. Moving a stable portal host avoids editor remounts. */
export function FloatingComposerSurface({ children }: ComposerSurfaceProps) {
  const anchorRef = useRef<HTMLDivElement>(null)
  const id = useComposerSurfaceId()!
  const groupId = usePaneGroup()

  // Identity must exist before descendant layout effects publish measurements.
  const [host] = useState(() => {
    const element = document.createElement('div')
    element.dataset.composerOwner = id
    element.dataset.treeGroup = groupId

    return element
  })

  const { target } = useComposerScope()
  const paneVisible = usePaneVisible()
  const enabled = useStore($composerPopoutGesturesEnabled)
  const floating = useStoreSelector($composerPopout, state => state.poppedOut) && enabled && !isSecondaryWindow()
  const selected = useStoreSelector($floatingComposerOwner, owner => owner?.id === id)
  const visible = paneVisible && (!floating || selected)

  useLayoutEffect(() => {
    const surface = anchorRef.current?.closest<HTMLElement>('[data-chat-surface]')

    if (!surface || !paneVisible) {
      return undefined
    }

    return registerFloatingComposer(id, { groupId, target })
    // Re-register when Fast Refresh replaces the module-local recipient registry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupId, id, paneVisible, registerFloatingComposer, target])

  useLayoutEffect(() => {
    host.dataset.composerOwner = id
    host.dataset.treeGroup = groupId

    if (floating) {
      host.dataset.floatingComposerTarget = target
    } else {
      delete host.dataset.floatingComposerTarget
    }

    host.style.display = visible ? 'contents' : 'none'
    host.toggleAttribute('data-pane-hidden', !visible)
    host.inert = !visible

    const parent = floating ? document.body : anchorRef.current

    if (parent && host.parentElement !== parent) {
      parent.appendChild(host)
    }
  }, [floating, groupId, host, id, target, visible])

  useLayoutEffect(() => () => host.remove(), [host])

  return (
    <>
      <div className="contents" ref={anchorRef} {...hiddenPaneProps(!visible)} />
      {createPortal(<ComposerVisibleContext value={visible}>{children}</ComposerVisibleContext>, host)}
    </>
  )
}
