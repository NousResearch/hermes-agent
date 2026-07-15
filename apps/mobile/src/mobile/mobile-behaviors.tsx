import { App } from '@capacitor/app'
import { Keyboard } from '@capacitor/keyboard'
import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import {
  $fileBrowserOpen,
  $panesFlipped,
  $sidebarOpen,
  CHAT_SIDEBAR_PANE_ID,
  FILE_BROWSER_PANE_ID,
  toggleFileBrowserOpen,
  toggleSidebarOpen,
} from '@/store/layout'

/**
 * MobileBehaviors — the touch adaptations layered over the reused desktop UI.
 *
 *  1. Sidebar drawers: the titlebar burgers flip $sidebarOpen/$fileBrowserOpen,
 *     but collapsed panes only show via PANE_TOGGLE_REVEAL_EVENT (the mod+B
 *     path). Bridge that, normalize the pane flip, and dismiss on tap-outside.
 *  2. One Android back handler: keyboard → dismiss, else drawer → closed, else
 *     default (history back / exit).
 *
 * Overlay screens (Settings/Skills/Profiles) get their responsive master-detail
 * from upstream now (overlays/overlay-split-layout.tsx), so nothing here.
 */
function revealPane(id: string) {
  window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id } }))
}
function anyDrawerOpen() {
  return $sidebarOpen.get() || $fileBrowserOpen.get()
}
function closeOpenDrawer() {
  if ($sidebarOpen.get()) toggleSidebarOpen()
  else if ($fileBrowserOpen.get()) toggleFileBrowserOpen()
}

export function MobileBehaviors() {
  const location = useLocation()

  // Navigating (selecting a session) dismisses an open chat drawer.
  useEffect(() => {
    if (anyDrawerOpen()) closeOpenDrawer()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname])

  useEffect(() => {
    // Standard orientation: sessions LEFT, files RIGHT.
    $panesFlipped.set(false)

    // Track keyboard visibility so the Android back button can dismiss it first
    // (see the backButton handler). The WebView resizes above the keyboard on its
    // own — Chromium handles that natively on Android edge-to-edge, and the
    // SafeArea plugin is patched (patches/) so it doesn't double-count the IME
    // inset (which collapsed the viewport to ~241px).
    let keyboardOpen = false
    const kbShow = Keyboard.addListener('keyboardWillShow', () => {
      keyboardOpen = true
    })
    const kbHide = Keyboard.addListener('keyboardWillHide', () => {
      keyboardOpen = false
    })

    const offSidebar = $sidebarOpen.listen(() => revealPane(CHAT_SIDEBAR_PANE_ID))
    const offFiles = $fileBrowserOpen.listen(() => revealPane(FILE_BROWSER_PANE_ID))

    const syncDrawerAttr = () => {
      document.documentElement.toggleAttribute('data-drawer-open', anyDrawerOpen())
    }
    const offSidebarAttr = $sidebarOpen.subscribe(syncDrawerAttr)
    const offFilesAttr = $fileBrowserOpen.subscribe(syncDrawerAttr)

    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Element | null

      // Tap outside an open chat drawer → dismiss it.
      if (anyDrawerOpen() && !target?.closest('[data-pane-hover-reveal="open"]')) {
        e.stopPropagation()
        closeOpenDrawer()
      }
    }
    document.addEventListener('pointerdown', onPointerDown, true)

    let backHandle: { remove: () => void } | undefined
    void App.addListener('backButton', ({ canGoBack }) => {
      // Keyboard open → just dismiss it (and blur, so it doesn't auto-reopen
      // from the composer regaining focus). Don't navigate.
      if (keyboardOpen) {
        ;(document.activeElement as HTMLElement | null)?.blur()
        void Keyboard.hide()
        return
      }
      if (anyDrawerOpen()) {
        closeOpenDrawer()
        return
      }
      if (canGoBack) window.history.back()
      else void App.exitApp()
    }).then((h) => {
      backHandle = h
    })

    return () => {
      offSidebar()
      offFiles()
      offSidebarAttr()
      offFilesAttr()
      document.removeEventListener('pointerdown', onPointerDown, true)
      backHandle?.remove()
      void kbShow.then((h) => h.remove())
      void kbHide.then((h) => h.remove())
      document.documentElement.removeAttribute('data-drawer-open')
    }
  }, [])

  // No UI — this component exists only for the side effects wired up above.
  return null
}
