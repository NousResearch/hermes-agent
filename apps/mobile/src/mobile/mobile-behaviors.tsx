import { App } from '@capacitor/app'
import { Keyboard } from '@capacitor/keyboard'
import { useEffect } from 'react'
import { createPortal } from 'react-dom'
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
 *  2. Overlay master-detail: Settings/Skills/Profiles are a desktop two-pane
 *     split; on a phone we show the category list full-screen, then a tapped
 *     category full-screen (data-detail on the split, styled in theme-fallback),
 *     with a back button to return to the list.
 *  3. One Android back handler: detail → list, else drawer → closed, else default.
 *
 * The back button visibility is pure CSS (`body:has([data-overlay-split][data-detail])`),
 * so it correctly hides when the overlay is closed via its own X.
 */
const MOBILE_QUERY = '(max-width: 47.5rem)'

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
function openDetailSplit(): Element | null {
  return document.querySelector('[data-overlay-split][data-detail]')
}
function closeDetail() {
  document
    .querySelectorAll('[data-overlay-split][data-detail]')
    .forEach((el) => el.removeAttribute('data-detail'))
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

    // Drill into a tapped settings/skills/profiles category, full-screen. This
    // runs on CLICK (not pointerdown) so the nav item's OWN onClick — which
    // selects the category — fires first. Setting data-detail on pointerdown hid
    // the sidebar before the click could land, so the selection never happened
    // and every category opened the same (default) page.
    const onNavClick = (e: MouseEvent) => {
      const navItem = (e.target as Element | null)?.closest('[data-overlay-nav-item]')
      if (navItem && window.matchMedia(MOBILE_QUERY).matches) {
        navItem.closest('[data-overlay-split]')?.setAttribute('data-detail', '')
      }
    }
    document.addEventListener('click', onNavClick)

    let backHandle: { remove: () => void } | undefined
    void App.addListener('backButton', ({ canGoBack }) => {
      // Keyboard open → just dismiss it (and blur, so it doesn't auto-reopen
      // from the composer regaining focus). Don't navigate.
      if (keyboardOpen) {
        ;(document.activeElement as HTMLElement | null)?.blur()
        void Keyboard.hide()
        return
      }
      if (openDetailSplit()) {
        closeDetail()
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
      document.removeEventListener('click', onNavClick)
      backHandle?.remove()
      void kbShow.then((h) => h.remove())
      void kbHide.then((h) => h.remove())
      document.documentElement.removeAttribute('data-drawer-open')
    }
  }, [])

  // Back affordance for the overlay master-detail. Always rendered; CSS shows it
  // only while a detail page is open (and hides it if the overlay is X-closed).
  return createPortal(
    <button
      type="button"
      className="mobile-overlay-back"
      aria-label="Back to settings menu"
      onClick={closeDetail}
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 18l-6-6 6-6" />
      </svg>
    </button>,
    document.body,
  )
}
