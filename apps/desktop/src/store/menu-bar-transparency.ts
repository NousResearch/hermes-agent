import { atom } from 'nanostores'

import { broadcastDesktopStateChange, onDesktopStateSync } from '@/lib/desktop-state-sync'
import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.menuBarTransparency.v1'

export const DEFAULT_MENU_BAR_TRANSPARENCY = 20

export const clampMenuBarTransparency = (value: number): number => Math.min(100, Math.max(0, Math.round(value)))

export function menuBarTransparencyFromStorage(stored: null | string): number {
  if (stored === null || stored.trim() === '') {
    return DEFAULT_MENU_BAR_TRANSPARENCY
  }

  const value = Number(stored)

  return Number.isFinite(value) ? clampMenuBarTransparency(value) : DEFAULT_MENU_BAR_TRANSPARENCY
}

const read = (): number => menuBarTransparencyFromStorage(storedString(KEY))

export function getMenuBarSurfaceAlphas(transparency: number) {
  const value = clampMenuBarTransparency(transparency)

  return {
    background: Number((0.98 - value * 0.006).toFixed(3)),
    gradientBottom: Number((0.84 - value * 0.006).toFixed(3)),
    gradientTop: Number((0.65 - value * 0.005).toFixed(3)),
    panel: Number((0.96 - value * 0.005).toFixed(3))
  }
}

export const $menuBarTransparency = atom<number>(typeof window === 'undefined' ? DEFAULT_MENU_BAR_TRANSPARENCY : read())

export function setMenuBarTransparency(transparency: number): void {
  const next = clampMenuBarTransparency(transparency)
  $menuBarTransparency.set(next)
  broadcastDesktopStateChange('menu-bar-transparency', { value: next })
}

if (typeof window !== 'undefined') {
  $menuBarTransparency.subscribe(transparency => {
    persistString(KEY, String(transparency))
  })

  onDesktopStateSync(message => {
    if (message.type === 'changed' && message.domain === 'menu-bar-transparency' && typeof message.value === 'number') {
      $menuBarTransparency.set(clampMenuBarTransparency(message.value))
    }
  })
}
