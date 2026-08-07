/**
 * System-tray helpers for the minimize-to-tray feature.
 *
 * Kept in a small module so the pure pieces (persistence + menu-template
 * construction + icon sizing) are unit-testable without booting Electron. The
 * tray *instance* itself (Tray construction + lifecycle) lives in main.ts
 * because it needs the live `mainWindow`/`tray` closures alongside the rest of
 * the window lifecycle.
 */

import { Menu, nativeImage, Tray } from 'electron'
import type { MenuItemConstructorOptions } from 'electron'

const MINIMIZE_TO_TRAY_CONFIG_PATH = (userData: string): string =>
  require('node:path').join(userData, 'minimize-to-tray.json')

// ── Persistence ────────────────────────────────────────────────────────────
// `userData` is injected so tests can point at a temp dir instead of the real
// app data path (which isn't available under the test runner).

export function readPersistedMinimizeToTray(userData: string): boolean {
  try {
    const fs = require('node:fs')
    const parsed = JSON.parse(fs.readFileSync(MINIMIZE_TO_TRAY_CONFIG_PATH(userData), 'utf8'))

    return parsed && typeof parsed.enabled === 'boolean' ? parsed.enabled : false
  } catch {
    // Missing / malformed → default off, like a fresh install.
    return false
  }
}

export function writePersistedMinimizeToTray(userData: string, enabled: boolean): void {
  try {
    const fs = require('node:fs')
    const path = require('node:path')
    const target = MINIMIZE_TO_TRAY_CONFIG_PATH(userData)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, JSON.stringify({ enabled }, null, 2), 'utf8')
  } catch {
    // Persistence is best-effort; never throw into the lifecycle path.
  }
}

// ── Menu template ──────────────────────────────────────────────────────────
// Pure: given the current window visibility + the callbacks, return the menu
// items. Extracted so the label-switching (Show vs Hide) and locale lookup are
// testable without a live Tray.
//
// The main process has no i18n machinery (menus there are hardcoded English,
// see e.g. the app menu). To make the tray menu follow the renderer's chosen
// display language, the renderer pushes its current locale to the main process
// (see setTrayMenuLocale bridge), and we translate these five short labels
// here. Kept intentionally small — only what the tray needs.

export type TrayLocale = 'en' | 'zh' | 'zh-hant' | 'ja' | 'ar'

const TRAY_MENU_STRINGS: Record<TrayLocale, {
  show: string
  hide: string
  newSession: string
  openSettings: string
  quit: string
}> = {
  en: { show: 'Show Window', hide: 'Hide Window', newSession: 'New Session', openSettings: 'Open Settings', quit: 'Quit' },
  zh: { show: '显示窗口', hide: '隐藏窗口', newSession: '新建会话', openSettings: '打开设置', quit: '退出' },
  'zh-hant': { show: '顯示視窗', hide: '隱藏視窗', newSession: '新建會話', openSettings: '開啟設定', quit: '退出' },
  ja: { show: 'ウィンドウを表示', hide: 'ウィンドウを隠す', newSession: '新規セッション', openSettings: '設定を開く', quit: '終了' },
  ar: { show: 'إظهار النافذة', hide: 'إخفاء النافذة', newSession: 'جلسة جديدة', openSettings: 'فتح الإعدادات', quit: 'إنهاء' }
}

export function trayMenuStringsFor(locale: string): TrayLocale {
  return (locale === 'zh' || locale === 'zh-hant' || locale === 'ja' || locale === 'ar') ? (locale as TrayLocale) : 'en'
}

export interface TrayMenuDeps {
  isWindowVisible: boolean
  locale: string
  onToggleVisibility: () => void
  onNewSession: () => void
  onOpenSettings: () => void
  onQuit: () => void
}

export function buildTrayMenuItems(deps: TrayMenuDeps): MenuItemConstructorOptions[] {
  const strings = TRAY_MENU_STRINGS[trayMenuStringsFor(deps.locale)]
  const visibilityLabel = deps.isWindowVisible ? strings.hide : strings.show

  return [
    {
      label: visibilityLabel,
      click: deps.onToggleVisibility
    },
    {
      label: strings.newSession,
      click: deps.onNewSession
    },
    {
      label: strings.openSettings,
      click: deps.onOpenSettings
    },
    { type: 'separator' },
    {
      label: strings.quit,
      click: deps.onQuit
    }
  ]
}

// ── Icon ─────────────────────────────────────────────────────────────────────
// Windows renders the tray at 16×16; resize down so it isn't blurry. Returns
// null when no icon can be resolved (caller skips tray creation).

export function buildTrayIcon(iconPath: string | null, isWindows: boolean): Electron.NativeImage | null {
  if (!iconPath) {
    return null
  }

  const base = nativeImage.createFromPath(iconPath)

  return isWindows ? base.resize({ width: 16, height: 16 }) : base
}

export { Menu, Tray }
