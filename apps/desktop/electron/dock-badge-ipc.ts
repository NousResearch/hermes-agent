// Dock badge IPC: the main window reports how many sessions are waiting on the owner,
// and main paints it on the app icon. 0 (or anything that is not a positive count) clears it.

import { app, type IpcMain } from 'electron'

export const DOCK_BADGE_CHANNEL = 'hermes:dock-badge:set'

/** A renderer value as a badge count: whole, non-negative, bounded. */
export function dockBadgeCount(value: unknown): number {
  const count = Math.floor(Number(value))

  return Number.isFinite(count) && count > 0 ? Math.min(count, 9999) : 0
}

export function registerDockBadgeIpc(
  ipc: Pick<IpcMain, 'handle'>,
  setBadgeCount: (count: number) => boolean = count => app.setBadgeCount(count)
): void {
  ipc.handle(DOCK_BADGE_CHANNEL, (_event, value: unknown): boolean => setBadgeCount(dockBadgeCount(value)))
}
