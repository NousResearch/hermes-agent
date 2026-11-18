// Wires the machine's UI preferences store (ui-preferences.ts) into Electron:
// every app window hears about a change, whoever made it.
import { BrowserWindow, ipcMain } from 'electron'

import { registerUiPreferencesIpc, type UiPreferencesSnapshot } from './ui-preferences'

/** `hermesHome` is main's resolved ROOT home (never a profile's), the one the
 *  Webapp's Python server resolves too. `onChange` lets main follow the
 *  preferences it acts on itself (the shell's titlebar band). */
export function registerMachineUiPreferences(
  hermesHome: string,
  onChange: (snapshot: UiPreferencesSnapshot) => void
): () => void {
  return registerUiPreferencesIpc({
    ipcMain,
    hermesHome,
    broadcast: snapshot => {
      onChange(snapshot)

      for (const window of BrowserWindow.getAllWindows()) {
        window.webContents.send('hermes:ui-preferences:changed', snapshot)
      }
    }
  })
}
