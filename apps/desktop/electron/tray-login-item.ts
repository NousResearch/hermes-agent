interface LoginItemReader {
  getLoginItemSettings: () => { openAtLogin?: boolean }
}

interface LoginItemWriter {
  setLoginItemSettings: (settings: { args: string[]; openAtLogin: boolean; path: string }) => void
}

export function readLoginItemPreference(api: LoginItemReader): boolean | null {
  try {
    return Boolean(api.getLoginItemSettings().openAtLogin)
  } catch {
    return null
  }
}

export function applyLoginItemPreference(
  enabled: boolean,
  api: LoginItemWriter,
  executable = process.execPath
): boolean {
  try {
    api.setLoginItemSettings({ args: [], openAtLogin: enabled, path: executable })

    return true
  } catch {
    return false
  }
}
