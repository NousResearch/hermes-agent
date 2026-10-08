interface PickerStartPathOptions {
  defaultPath?: unknown
  fallbackToDownloads?: unknown
}

export function resolvePickerStartPath(
  options: PickerStartPathOptions,
  downloadsPath: () => string
): string | undefined {
  if (options.defaultPath) {
    return String(options.defaultPath)
  }

  if (options.fallbackToDownloads !== true) {
    return undefined
  }

  try {
    return downloadsPath() || undefined
  } catch {
    return undefined
  }
}
