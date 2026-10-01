interface PickerStartPathOptions {
  defaultPath?: unknown
  fallbackToDownloads?: unknown
}

export function resolvePickerStartPath(
  options: PickerStartPathOptions,
  downloadsPath: () => string
): string | undefined {
  if (typeof options.defaultPath === 'string' && options.defaultPath.trim().length > 0) {
    return options.defaultPath
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
