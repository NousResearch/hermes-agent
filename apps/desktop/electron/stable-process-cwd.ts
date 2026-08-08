export function setStableProcessCwd(target: string, chdir: (path: string) => void = process.chdir) {
  try {
    chdir(target)

    return { changed: true, error: null }
  } catch (error) {
    return { changed: false, error: error instanceof Error ? error.message : String(error) }
  }
}
