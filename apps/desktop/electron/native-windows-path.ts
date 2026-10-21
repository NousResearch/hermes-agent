// Native Windows spelling for a path a renderer handed us.
//
// A delivered `MEDIA:` value becomes `#media:%2FC%3A%5CUsers%5Cme%5Creport.zip`,
// i.e. the literal `/C:/Users/me/report.zip`. `path.resolve` reads that leading
// slash as a root WITHOUT a drive and lands on `C:\C:\Users\me\...`, so the
// preview and every `/api/fs/download` answering the card's Download button
// 404 with "File not found" while the file sits exactly where the transcript
// says.
//
// The rule lives here — ONE implementation — because two seams need it: the
// Electron main process resolving a local preview target, and the download
// bridge transcoding a delivered path for the gateway. The two copies drifted
// apart, so a fix in one left the other broken.
//
// Only a drive form is touched — `/home/me/x` and `/c/Users/me` (MSYS) keep
// their slashes — and POSIX hosts get the raw value back, since `/C:/...` is
// legitimate there. The platform is a parameter (not read from `process`)
// so both branches are reachable from any runner.

/**
 * Map the slash-prefixed drive form onto the native one for `platform`.
 *
 * @param filePath the raw path as delivered
 * @param platform the target platform ('win32' applies the mapping)
 * @param isWindowsHost whether the LOCAL host is Windows; a Windows host is the
 *   only one whose `path.resolve` mis-reads `/C:/...` as a drive-less root.
 *   Callers on a POSIX host keep the value untouched even for a 'win32'
 *   target, because their own `path.resolve` handles `/C:/...` correctly.
 */
export function nativeWindowsDrivePath(
  filePath: string,
  platform: NodeJS.Platform = process.platform,
  isWindowsHost: boolean = process.platform === 'win32'
): string {
  const raw = String(filePath ?? '').trim()

  if (!isWindowsHost || platform !== 'win32' || !raw || /^file:/i.test(raw)) {
    return raw
  }

  const asPosix = raw.replace(/\\/g, '/')

  return /^\/?[A-Za-z]:(\/|$)/.test(asPosix) ? asPosix.replace(/^\/+/, '').replace(/\//g, '\\') : raw
}
