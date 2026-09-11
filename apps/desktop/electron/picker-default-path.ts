import fs from 'node:fs'

/**
 * Gate the native file dialog's `defaultPath` on local readability.
 *
 * The picker browses THIS machine, but `defaultPath` is usually the backend
 * session's cwd — and in remote mode the backend is another machine (often
 * running as a different user). Its `/root` exists locally only as a path
 * this process cannot read, and a dialog seeded with it fails wholesale:
 * GTK shows "Could not read the contents of root: Permission denied" and
 * Linux/Windows pickers fall back to a broken or empty view. Existing local
 * read paths ("local-first") already validate; the picker never did.
 *
 * Returns the path unchanged when a local process can read it, and `null`
 * when it cannot (absent, permission-denied, or any other stat failure) so
 * the caller drops the hint and the dialog opens at its native default.
 */
export function locallyReadable(defaultPath: null | string | undefined): null | string {
  if (!defaultPath) {
    return null
  }

  try {
    fs.accessSync(String(defaultPath), fs.constants.R_OK)

    return String(defaultPath)
  } catch {
    return null
  }
}
