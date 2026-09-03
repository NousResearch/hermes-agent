/**
 * Apply the Windows environment guards needed by MSYS-family shells.
 *
 * Git for Windows honors MSYS_NO_PATHCONV, while MSYS2 and Cygwin honor
 * MSYS2_ARG_CONV_EXCL. Setting both protects native Windows commands launched
 * from any supported bash without changing POSIX shells or explicit overrides.
 */
export function applyWindowsMsysBashEnvDefaults(
  env: Record<string, string | undefined>,
  isWindows: boolean
): void {
  if (!isWindows) {
    return
  }

  env.MSYS_NO_PATHCONV ??= '1'
  env.MSYS2_ARG_CONV_EXCL ??= '*'
}
