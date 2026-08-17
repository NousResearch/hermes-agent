/**
 * windows-child-options.ts
 *
 * Shared helper for opting Windows child processes (spawn/execFileSync) into
 * a hidden console. Windows spawns a visible console window per child by
 * default; every desktop-launched helper process (git, curl, taskkill, the
 * backend itself, the bootstrap PowerShell runner, ...) needs `windowsHide:
 * true` so the user doesn't see consoles flashing on screen.
 *
 * Extracted into its own dependency-free module (no electron import) so it
 * can be unit-tested directly for both platforms without reading source
 * text, and so main.ts and bootstrap-runner.ts share exactly one
 * implementation instead of each defining their own copy.
 */

import type { ExecFileSyncOptionsWithStringEncoding } from 'node:child_process'

/**
 * Merge `windowsHide: true` into `options` when running on Windows, unless
 * the caller already specified a `windowsHide` value (which is preserved
 * as-is, including an explicit `false` for cases that intentionally want a
 * visible/interactive console). No-op on non-Windows platforms.
 *
 * @param options - spawn/execFileSync options to (possibly) augment.
 * @param isWindows - defaults to the real platform check; injectable for
 *   tests so both branches can be exercised without mocking process.platform.
 */
export function hiddenWindowsChildOptions(
  options: any = {},
  isWindows: boolean = process.platform === 'win32'
): ExecFileSyncOptionsWithStringEncoding {
  if (!isWindows || Object.prototype.hasOwnProperty.call(options, 'windowsHide')) {
    return options as any
  }

  return { ...options, windowsHide: true } as any
}
