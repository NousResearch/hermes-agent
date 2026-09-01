// The clipboard-image fallback ladder for `hermes:saveClipboardImage`
// (#85782). When Electron's clipboard.readImage() reports an empty image,
// the handler walks these rungs in a fixed order, and the first rung that
// yields validated PNG bytes wins. Extracted from main.ts so the precedence
// (and the platform gates that keep macOS/Windows/WSL behavior unchanged)
// are provable without booting Electron.

interface FallbackRungs {
  /** WSL2/WSLg detection (bootstrap-platform.isWslEnvironment), precomputed. */
  isWsl: boolean
  /** process.platform at the call site. */
  platform: NodeJS.Platform
  /** WSL reader: pulls the image off the Windows host clipboard. */
  readWsl: () => Buffer | null
  /** Wayland reader: wl-paste + PNG-byte validation (wayland-clipboard-image). */
  readWayland: () => Buffer | null
}

/** WSL first, then Wayland; first validated PNG wins; null when all rungs
 *  come up empty. Non-Linux platforms skip the ladder entirely. */
export function readFallbackClipboardPng(rungs: FallbackRungs): Buffer | null {
  if (rungs.platform !== 'linux') {
    return null
  }

  if (rungs.isWsl) {
    const wsl = rungs.readWsl()

    if (wsl) {
      return wsl
    }
    // Fall through: WSLg sessions may still expose wl-paste.
  }

  return rungs.readWayland()
}
