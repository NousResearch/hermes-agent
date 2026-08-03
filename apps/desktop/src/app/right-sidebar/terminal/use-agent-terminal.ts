import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'

import { writeClipboardText } from '@/components/ui/copy-button'
import { markRightPanePerf } from '@/debug/right-pane-events'
import { triggerHaptic } from '@/lib/haptics'
import { useTheme } from '@/themes/context'

import { observeActiveTerminalResize } from './active-resize'
import { registerAgentTerminalWriter } from './agent-terminal-stream'
import { makeTerminalReader, registerTerminalReader } from './buffer'
import { mirrorSelection, terminalClipboardIntent } from './clipboard'
import { terminalLinkHandler, terminalWebLinksAddon } from './links'
import { isMacPlatform, resolveSurfaceColor, terminalTheme } from './selection'
import { prepareTerminalFontFamily } from './terminal-font'
import { useTerminalFontController } from './use-terminal-font'

// Read-only terminal for an agent background process: a write-only xterm (no PTY,
// no input) fed live by the backend output stream, keyed by process id. Shares
// the user terminal's look so the two read as one surface.
export function useAgentTerminal({ active, id, procId }: { active: boolean; id: string; procId: string }) {
  const { renderedMode, theme, themeName } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const webglRef = useRef<WebglAddon | null>(null)
  const fitRef = useRef<((visible: boolean) => void) | null>(null)
  const initialActiveFitRef = useRef(false)
  const { latestFontFamilyRef, mountedRef } = useTerminalFontController({ fitRef, termRef, webglRef })

  const surfaceTheme = () => {
    const ansi = renderedMode === 'dark' ? (theme.darkTerminal ?? theme.terminal) : theme.terminal
    const base = terminalTheme(renderedMode, ansi)
    // Fall back to the palette's own background, not white — a hardcoded
    // '#ffffff' flashes a white slab in dark mode whenever the probe can't read
    // the token (pre-paint mount). Same contract as the user terminal.
    const surface = resolveSurfaceColor(base.background ?? '#ffffff')

    return { ...base, background: surface, cursorAccent: surface }
  }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    let disposed = false

    let unregister = () => {}

    let unregisterReader = () => {}

    const term = new Terminal({
      allowProposedApi: true,
      allowTransparency: false,
      convertEol: true,
      cursorBlink: false,
      disableStdin: true,
      fontFamily: latestFontFamilyRef.current,
      fontSize: 11,
      fontWeight: 'normal',
      fontWeightBold: 'bold',
      letterSpacing: 0,
      lineHeight: 1.12,
      linkHandler: terminalLinkHandler,
      minimumContrastRatio: 4.5,
      scrollback: 1000,
      theme: surfaceTheme()
    })

    const fit = new FitAddon()
    term.loadAddon(fit)
    term.loadAddon(new Unicode11Addon())
    term.loadAddon(terminalWebLinksAddon())
    term.unicode.activeVersion = '11'

    // Read-only mirror, but the output is exactly what people want to copy.
    // No paste path: this terminal has no PTY to paste into.
    const selectionDisposable = term.onSelectionChange(() => mirrorSelection(host, term.getSelection()))

    term.attachCustomKeyEventHandler(event => {
      const intent = terminalClipboardIntent(event, {
        hasSelection: Boolean(term.getSelection()),
        isMac: isMacPlatform()
      })

      if (intent !== 'copy') {
        return true
      }

      event.preventDefault()
      void writeClipboardText(term.getSelection()).catch(() => {
        // Clipboard unavailable — leave the selection so the user can retry.
      })
      term.clearSelection()
      triggerHaptic('selection')

      return false
    })

    fitRef.current = visible => {
      if (host.clientWidth > 0 && host.clientHeight > 0) {
        try {
          fit.fit()
          markRightPanePerf(visible ? 'terminal-fit-active' : 'terminal-fit-hidden', id)
        } catch {
          // Mid-transition layout — the next observer tick refits.
        }
      }
    }

    const mount = () => {
      if (disposed || !host.isConnected) {
        return
      }

      term.open(host)
      termRef.current = term
      mountedRef.current = true

      try {
        const webgl = new WebglAddon()
        webgl.onContextLoss(() => {
          webgl.dispose()
          webglRef.current = null
        })
        term.loadAddon(webgl)
        webglRef.current = webgl
      } catch {
        // No WebGL — xterm falls back to the DOM renderer.
      }

      fitRef.current?.(active)
      initialActiveFitRef.current = active

      // Stream live output straight into the terminal (replays backlog on attach).
      unregister = registerAgentTerminalWriter(procId, chunk => term.write(chunk))
      unregisterReader = registerTerminalReader(id, makeTerminalReader(term))
    }

    void prepareTerminalFontFamily(
      () => latestFontFamilyRef.current,
      () => !disposed && host.isConnected
    ).then(fontFamily => {
      if (!fontFamily) {
        return
      }

      term.options.fontFamily = fontFamily
      mount()
    })

    return () => {
      disposed = true
      mountedRef.current = false
      unregister()
      unregisterReader()
      selectionDisposable.dispose()
      fitRef.current = null
      term.dispose()
      termRef.current = null
      webglRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const term = termRef.current

    if (!term) {
      return
    }

    const raf = requestAnimationFrame(() => {
      term.options.theme = surfaceTheme()
      webglRef.current?.clearTextureAtlas()
    })

    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renderedMode, themeName])

  // Keep inactive agent terminals mounted for their backlog, but do not observe
  // or fit them until they become the visible tab.
  // eslint-disable-next-line no-restricted-syntax -- lifecycle flag prevents a duplicate first-mount fit
  useEffect(() => {
    if (!active) {
      initialActiveFitRef.current = false

      return
    }

    const host = hostRef.current

    if (!host) {
      return
    }

    const fitOnActivate = !initialActiveFitRef.current
    initialActiveFitRef.current = false

    return observeActiveTerminalResize(host, {
      fitOnActivate,
      onFit: () => fitRef.current?.(true),
      onActivate: () => {
        const term = termRef.current

        webglRef.current?.clearTextureAtlas()
        term?.refresh(0, term.rows - 1)
        // Take focus on activation (parity with the user terminal) so the active
        // agent tab holds focus and ⌘W's isFocusWithin('[data-terminal]') routes
        // the close to this tab rather than to a preview.
        term?.focus()
      }
    })
  }, [active])

  return { hostRef }
}
