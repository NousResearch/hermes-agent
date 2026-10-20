import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'

import { writeClipboardText } from '@/components/ui/copy-button'
import { triggerHaptic } from '@/lib/haptics'
import { useTheme } from '@/themes/context'

import { registerAgentTerminalWriter } from './agent-terminal-stream'
import { makeTerminalReader, registerTerminalReader } from './buffer'
import { mirrorSelection, terminalClipboardIntent } from './clipboard'
import { terminalLinkHandler, terminalWebLinksAddon } from './links'
import { isMacPlatform, resolveSurfaceColor, terminalTheme } from './selection'
import { registerTerminalContextMenu } from './terminal-context-menu'
import { prepareTerminalFontFamily } from './terminal-font'
import { redrawAllTerminals, registerWebglRefresh } from './terminals'
import { useTerminalFontController } from './use-terminal-font'

// Read-only terminal for an agent background process: a write-only xterm (no PTY,
// no input) fed live by the backend output stream, keyed by process id. Shares
// the user terminal's look so the two read as one surface.
export function useAgentTerminal({ active, id, procId }: { active: boolean; id: string; procId: string }) {
  const { renderedMode, theme, themeName } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const webglRef = useRef<WebglAddon | null>(null)
  const fitRef = useRef<(() => void) | null>(null)
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
    let observer: ResizeObserver | null = null
    let mounted = false
    let mountWatchFrame = 0

    const cancelMountWatch = () => {
      if (mountWatchFrame) {
        window.cancelAnimationFrame(mountWatchFrame)
        mountWatchFrame = 0
      }
    }

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
      screenReaderMode: true,
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

    // Right-clicks resolve through the app context menu; the handle carries
    // the xterm selection the DOM resolver cannot see. paste stays null —
    // there is nothing to paste into. reload swallows the chord for the same
    // reason: no PTY, and focus being here must still stop the app-level
    // Ctrl/Cmd+R reload fallback.
    const contextMenuDisposable = registerTerminalContextMenu(host, {
      getSelection: () => term.getSelection(),
      paste: null,
      reload: () => {},
      selectAll: () => term.selectAll(),
      // No PTY input, and the mirror's tab stays deliberately closeable, so
      // the close-tab chord keeps its close meaning here.
      wordErase: null
    })

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

    fitRef.current = () => {
      if (host.clientWidth > 0 && host.clientHeight > 0) {
        try {
          fit.fit()
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
      mounted = true

      try {
        const webgl = new WebglAddon()
        webgl.onContextLoss(() => {
          webgl.dispose()
          webglRef.current = null

          // Same as the user terminal: repaint the buffered rows with the DOM
          // renderer so the viewport doesn't stay black after a context loss.
          try {
            fitRef.current?.()
            term.refresh(0, term.rows - 1)
          } catch {
            // Best-effort repaint; the next resize repaints anyway.
          }
        })
        term.loadAddon(webgl)
        webglRef.current = webgl
      } catch {
        // No WebGL — xterm falls back to the DOM renderer.
      }

      fitRef.current?.()
      observer = new ResizeObserver(() => fitRef.current?.())
      observer.observe(host)

      // Stream live output straight into the terminal (replays backlog on attach).
      unregister = registerAgentTerminalWriter(procId, chunk => term.write(chunk))
      unregisterReader = registerTerminalReader(id, makeTerminalReader(term))
    }

    // Join the shared-atlas refresh fan-out (see redrawAllTerminals in
    // terminals.ts): clearing this terminal's atlas mutates texture pages the
    // user terminals draw from, so they must rebuild their models too.
    const unregisterWebglRefresh = registerWebglRefresh(term, () => webglRef.current)

    void prepareTerminalFontFamily(
      () => latestFontFamilyRef.current,
      () => !disposed && host.isConnected
    ).then(fontFamily => {
      if (!fontFamily) {
        // Same host-connection race as the user terminal (#118004): the font
        // wait resolves null when host.isConnected went false at an await
        // boundary, and returning here used to strand the pane blank. Poll
        // frames until the host connects, then retry the wait+mount.
        const watchForHost = () => {
          if (disposed || mounted) {
            return
          }

          if (host.isConnected) {
            void prepareTerminalFontFamily(
              () => latestFontFamilyRef.current,
              () => !disposed && host.isConnected
            ).then(next => {
              if (next && !disposed && !mounted && host.isConnected) {
                term.options.fontFamily = next
                mount()
              }
            })

            return
          }

          mountWatchFrame = window.requestAnimationFrame(watchForHost)
        }

        mountWatchFrame = window.requestAnimationFrame(watchForHost)

        return
      }

      term.options.fontFamily = fontFamily
      mount()
    })

    return () => {
      disposed = true
      mountedRef.current = false
      cancelMountWatch()
      unregister()
      unregisterReader()
      unregisterWebglRefresh()
      selectionDisposable.dispose()
      contextMenuDisposable()
      observer?.disconnect()
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
      // The atlas is shared across every terminal with the same render config,
      // so the clear must fan out to the siblings too (see redrawAllTerminals).
      redrawAllTerminals()
    })

    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renderedMode, themeName])

  // A visibility:hidden xterm doesn't paint — refit + redraw on re-activation.
  useEffect(() => {
    if (!active) {
      return
    }

    const frame = requestAnimationFrame(() => {
      const term = termRef.current

      fitRef.current?.()
      redrawAllTerminals()
      // Take focus on activation (parity with the user terminal) so the active
      // agent tab holds focus and ⌘W's isFocusWithin('[data-terminal]') routes
      // the close to this tab rather than to a preview.
      term?.focus()
    })

    return () => cancelAnimationFrame(frame)
  }, [active])

  return { hostRef }
}
