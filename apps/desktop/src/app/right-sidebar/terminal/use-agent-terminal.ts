import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'

import { useTheme } from '@/themes/context'

import { registerAgentTerminalWriter } from './agent-terminal-stream'
import { makeTerminalReader, registerTerminalReader } from './buffer'
import { resolveSurfaceColor, terminalTheme } from './selection'

// Read-only terminal for an agent background process: a write-only xterm (no PTY,
// no input) fed live by the backend output stream, keyed by process id. Shares
// the user terminal's look so the two read as one surface.
export function useAgentTerminal({ active, id, procId }: { active: boolean; id: string; procId: string }) {
  const { renderedMode, theme, themeName } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const webglRef = useRef<WebglAddon | null>(null)
  const fitRef = useRef<(() => void) | null>(null)

  const surfaceTheme = () => {
    const ansi = renderedMode === 'dark' ? (theme.darkTerminal ?? theme.terminal) : theme.terminal
    const surface = resolveSurfaceColor('#ffffff')

    return { ...terminalTheme(renderedMode, ansi), background: surface, cursorAccent: surface }
  }

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    const term = new Terminal({
      allowProposedApi: true,
      allowTransparency: false,
      convertEol: true,
      cursorBlink: false,
      disableStdin: true,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo, Consolas, monospace",
      fontSize: 11,
      fontWeight: 'normal',
      fontWeightBold: 'bold',
      letterSpacing: 0,
      lineHeight: 1.12,
      minimumContrastRatio: 4.5,
      scrollback: 1000,
      theme: surfaceTheme()
    })

    const fit = new FitAddon()
    term.loadAddon(fit)
    term.loadAddon(new Unicode11Addon())
    term.loadAddon(new WebLinksAddon())
    term.unicode.activeVersion = '11'
    term.open(host)
    termRef.current = term

    fitRef.current = () => {
      if (host.clientWidth > 0 && host.clientHeight > 0) {
        try {
          fit.fit()
        } catch {
          // Mid-transition layout — the next observer tick refits.
        }
      }
    }

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

    fitRef.current()
    const observer = new ResizeObserver(() => fitRef.current?.())
    observer.observe(host)

    // Stream live output straight into the terminal (replays backlog on attach).
    const unregister = registerAgentTerminalWriter(procId, chunk => term.write(chunk))
    const unregisterReader = registerTerminalReader(id, makeTerminalReader(term))

    return () => {
      unregister()
      unregisterReader()
      observer.disconnect()
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

  // A visibility:hidden xterm doesn't paint — refit + redraw on re-activation.
  useEffect(() => {
    if (!active) {
      return
    }

    const frame = requestAnimationFrame(() => {
      const term = termRef.current

      fitRef.current?.()
      webglRef.current?.clearTextureAtlas()
      term?.refresh(0, term.rows - 1)
      // Take focus on activation (parity with the user terminal) so the active
      // agent tab holds focus and ⌘W's isFocusWithin('[data-terminal]') routes
      // the close to this tab rather than to a preview.
      term?.focus()
    })

    return () => cancelAnimationFrame(frame)
  }, [active])

  return { hostRef }
}
