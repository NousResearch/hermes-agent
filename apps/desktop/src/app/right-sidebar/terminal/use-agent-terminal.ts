import { FitAddon } from '@xterm/addon-fit'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'

import { useTheme } from '@/themes/context'

import { resolveSurfaceColor, terminalTheme } from './selection'

// Read-only terminal driven by a string (an agent background process's output
// tail), not a PTY — no input, no shell. Shares the user terminal's look so the
// two read as one surface.
export function useAgentTerminal({ active, output }: { active: boolean; output: string }) {
  const { renderedMode, theme, themeName } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const webglRef = useRef<WebglAddon | null>(null)
  const fitRef = useRef<(() => void) | null>(null)
  const writtenRef = useRef('')

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
      convertEol: true,
      cursorBlink: false,
      disableStdin: true,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo, Consolas, monospace",
      fontSize: 11,
      lineHeight: 1.12,
      minimumContrastRatio: 4.5,
      scrollback: 5000,
      theme: surfaceTheme()
    })

    const fit = new FitAddon()
    term.loadAddon(fit)
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

    return () => {
      observer.disconnect()
      term.dispose()
      termRef.current = null
      webglRef.current = null
      writtenRef.current = ''
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Append the delta when the tail just grew; otherwise (the rolling window slid)
  // reset and rewrite. Avoids reflowing the whole buffer on every poll.
  useEffect(() => {
    const term = termRef.current

    if (!term) {
      return
    }

    if (output.startsWith(writtenRef.current)) {
      term.write(output.slice(writtenRef.current.length))
    } else {
      term.reset()
      term.write(output)
    }

    writtenRef.current = output
  }, [output])

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
    })

    return () => cancelAnimationFrame(frame)
  }, [active])

  return { hostRef }
}
