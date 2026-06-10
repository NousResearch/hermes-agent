import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { useTheme } from '@/themes/context'

import { makeTerminalReader, setActiveTerminalReader } from './buffer'
import {
  isAddSelectionShortcut,
  resolveSurfaceColor,
  terminalSelectionAnchor,
  terminalSelectionLabel,
  terminalTheme
} from './selection'

type TerminalStatus = 'closed' | 'open' | 'starting'

const HERMES_PATHS_MIME = 'application/x-hermes-paths'

function readEscapeSequence(data: string, index: number) {
  if (data.charCodeAt(index) !== 0x1b || index + 1 >= data.length) {
    return null
  }

  const kind = data[index + 1]

  if (kind === '[') {
    for (let i = index + 2; i < data.length; i += 1) {
      const code = data.charCodeAt(i)

      if (code >= 0x40 && code <= 0x7e) {
        return data.slice(index, i + 1)
      }
    }
  }

  if (kind === ']') {
    for (let i = index + 2; i < data.length; i += 1) {
      if (data.charCodeAt(i) === 0x07) {
        return data.slice(index, i + 1)
      }

      if (data.charCodeAt(i) === 0x1b && data[i + 1] === '\\') {
        return data.slice(index, i + 2)
      }
    }
  }

  return data.slice(index, Math.min(index + 2, data.length))
}

function stripEscapeSequences(data: string) {
  let index = 0
  let text = ''

  while (index < data.length) {
    const sequence = readEscapeSequence(data, index)

    if (sequence) {
      index += sequence.length
    } else {
      text += data[index]
      index += 1
    }
  }

  return text
}

// Keep only the ANSI escape sequences from a chunk, dropping printable text. Lets
// us apply control codes (e.g. a clear-screen) while discarding boot spacers and
// zsh's reverse-video "%" partial-line marker.
function keepEscapeSequences(data: string) {
  let index = 0
  let out = ''

  while (index < data.length) {
    if (data.charCodeAt(index) === 0x1b) {
      const sequence = readEscapeSequence(data, index)

      if (sequence) {
        out += sequence
        index += sequence.length

        continue
      }
    }

    index += 1
  }

  return out
}

function stripInitialPromptGap(data: string) {
  let index = 0
  let prefix = ''

  while (index < data.length) {
    const sequence = readEscapeSequence(data, index)

    if (sequence) {
      prefix += sequence
      index += sequence.length
    } else if (data[index] === '\r' || data[index] === '\n') {
      index += 1
    } else {
      return prefix + data.slice(index)
    }
  }

  return prefix
}

interface UseTerminalSessionOptions {
  cwd: string
  onAddSelectionToChat: (text: string, label?: string) => void
}

// Bind the palette to the live skin surface so the terminal blends with the app
// (and the contrast clamp has a real background to work against).
function withSurface(theme: ReturnType<typeof terminalTheme>) {
  const surface = resolveSurfaceColor(theme.background ?? '#ffffff')

  return { ...theme, background: surface, cursorAccent: surface }
}

function transferHasDropCandidates(t: DataTransfer): boolean {
  if (t.types?.includes(HERMES_PATHS_MIME)) {
    return true
  }

  if ((t.files?.length ?? 0) > 0) {
    return true
  }

  for (let i = 0; i < (t.items?.length ?? 0); i += 1) {
    if (t.items[i]?.kind === 'file') {
      return true
    }
  }

  return false
}

function collectDroppedPaths(t: DataTransfer): string[] {
  const seen = new Set<string>()

  const push = (value: unknown) => {
    if (typeof value !== 'string') {
      return
    }

    const path = value.trim()

    if (path) {
      seen.add(path)
    }
  }

  try {
    const raw = t.getData(HERMES_PATHS_MIME)

    if (raw) {
      for (const entry of JSON.parse(raw) as { path?: unknown }[]) {
        push(entry?.path)
      }
    }
  } catch {
    // Malformed in-app drag payload — fall through to OS files.
  }

  const getPath = window.hermesDesktop?.getPathForFile

  const addFile = (file: File | null) => {
    if (!file || !getPath) {
      return
    }

    try {
      push(getPath(file))
    } catch {
      // File handle unavailable.
    }
  }

  for (let i = 0; i < (t.files?.length ?? 0); i += 1) {
    addFile(t.files.item(i))
  }

  for (let i = 0; i < (t.items?.length ?? 0); i += 1) {
    const item = t.items[i]

    if (item?.kind === 'file') {
      addFile(item.getAsFile())
    }
  }

  return [...seen]
}

function quotePathForShell(path: string, shellName: string): string {
  const shell = shellName.toLowerCase()

  if (shell.includes('powershell') || shell.includes('pwsh')) {
    return `'${path.replace(/'/g, "''")}'`
  }

  if (shell.includes('cmd')) {
    return `"${path.replace(/"/g, '""')}"`
  }

  return `'${path.replace(/'/g, "'\\''")}'`
}

export function useTerminalSession({ cwd, onAddSelectionToChat }: UseTerminalSessionOptions) {
  // Key off renderedMode (the painted surface type), not resolvedMode (the
  // clicked switch) — a skin can keep a light surface in "dark" mode, and we
  // must match the surface or the ANSI palette inverts against it. themeName
  // re-resolves the canvas surface on skin switches (same mode, new tint).
  const { renderedMode, theme, themeName } = useTheme()
  // Adopt the skin's ANSI palette when it ships one (imported VS Code themes do),
  // matched to the painted variant; built-in skins carry none, so the terminal
  // keeps its VS Code defaults. withSurface still owns the background, so this
  // never touches transparency.
  const ansiPalette = renderedMode === 'dark' ? (theme.darkTerminal ?? theme.terminal) : theme.terminal
  const activeTheme = useMemo(() => terminalTheme(renderedMode, ansiPalette), [renderedMode, ansiPalette])
  const initialThemeRef = useRef(activeTheme)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const webglRef = useRef<WebglAddon | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const shellNameRef = useRef('shell')
  const selectionLabelRef = useRef('')
  const selectionRef = useRef('')
  const onAddSelectionToChatRef = useRef(onAddSelectionToChat)
  const [status, setStatus] = useState<TerminalStatus>('starting')
  const [selection, setSelection] = useState('')
  const [selectionStyle, setSelectionStyle] = useState<CSSProperties | null>(null)
  const [shellName, setShellName] = useState('shell')

  useEffect(() => {
    onAddSelectionToChatRef.current = onAddSelectionToChat
  }, [onAddSelectionToChat])

  // Live selection at call time. A redraw-heavy TUI (spinners, clocks) outruns
  // onSelectionChange, so trust xterm directly — fall back to the native
  // selection — rather than the cached ref / React state.
  const readSelection = useCallback(
    () => termRef.current?.getSelection() || window.getSelection()?.toString() || '',
    []
  )

  const addSelectionToChat = useCallback(() => {
    const selectedText = readSelection() || selectionRef.current
    const trimmed = selectedText.trim()

    if (!trimmed) {
      return
    }

    const label =
      selectionLabelRef.current ||
      (termRef.current ? terminalSelectionLabel(termRef.current, shellNameRef.current, selectedText) : 'selection')

    onAddSelectionToChatRef.current(trimmed, label)
    termRef.current?.clearSelection()
    selectionRef.current = ''
    selectionLabelRef.current = ''
    setSelection('')
    setSelectionStyle(null)
    triggerHaptic('selection')
  }, [readSelection])

  // Always listen — gating on the React selection state misses selections the
  // TUI redraw races. Only swallow ⌘/Ctrl+L when there's text to send, else it
  // must reach the shell as clear-screen.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isAddSelectionShortcut(event) || !readSelection().trim()) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      addSelectionToChat()
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [addSelectionToChat, readSelection])

  useEffect(() => {
    const host = hostRef.current
    const terminalApi = window.hermesDesktop?.terminal

    if (!host || !terminalApi) {
      setStatus('closed')

      return
    }

    let disposed = false
    const cleanup: Array<() => void> = []
    let lastSentSize: { cols: number; rows: number } | null = null

    const term = new Terminal({
      allowProposedApi: true,
      allowTransparency: true,
      convertEol: true,
      cursorBlink: true,
      fontFamily: "'SF Mono', 'Menlo', 'Cascadia Code', 'JetBrains Mono', monospace",
      fontSize: 11,
      lineHeight: 1.12,
      // Full-screen TUIs (hermes --tui, vim) grab the mouse, so a plain drag
      // can't select — ⌥-drag (macOS) / Shift-drag (else) forces a native
      // selection over mouse-mode apps, which ⌘/Ctrl+L then sends to chat.
      macOptionClickForcesSelection: true,
      macOptionIsMeta: true,
      // VS Code/Cursor's secret sauce: terminal.integrated.minimumContrastRatio
      // defaults to 4.5 there. xterm defaults to 1 (off), which paints the raw
      // saturated ANSI palette — vivid green/cyan on white reads as candy.
      // Clamping to 4.5:1 darkens/lightens foregrounds against the background
      // at render time, matching the muted ink-like look of their terminal.
      minimumContrastRatio: 4.5,
      scrollback: 1000,
      theme: withSurface(initialThemeRef.current)
    })

    const fit = new FitAddon()

    termRef.current = term
    term.loadAddon(fit)
    term.loadAddon(new Unicode11Addon())
    term.loadAddon(new WebLinksAddon())
    term.unicode.activeVersion = '11'

    // Let the GUI chat agent read this pane via the `read_terminal` tool: the
    // gateway's terminal.read.request handler serializes the buffer through this.
    setActiveTerminalReader(makeTerminalReader(term))

    const onDragOver = (e: DragEvent) => {
      if (!e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) {
        return
      }

      e.preventDefault()
      e.stopPropagation()
      e.dataTransfer.dropEffect = 'copy'
    }

    const onDrop = (e: DragEvent) => {
      const id = sessionIdRef.current

      if (!id || !e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) {
        return
      }

      e.preventDefault()
      e.stopPropagation()
      const paths = collectDroppedPaths(e.dataTransfer)

      if (!paths.length) {
        return
      }

      void terminalApi.write(id, `${paths.map(p => quotePathForShell(p, shellNameRef.current)).join(' ')} `)
      term.focus()
      triggerHaptic('selection')
    }

    host.addEventListener('dragenter', onDragOver)
    host.addEventListener('dragover', onDragOver)
    host.addEventListener('drop', onDrop)
    cleanup.push(() => {
      host.removeEventListener('dragenter', onDragOver)
      host.removeEventListener('dragover', onDragOver)
      host.removeEventListener('drop', onDrop)
    })

    // A fresh prompt should sit at the top. Every resize SIGWINCHes the shell,
    // which reprints its prompt and can leave stale blank rows above it. While
    // the session is pristine (nothing run yet) we ask the shell to clear +
    // redraw via Ctrl-L (\f) after the resize settles. Ctrl-L preserves
    // multi-line prompts (term.clear() would drop all but the cursor row) and we
    // stop the moment real output exists, so command scrollback is never wiped.
    let promptPristine = true
    let gapCleanupTimer = 0

    // While armed, strip leading blank rows so the prompt lands at the very top
    // (no starship `add_newline` gap). Re-armed before each Ctrl-L redraw so the
    // resize cleanup doesn't reintroduce the blank line.
    let stripLeading = true

    const armedWrite = (data: string) => {
      if (!stripLeading) {
        term.write(data)

        return
      }

      const next = stripInitialPromptGap(data)
      const visible = stripEscapeSequences(next).replace(/[\s%]/g, '')

      if (!visible) {
        // Spacer / lone clear-screen / zsh `%` marker: apply control codes but
        // drop the blank text and stay armed so the prompt still lands at top.
        const controls = keepEscapeSequences(next)

        if (controls) {
          term.write(controls)
        }

        return
      }

      stripLeading = false
      term.write(next)
    }

    const scheduleGapCleanup = () => {
      if (!promptPristine) {
        return
      }

      if (gapCleanupTimer) {
        window.clearTimeout(gapCleanupTimer)
      }

      gapCleanupTimer = window.setTimeout(() => {
        gapCleanupTimer = 0
        const id = sessionIdRef.current

        if (disposed || !id || !promptPristine) {
          return
        }

        stripLeading = true
        void terminalApi.write(id, '\f')
        term.clearSelection()
      }, 120)
    }

    cleanup.push(() => {
      if (gapCleanupTimer) {
        window.clearTimeout(gapCleanupTimer)
      }
    })

    const fitAndResize = () => {
      if (disposed || !host.isConnected || host.clientWidth <= 0 || host.clientHeight <= 0) {
        return
      }

      try {
        fit.fit()
      } catch {
        return
      }

      const id = sessionIdRef.current

      if (id && (lastSentSize?.cols !== term.cols || lastSentSize?.rows !== term.rows)) {
        lastSentSize = { cols: term.cols, rows: term.rows }
        void terminalApi.resize(id, { cols: term.cols, rows: term.rows })
        scheduleGapCleanup()
      }
    }

    // Coalesce ResizeObserver bursts through rAF — running fit.fit()
    // synchronously while sibling panes are mid-transition (e.g. file browser
    // collapsing to 0px) crashes the WebGL renderer mid texture-atlas rebuild.
    let pendingFrame = 0

    const scheduleResize = () => {
      if (pendingFrame) {
        return
      }

      pendingFrame = window.requestAnimationFrame(() => {
        pendingFrame = 0

        if (!disposed) {
          fitAndResize()
        }
      })
    }

    const resizeObserver = new ResizeObserver(scheduleResize)
    resizeObserver.observe(host)
    cleanup.push(() => {
      resizeObserver.disconnect()

      if (pendingFrame) {
        window.cancelAnimationFrame(pendingFrame)
      }
    })

    const dataDisposable = term.onData(data => {
      const id = sessionIdRef.current

      if (id) {
        // Once the user submits a line, real output may follow — stop the
        // pristine-prompt gap cleanup so we never clear command scrollback.
        if (promptPristine && data.includes('\r')) {
          promptPristine = false
        }

        void terminalApi.write(id, data)
      }
    })

    cleanup.push(() => dataDisposable.dispose())

    const selectionDisposable = term.onSelectionChange(() => {
      const next = term.getSelection()
      selectionRef.current = next
      selectionLabelRef.current = next.trim() ? terminalSelectionLabel(term, shellNameRef.current, next) : ''
      setSelection(next)
      setSelectionStyle(next.trim() ? terminalSelectionAnchor(host) : null)
    })

    cleanup.push(() => selectionDisposable.dispose())

    const startSession = () =>
      void terminalApi
        .start({ cols: term.cols, cwd, rows: term.rows })
        .then(session => {
          if (disposed) {
            void terminalApi.dispose(session.id)

            return
          }

          sessionIdRef.current = session.id
          lastSentSize = { cols: term.cols, rows: term.rows }
          shellNameRef.current = session.shell || 'shell'
          setShellName(session.shell || 'shell')

          const initial = term.hasSelection() ? term.getSelection() : ''
          selectionRef.current = initial
          selectionLabelRef.current = initial ? terminalSelectionLabel(term, shellNameRef.current, initial) : ''

          setStatus('open')

          cleanup.push(
            terminalApi.onData(session.id, armedWrite),
            terminalApi.onExit(session.id, ({ code, signal }) => {
              setStatus('closed')
              term.write(`\r\n[terminal exited${signal ? `: ${signal}` : code !== null ? `: ${code}` : ''}]\r\n`)
            })
          )

          window.requestAnimationFrame(() => {
            fitAndResize()
            term.clearSelection() // drop any selection painted over transient boot rows
            term.focus()
          })
        })
        .catch(error => {
          setStatus('closed')
          term.write(`Terminal failed to start: ${error instanceof Error ? error.message : String(error)}\r\n`)
        })

    // Open + fit + start only once webfonts settle. Fitting with fallback metrics
    // picks the wrong row count, the shell boots at that size, then the real font
    // loads -> refit -> SIGWINCH -> the shell reprints its prompt lower, leaving
    // stale blank rows (and a stray selection) above it.
    const mount = () => {
      if (disposed || !host.isConnected) {
        return
      }

      term.open(host)
      term.focus()

      // WebGL renderer matches the dashboard ChatPage path; xterm's default DOM
      // renderer paints SGR via CSS classes that visibly mute against our skins.
      try {
        const webgl = new WebglAddon()
        webgl.onContextLoss(() => {
          webgl.dispose()
          webglRef.current = null
        })
        term.loadAddon(webgl)
        webglRef.current = webgl
      } catch (err) {
        console.warn('[hermes-terminal] WebGL unavailable; falling back to DOM', err)
      }

      fitAndResize()
      startSession()
    }

    const fonts = typeof document !== 'undefined' ? document.fonts : undefined

    if (fonts?.ready) {
      void fonts.ready.then(mount, mount)
    } else {
      mount()
    }

    return () => {
      disposed = true
      cleanup.forEach(run => run())
      setActiveTerminalReader(null)

      const id = sessionIdRef.current
      sessionIdRef.current = null

      if (id) {
        void terminalApi.dispose(id)
      }

      term.dispose()
      termRef.current = null
      webglRef.current = null
      shellNameRef.current = 'shell'
      selectionRef.current = ''
      selectionLabelRef.current = ''
    }
  }, [addSelectionToChat, cwd])

  useEffect(() => {
    const term = termRef.current

    if (!term) {
      return
    }

    // Re-resolve the surface in a rAF: ThemeProvider's applyTheme repaints the
    // CSS vars in a sibling effect that runs after this one, so reading now
    // would lag a mode behind. By the next frame the vars are current.
    const raf = requestAnimationFrame(() => {
      term.options.theme = withSurface(activeTheme)
      // The WebGL renderer caches glyph colors in a texture atlas, so a
      // light/dark switch leaves already-drawn cells stale until the atlas is
      // cleared. No-op for the DOM fallback.
      webglRef.current?.clearTextureAtlas()
    })

    return () => cancelAnimationFrame(raf)
  }, [activeTheme, themeName])

  return {
    addSelectionToChat,
    hostRef,
    selection,
    selectionStyle,
    shellName,
    status
  }
}
