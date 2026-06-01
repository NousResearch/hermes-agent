import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { WebglAddon } from '@xterm/addon-webgl'
import { Terminal } from '@xterm/xterm'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { triggerHaptic } from '@/lib/haptics'

import { isAddSelectionShortcut, terminalSelectionAnchor, terminalSelectionLabel, terminalTheme } from './selection'

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

function isStartupSpacer(data: string) {
  const text = stripEscapeSequences(data).replace(/[\s\r\n]/g, '')

  return text === '' || text === '%'
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

function transferHasDropCandidates(t: DataTransfer): boolean {
  if (t.types?.includes(HERMES_PATHS_MIME)) return true
  if ((t.files?.length ?? 0) > 0) return true

  for (let i = 0; i < (t.items?.length ?? 0); i += 1) {
    if (t.items[i]?.kind === 'file') return true
  }

  return false
}

function collectDroppedPaths(t: DataTransfer): string[] {
  const seen = new Set<string>()
  const push = (value: unknown) => {
    if (typeof value !== 'string') return
    const path = value.trim()
    if (path) seen.add(path)
  }

  try {
    const raw = t.getData(HERMES_PATHS_MIME)
    if (raw) for (const entry of JSON.parse(raw) as { path?: unknown }[]) push(entry?.path)
  } catch {
    // Malformed in-app drag payload — fall through to OS files.
  }

  const getPath = window.hermesDesktop?.getPathForFile
  const addFile = (file: File | null) => {
    if (!file || !getPath) return
    try {
      push(getPath(file))
    } catch {
      // File handle unavailable.
    }
  }

  for (let i = 0; i < (t.files?.length ?? 0); i += 1) addFile(t.files.item(i))
  for (let i = 0; i < (t.items?.length ?? 0); i += 1) {
    const item = t.items[i]
    if (item?.kind === 'file') addFile(item.getAsFile())
  }

  return [...seen]
}

function quotePathForShell(path: string, shellName: string): string {
  const shell = shellName.toLowerCase()
  if (shell.includes('powershell') || shell.includes('pwsh')) return `'${path.replace(/'/g, "''")}'`
  if (shell.includes('cmd')) return `"${path.replace(/"/g, '""')}"`
  return `'${path.replace(/'/g, "'\\''")}'`
}

export function useTerminalSession({ cwd, onAddSelectionToChat }: UseTerminalSessionOptions) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
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

  const addSelectionToChat = useCallback(() => {
    const selectedText = selectionRef.current || termRef.current?.getSelection() || ''

    const label =
      selectionLabelRef.current ||
      (termRef.current ? terminalSelectionLabel(termRef.current, shellNameRef.current, selectedText) : 'selection')

    const trimmed = selectedText.trim()

    if (!trimmed) {
      return
    }

    onAddSelectionToChatRef.current(trimmed, label)
    termRef.current?.clearSelection()
    selectionRef.current = ''
    selectionLabelRef.current = ''
    setSelection('')
    setSelectionStyle(null)
    triggerHaptic('selection')
  }, [])

  useEffect(() => {
    if (!selection.trim()) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isAddSelectionShortcut(event)) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      addSelectionToChat()
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [addSelectionToChat, selection])

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
      macOptionIsMeta: true,
      scrollback: 1000,
      theme: terminalTheme()
    })

    const fit = new FitAddon()

    termRef.current = term
    term.loadAddon(fit)
    term.loadAddon(new Unicode11Addon())
    term.loadAddon(new WebLinksAddon())
    term.unicode.activeVersion = '11'
    term.open(host)
    term.focus()

    // WebGL renderer matches the dashboard ChatPage path; xterm's default DOM
    // renderer paints SGR via CSS classes that visibly mute against our skins.
    try {
      const webgl = new WebglAddon()
      webgl.onContextLoss(() => webgl.dispose())
      term.loadAddon(webgl)
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[hermes-terminal] WebGL unavailable; falling back to DOM', err)
    }

    const onDragOver = (e: DragEvent) => {
      if (!e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) return
      e.preventDefault()
      e.stopPropagation()
      e.dataTransfer.dropEffect = 'copy'
    }

    const onDrop = (e: DragEvent) => {
      const id = sessionIdRef.current
      if (!id || !e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) return
      e.preventDefault()
      e.stopPropagation()
      const paths = collectDroppedPaths(e.dataTransfer)
      if (!paths.length) return
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
      }
    }

    // Coalesce ResizeObserver bursts through rAF — running fit.fit()
    // synchronously while sibling panes are mid-transition (e.g. file browser
    // collapsing to 0px) crashes the WebGL renderer mid texture-atlas rebuild.
    let pendingFrame = 0
    const scheduleResize = () => {
      if (pendingFrame) return
      pendingFrame = window.requestAnimationFrame(() => {
        pendingFrame = 0
        if (!disposed) fitAndResize()
      })
    }

    const resizeObserver = new ResizeObserver(scheduleResize)
    resizeObserver.observe(host)
    cleanup.push(() => {
      resizeObserver.disconnect()
      if (pendingFrame) window.cancelAnimationFrame(pendingFrame)
    })

    const dataDisposable = term.onData(data => {
      const id = sessionIdRef.current

      if (id) {
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

    term.attachCustomKeyEventHandler(event => {
      if (event.type !== 'keydown') {
        return true
      }

      if (isAddSelectionShortcut(event) && term.hasSelection()) {
        event.preventDefault()
        addSelectionToChat()

        return false
      }

      return true
    })

    fitAndResize()

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

        if (term.hasSelection()) {
          const currentSelection = term.getSelection()
          selectionRef.current = currentSelection
          selectionLabelRef.current = terminalSelectionLabel(term, shellNameRef.current, currentSelection)
        } else {
          selectionRef.current = ''
          selectionLabelRef.current = ''
        }

        setStatus('open')
        let wrotePromptContent = false

        cleanup.push(
          terminalApi.onData(session.id, data => {
            if (wrotePromptContent) {
              term.write(data)

              return
            }

            if (isStartupSpacer(data)) {
              return
            }

            const next = stripInitialPromptGap(data)

            if (next) {
              wrotePromptContent = true
              term.write(next)
            }
          }),
          terminalApi.onExit(session.id, sessionExit => {
            const { code, signal } = sessionExit
            setStatus('closed')
            term.write(`\r\n[terminal exited${signal ? `: ${signal}` : code !== null ? `: ${code}` : ''}]\r\n`)
          })
        )
        window.requestAnimationFrame(() => {
          fitAndResize()
          term.focus()
        })
      })
      .catch(error => {
        setStatus('closed')
        term.write(`Terminal failed to start: ${error instanceof Error ? error.message : String(error)}\r\n`)
      })

    return () => {
      disposed = true
      cleanup.forEach(run => run())

      const id = sessionIdRef.current
      sessionIdRef.current = null

      if (id) {
        void terminalApi.dispose(id)
      }

      term.dispose()
      termRef.current = null
      shellNameRef.current = 'shell'
      selectionRef.current = ''
      selectionLabelRef.current = ''
    }
  }, [addSelectionToChat, cwd])

  return {
    addSelectionToChat,
    hostRef,
    selection,
    selectionStyle,
    shellName,
    status
  }
}
