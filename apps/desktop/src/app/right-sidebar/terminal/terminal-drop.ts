import type { Terminal } from '@xterm/xterm'

import { triggerHaptic } from '@/lib/haptics'

type TerminalApi = Window['hermesDesktop']['terminal']

const HERMES_PATHS_MIME = 'application/x-hermes-paths'

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

interface TerminalDropOptions {
  getSessionId: () => string | null
  getShellName: () => string
  markActivity: () => void
  term: Terminal
  terminalApi: TerminalApi
}

// Dropping files (from the OS or the app's file trees) types their shell-quoted
// paths at the prompt, like a native terminal. Returns the listener teardown.
export function bindTerminalDrop(
  host: HTMLElement,
  { getSessionId, getShellName, markActivity, term, terminalApi }: TerminalDropOptions
): () => void {
  const onDragOver = (e: DragEvent) => {
    if (!e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) {
      return
    }

    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
  }

  const onDrop = (e: DragEvent) => {
    const id = getSessionId()

    if (!id || !e.dataTransfer || !transferHasDropCandidates(e.dataTransfer)) {
      return
    }

    e.preventDefault()
    e.stopPropagation()
    const paths = collectDroppedPaths(e.dataTransfer)

    if (!paths.length) {
      return
    }

    markActivity()
    void terminalApi.write(id, `${paths.map(p => quotePathForShell(p, getShellName())).join(' ')} `)
    term.focus()
    triggerHaptic('selection')
  }

  host.addEventListener('dragenter', onDragOver)
  host.addEventListener('dragover', onDragOver)
  host.addEventListener('drop', onDrop)

  return () => {
    host.removeEventListener('dragenter', onDragOver)
    host.removeEventListener('dragover', onDragOver)
    host.removeEventListener('drop', onDrop)
  }
}
