import type { Terminal } from '@xterm/xterm'

import { translateNow } from '@/i18n'
import { BROWSER_UPLOAD_MAX_BYTES } from '@/lib/browser-uploads'
import { triggerHaptic } from '@/lib/haptics'
import { notify, notifyError } from '@/store/notifications'

const HERMES_PATHS_MIME = 'application/x-hermes-paths'

/** A host path, or an OS file with no path the browser must upload to get one. */
type DroppedEntry = File | string

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

function collectDroppedEntries(t: DataTransfer): DroppedEntry[] {
  const seen = new Set<string>()
  const entries: DroppedEntry[] = []

  const push = (value: unknown) => {
    if (typeof value !== 'string') {
      return
    }

    const path = value.trim()

    if (path && !seen.has(path)) {
      seen.add(path)
      entries.push(path)
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
  // Only the browser-hosted Desktop stages bytes; Electron resolves an OS file to its own path.
  const canStage = Boolean(window.hermesDesktop?.stageFileForAttach)
  const fileCount = t.files?.length ?? 0

  // `items` repeats `files` with fresh File objects, so it supplies files to stage only when
  // `files` is empty.
  const addFile = (file: File | null, stageable: boolean) => {
    if (!file) {
      return
    }

    let path = ''

    try {
      path = getPath?.(file) ?? ''
    } catch {
      // File handle unavailable.
    }

    if (path) {
      push(path)
    } else if (canStage && stageable) {
      entries.push(file)
    }
  }

  for (let i = 0; i < fileCount; i += 1) {
    addFile(t.files.item(i), true)
  }

  for (let i = 0; i < (t.items?.length ?? 0); i += 1) {
    const item = t.items[i]

    if (item?.kind === 'file') {
      addFile(item.getAsFile(), fileCount === 0)
    }
  }

  return entries
}

// The browser uploads each dropped file to the host (the composer's staged upload) and types the
// host path it gets back. A file over the upload limit is never sent; a warning names it.
async function stageDroppedEntries(entries: DroppedEntry[]): Promise<string[]> {
  const stage = window.hermesDesktop?.stageFileForAttach
  const tooLarge: string[] = []

  const paths = await Promise.all(
    entries.map(async entry => {
      if (typeof entry === 'string') {
        return entry
      }

      if (entry.size > BROWSER_UPLOAD_MAX_BYTES) {
        tooLarge.push(entry.name)

        return ''
      }

      try {
        return (await stage?.(entry)) ?? ''
      } catch (error) {
        notifyError(error, translateNow('rightSidebar.terminalDropUploadFailed', entry.name))

        return ''
      }
    })
  )

  if (tooLarge.length) {
    notify({
      kind: 'warning',
      title: translateNow('rightSidebar.terminalDropTooLargeTitle'),
      message: translateNow(
        'rightSidebar.terminalDropTooLarge',
        tooLarge.join(', '),
        `${BROWSER_UPLOAD_MAX_BYTES / (1024 * 1024)} MiB`
      )
    })
  }

  return paths.filter(Boolean)
}

const POSIX_SHELLS = new Set(['ash', 'bash', 'dash', 'ksh', 'mksh', 'sh', 'zsh'])

// Every shell reads a path made only of these as one plain word, so a shell the quoter does not
// name still gets it as typed. A leading - or ~ would read as an option or a home directory.
const PLAIN_PATH = /^[\w./:][\w./:-]*$/

/** The shell's program name: `C:\...\pwsh.exe` is `pwsh`, a login shell's `-zsh` is `zsh`. */
function shellProgram(shellName: string): string {
  return shellName
    .toLowerCase()
    .replace(/^.*[\\/]/, '')
    .replace(/^-/, '')
    .replace(/\.exe$/, '')
}

// The path quoted so the terminal's shell reads it as one literal argument, whatever characters
// the name holds: a name must never end the quoting and add a command for the next Enter. Null
// when this shell has no quoting that does that: a shell not named here (nu, xonsh, elvish, the
// remote shell of an SSH tab) gets only a plain path, typed as it is.
function quotePathForShell(path: string, shellName: string): null | string {
  const name = shellProgram(shellName)

  if (name.startsWith('pwsh') || name.startsWith('powershell')) {
    // PowerShell reads the typographic quotes U+2018..U+201B as single quotes too; doubling any of
    // them keeps it literal.
    return `'${path.replace(/['\u2018\u2019\u201a\u201b]/g, '$&$&')}'`
  }

  if (name === 'cmd') {
    // With delayed expansion on (cmd /V:ON, or set in the registry) cmd replaces !NAME! inside
    // double quotes too and drops a lone !. Nothing tells the terminal whether it is on, and no
    // quoting keeps a ! literal both ways.
    if (path.includes('!')) {
      return null
    }

    // cmd expands %NAME% inside double quotes before it reads them, and a value holding a quote
    // would end the quoting. Each % goes outside the quotes as ^%, which no variable name matches.
    return path
      .split('%')
      .map(part => `"${part.replace(/"/g, '""')}"`)
      .join('^%')
  }

  if (name === 'fish') {
    // Inside fish's single quotes \' and \\ are escapes.
    return `'${path.replace(/[\\']/g, '\\$&')}'`
  }

  const quoted = `'${path.replace(/'/g, "'\\''")}'`

  if (name === 'csh' || name === 'tcsh') {
    // csh and tcsh expand !-history even inside single quotes; \! is the escape there.
    return quoted.replace(/!/g, '\\!')
  }

  if (POSIX_SHELLS.has(name)) {
    return quoted
  }

  return PLAIN_PATH.test(path) ? path : null
}

// C0 controls, DEL and C1 controls. A line editor acts on them inside any quoting (^U erases the
// line typed so far, CR runs it), so a path holding one is never typed.
// eslint-disable-next-line no-control-regex -- matching control characters is the point
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f-\u009f]/g

function hasControlCharacter(path: string): boolean {
  return path.search(CONTROL_CHARACTERS) !== -1
}

/** The path with each control character spelled `\xNN`, so a notice can name it. */
function printablePath(path: string): string {
  return path.replace(CONTROL_CHARACTERS, char => `\\x${char.charCodeAt(0).toString(16).padStart(2, '0')}`)
}

interface TerminalDropOptions {
  getSessionId: () => string | null
  getShellName: () => string
  markActivity: () => void
  term: Terminal
}

// Dropping files (from the OS or the app's file trees) types their shell-quoted
// paths at the prompt, like a native terminal. In the browser-hosted Desktop an
// OS file is uploaded to the host first and its host path is typed. The paths
// go through xterm's paste, so a shell that turned bracketed paste on inserts
// them without acting on them. Returns the listener teardown.
export function bindTerminalDrop(
  host: HTMLElement,
  { getSessionId, getShellName, markActivity, term }: TerminalDropOptions
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
    const entries = collectDroppedEntries(e.dataTransfer)

    if (!entries.length) {
      return
    }

    const typePaths = (paths: string[]) => {
      const refused = paths.filter(hasControlCharacter)
      const shellName = getShellName()
      const typed: string[] = []
      const unquotable: string[] = []

      for (const path of paths) {
        if (hasControlCharacter(path)) {
          continue
        }

        const quoted = quotePathForShell(path, shellName)

        if (quoted === null) {
          unquotable.push(path)
        } else {
          typed.push(quoted)
        }
      }

      if (refused.length) {
        notify({
          kind: 'warning',
          title: translateNow('rightSidebar.terminalDropControlTitle'),
          message: translateNow('rightSidebar.terminalDropControl', refused.map(printablePath).join(', '))
        })
      }

      if (unquotable.length) {
        notify({
          kind: 'warning',
          title: translateNow('rightSidebar.terminalDropControlTitle'),
          message: translateNow('rightSidebar.terminalDropUnquotable', shellProgram(shellName), unquotable.join(', '))
        })
      }

      if (!typed.length) {
        return
      }

      markActivity()
      term.paste(`${typed.join(' ')} `)
      term.focus()
      triggerHaptic('selection')
    }

    if (entries.every((entry): entry is string => typeof entry === 'string')) {
      typePaths(entries)

      return
    }

    void stageDroppedEntries(entries).then(typePaths)
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
