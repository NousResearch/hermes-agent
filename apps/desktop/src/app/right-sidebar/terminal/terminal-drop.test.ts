import { Terminal } from '@xterm/xterm'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

import { bindTerminalDrop } from './terminal-drop'

const desktopWindow = window as unknown as { hermesDesktop?: Partial<Window['hermesDesktop']> }
const MIB = 1024 * 1024
// eslint-disable-next-line no-control-regex -- the assertion is that none of these reach the shell
const CONTROL_CHARACTER = /[\u0000-\u001f\u007f-\u009f]/
const BRACKETED_PASTE_ON = '\x1b[?2004h'
const originalMatchMedia = window.matchMedia

// xterm measures the device pixel ratio through matchMedia when it opens; jsdom has none.
beforeAll(() => {
  window.matchMedia = (query: string) =>
    ({
      addEventListener: () => undefined,
      addListener: () => undefined,
      matches: false,
      media: query,
      removeEventListener: () => undefined,
      removeListener: () => undefined
    }) as unknown as MediaQueryList
})

afterAll(() => {
  window.matchMedia = originalMatchMedia
})

function file(name: string, size = 4): File {
  const value = new File(['data'], name, { type: 'text/plain' })
  Object.defineProperty(value, 'size', { value: size })

  return value
}

function transfer(files: File[], hermesPaths?: string[]): DataTransfer {
  const list = Object.assign([...files], { item: (index: number) => files[index] ?? null })

  return {
    files: list,
    getData: (type: string) =>
      type === 'application/x-hermes-paths' && hermesPaths ? JSON.stringify(hermesPaths.map(path => ({ path }))) : '',
    items: files.map(entry => ({ getAsFile: () => entry, kind: 'file' })),
    types: [...(hermesPaths ? ['application/x-hermes-paths'] : []), ...(files.length ? ['Files'] : [])]
  } as unknown as DataTransfer
}

function drop(host: HTMLElement, data: DataTransfer): Event {
  const event = new Event('drop', { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'dataTransfer', { value: data })
  host.dispatchEvent(event)

  return event
}

// A real xterm, wired to the shell the way the session hook wires it: everything the terminal
// emits (keystrokes, pastes) is written to the PTY.
function bind(shellName = 'bash') {
  const host = document.createElement('div')
  document.body.append(host)
  const term = new Terminal({ allowProposedApi: true, cols: 80, rows: 10 })
  term.open(host)
  const write = vi.fn().mockResolvedValue(undefined)
  const input = term.onData(data => void write('term-1', data))
  const focus = vi.spyOn(term, 'focus')

  const unbindDrop = bindTerminalDrop(host, {
    getSessionId: () => 'term-1',
    getShellName: () => shellName,
    markActivity: () => undefined,
    term
  })

  const unbind = () => {
    unbindDrop()
    input.dispose()
    term.dispose()
    host.remove()
  }

  return { focus, host, term, unbind, write }
}

function typedText(write: ReturnType<typeof vi.fn>): string {
  return write.mock.calls.map(([, data]) => String(data)).join('')
}

afterEach(() => {
  Reflect.deleteProperty(desktopWindow, 'hermesDesktop')
  clearNotifications()
  vi.restoreAllMocks()
})

describe('terminal drop in the browser-hosted Desktop', () => {
  function browserBridge(stage: (file: File) => Promise<string>) {
    const stageFileForAttach = vi.fn(stage)
    desktopWindow.hermesDesktop = { getPathForFile: () => '', stageFileForAttach }

    return stageFileForAttach
  }

  it('stages each dropped file on the host and types its quoted host path', async () => {
    const stage = browserBridge(async entry => `/srv/hermes/uploads/${entry.name}`)
    const { host, unbind, write } = bind('bash')

    const event = drop(host, transfer([file("it's.txt"), file('b c.log')]))

    expect(event.defaultPrevented).toBe(true)
    await vi.waitFor(() => expect(write).toHaveBeenCalledTimes(1))
    expect(write).toHaveBeenCalledWith('term-1', "'/srv/hermes/uploads/it'\\''s.txt' '/srv/hermes/uploads/b c.log' ")
    expect(stage).toHaveBeenCalledTimes(2)
    unbind()
  })

  it("quotes staged paths for the terminal's own shell", async () => {
    browserBridge(async () => 'C:\\Users\\me\\.hermes\\uploads\\it\'s.txt')
    const { host, unbind, write } = bind('powershell.exe')

    drop(host, transfer([file("it's.txt")]))

    await vi.waitFor(() => expect(write).toHaveBeenCalledWith('term-1', "'C:\\Users\\me\\.hermes\\uploads\\it''s.txt' "))
    unbind()
  })

  it('keeps in-app paths and staged files in drop order', async () => {
    browserBridge(async entry => `/srv/hermes/uploads/${entry.name}`)
    const { host, unbind, write } = bind('zsh')

    drop(host, transfer([file('notes.md')], ['/work/project/src']))

    await vi.waitFor(() =>
      expect(write).toHaveBeenCalledWith('term-1', "'/work/project/src' '/srv/hermes/uploads/notes.md' ")
    )
    unbind()
  })

  it('never uploads a file over 16 MiB and names it in a warning', async () => {
    const stage = browserBridge(async entry => `/srv/hermes/uploads/${entry.name}`)
    const { host, unbind, write } = bind('bash')

    drop(host, transfer([file('huge.iso', 16 * MIB + 1), file('small.txt', 16 * MIB)]))

    await vi.waitFor(() => expect(write).toHaveBeenCalledWith('term-1', "'/srv/hermes/uploads/small.txt' "))
    expect(stage).toHaveBeenCalledTimes(1)
    expect(stage.mock.calls[0][0].name).toBe('small.txt')
    const [warning] = $notifications.get()
    expect(warning?.kind).toBe('warning')
    expect(`${warning?.title} ${warning?.message}`).toContain('huge.iso')
    expect(`${warning?.title} ${warning?.message}`).toContain('16 MiB')
    unbind()
  })

  it("reports a failed upload with the host's reason and types nothing for it", async () => {
    browserBridge(async () => {
      throw new Error('File upload failed (502)')
    })
    const { focus, host, unbind, write } = bind('bash')

    drop(host, transfer([file('report.pdf')]))

    await vi.waitFor(() => expect($notifications.get()).toHaveLength(1))
    const [error] = $notifications.get()
    expect(error?.kind).toBe('error')
    expect(`${error?.title} ${error?.message}`).toContain('report.pdf')
    expect(`${error?.title} ${error?.message}`).toContain('502')
    expect(write).not.toHaveBeenCalled()
    expect(focus).not.toHaveBeenCalled()
    unbind()
  })
})

describe('terminal drop in Electron Desktop', () => {
  it('types the native path of each dropped file without uploading it', async () => {
    const getPathForFile = vi.fn((entry: File) => `/home/me/${entry.name}`)
    desktopWindow.hermesDesktop = { getPathForFile }
    const { focus, host, unbind, write } = bind('bash')

    drop(host, transfer([file('a.txt'), file('b c.txt')]))

    expect(write).toHaveBeenCalledWith('term-1', "'/home/me/a.txt' '/home/me/b c.txt' ")
    expect(focus).toHaveBeenCalled()
    unbind()
  })

  it('ignores a dropped file Electron cannot resolve to a path', async () => {
    desktopWindow.hermesDesktop = { getPathForFile: () => '' }
    const { host, unbind, write } = bind('bash')

    drop(host, transfer([file('from-a-web-page.png')]))
    await Promise.resolve()

    expect(write).not.toHaveBeenCalled()
    expect($notifications.get()).toHaveLength(0)
    unbind()
  })
})

describe('terminal drop never types keystrokes', () => {
  // A file name may hold bytes a shell reads as keys: ^U erases what was typed and CR runs the
  // line, so a planted name could run a command with no Enter pressed. No quoting stops a
  // line editor from acting on them, so such a path is never typed: a warning names it (control
  // characters spelled as \xNN) and the other dropped paths are still typed.
  it('skips an app file-tree path holding control characters and names it in a warning', () => {
    const { host, unbind, write } = bind('bash')

    drop(host, transfer([], ['/work/drop/notes\x15touch PWNED\r.txt', '/work/drop/ok.txt', '/work/c1\x9b.md', '/work/del\x7f']))

    expect(typedText(write)).not.toMatch(CONTROL_CHARACTER)
    expect(typedText(write)).toBe("'/work/drop/ok.txt' ")
    const [warning] = $notifications.get()
    expect(warning?.kind).toBe('warning')
    expect(warning?.message).toContain('/work/drop/notes\\x15touch PWNED\\x0d.txt')
    expect(warning?.message).toContain('/work/c1\\x9b.md')
    expect(warning?.message).toContain('/work/del\\x7f')
    unbind()
  })

  it('types nothing for an OS file whose native path holds a control character', () => {
    desktopWindow.hermesDesktop = { getPathForFile: (entry: File) => `/home/me/${entry.name}` }
    const { focus, host, unbind, write } = bind('zsh')

    drop(host, transfer([file('report\n.txt')]))

    expect(write).not.toHaveBeenCalled()
    expect(focus).not.toHaveBeenCalled()
    expect($notifications.get()[0]?.message).toContain('/home/me/report\\x0a.txt')
    unbind()
  })

  // A shell that turned bracketed paste on (bash 5.1+, zsh) inserts pasted text without acting
  // on it; the drop arrives the way a paste does, wrapped in the paste markers.
  it('types the paths as a bracketed paste when the shell turned bracketed paste on', async () => {
    const { host, term, unbind, write } = bind('bash')
    await new Promise<void>(resolve => term.write(BRACKETED_PASTE_ON, resolve))

    drop(host, transfer([], ['/work/a b.txt', "/work/it's.md"]))

    expect(write).toHaveBeenCalledExactlyOnceWith('term-1', "\x1b[200~'/work/a b.txt' '/work/it'\\''s.md' \x1b[201~")
    unbind()
  })
})

// How each shell the terminal can run reads the typed text, enough to see where a word ends and
// where a new command starts. Each follows the shell's own manual.

// PowerShell reads U+0027 and the typographic quotes U+2018, U+2019, U+201A, U+201B all as single
// quotes (language spec, "single-quote-character"); inside a single-quoted string any of them
// followed by any of them is one literal quote, and a lone one ends the string.
const POWERSHELL_SINGLE_QUOTES = new Set(["'", '\u2018', '\u2019', '\u201a', '\u201b'])

function powershellWords(text: string): string[] {
  const words: string[] = []
  let word = ''
  let i = 0

  const flush = () => {
    if (word) {
      words.push(word)
    }

    word = ''
  }

  while (i < text.length) {
    const c = text[i]

    if (POWERSHELL_SINGLE_QUOTES.has(c)) {
      i += 1

      for (;;) {
        if (i >= text.length) {
          return [...words, `${word}<unterminated>`]
        }

        if (POWERSHELL_SINGLE_QUOTES.has(text[i])) {
          if (!POWERSHELL_SINGLE_QUOTES.has(text[i + 1] ?? '')) {
            break
          }

          i += 1
        }

        word += text[i]
        i += 1
      }

      i += 1
    } else if (c === ';') {
      flush()
      words.push(';')
      i += 1
    } else if (/\s/.test(c)) {
      flush()
      i += 1
    } else {
      word += c
      i += 1
    }
  }

  flush()

  return words
}

// fish: inside '...' only \' and \\ are escapes (fish manual, "Quotes"); outside quotes a backslash
// escapes the next character, ';' separates commands and '#' starts a comment.
function fishWords(text: string): string[] {
  const words: string[] = []
  let word = ''
  let i = 0

  const flush = () => {
    if (word) {
      words.push(word)
    }

    word = ''
  }

  while (i < text.length) {
    const c = text[i]

    if (c === "'") {
      i += 1

      while (i < text.length && text[i] !== "'") {
        if (text[i] === '\\' && (text[i + 1] === "'" || text[i + 1] === '\\')) {
          i += 1
        }

        word += text[i]
        i += 1
      }

      if (i >= text.length) {
        return [...words, `${word}<unterminated>`]
      }

      i += 1
    } else if (c === '\\') {
      word += text[i + 1] ?? ''
      i += 2
    } else if (c === ';') {
      flush()
      words.push(';')
      i += 1
    } else if (c === '#' && !word) {
      words.push('#comment')

      break
    } else if (/\s/.test(c)) {
      flush()
      i += 1
    } else {
      word += c
      i += 1
    }
  }

  flush()

  return words
}

// cmd.exe at its prompt first replaces %NAME% of every defined variable, inside double quotes too,
// and leaves an undefined one as typed; then a '"' toggles quoting, '^' outside quotes takes the
// next character literally and '&' outside quotes starts a new command.
function cmdWords(text: string, env: Record<string, string>): string[] {
  const expanded = text.replace(/%([^%]+)%/g, (whole, name: string) => env[name.toUpperCase()] ?? whole)
  const words: string[] = []
  let word = ''
  let quoted = false

  const flush = () => {
    if (word) {
      words.push(word)
    }

    word = ''
  }

  for (let i = 0; i < expanded.length; i += 1) {
    const c = expanded[i]

    if (c === '"') {
      quoted = !quoted
    } else if (quoted) {
      word += c
    } else if (c === '^') {
      i += 1
      word += expanded[i] ?? ''
    } else if (c === '&') {
      flush()
      words.push('&')
    } else if (/\s/.test(c)) {
      flush()
    } else {
      word += c
    }
  }

  flush()

  return quoted ? [...words, '<unterminated>'] : words
}

function droppedText(shellName: string, path: string): string {
  const { host, unbind, write } = bind(shellName)

  drop(host, transfer([], [path]))
  const typed = typedText(write)
  unbind()

  return typed
}

describe("terminal drop types each path as one literal word of the terminal's shell", () => {
  // A file name comes from wherever the files did (a cloned repo, an unpacked archive, the agent),
  // so its characters must never end the quoting the drop adds: what the user then runs with
  // Enter is the command they typed, with the path as one argument. The Webapp's host terminal
  // runs PowerShell on Windows and $SHELL elsewhere; native Desktop runs the same shells.
  it.each(['pwsh.exe', 'powershell.exe'])('doubles every quote PowerShell reads as a single quote (%s)', shell => {
    for (const path of [
      'C:\\Users\\me\\Documents\\Jeff\u2019s notes.txt',
      'C:\\work\\repo\\a\u2019; Start-Process calc; \u2019b.txt',
      "C:\\work\\\u2018x\u201a'\u201b; calc; '.txt"
    ]) {
      expect(powershellWords(droppedText(shell, path))).toEqual([path])
    }
  })

  it('escapes the backslashes and quotes fish reads inside single quotes', () => {
    for (const path of [
      "/home/me/repo/a\\'; touch PWNED; #",
      '/home/me/trailing\\',
      '/home/me/two\\\\b',
      "/home/me/it's"
    ]) {
      expect(fishWords(droppedText('fish', path))).toEqual([path])
    }
  })

  it('types each % outside the quotes so cmd expands no variable into the path', () => {
    // A variable whose value holds a quote would turn the rest of the name into a command.
    const env = { HERMESPROBE: 'X"&calc&"Y', PATH: 'C:\\Windows\\system32' }

    for (const path of [
      'C:\\work\\a%HERMESPROBE%b.txt',
      'C:\\work\\a%PATH%b.txt',
      'C:\\work\\my%20file%20name.pdf',
      'C:\\work\\100%'
    ]) {
      expect(cmdWords(droppedText('cmd.exe', path), env)).toEqual([path])
    }
  })

  it('keeps csh and tcsh from expanding history inside the quotes', () => {
    // csh and tcsh expand !-history even inside single quotes; \! is the one escape there.
    for (const shell of ['csh', 'tcsh']) {
      expect(droppedText(shell, '/home/me/x!!:s^e^;touch PWNED;^.txt')).toBe(
        "'/home/me/x\\!\\!:s^e^;touch PWNED;^.txt' "
      )
    }
  })

  it('keeps the POSIX quoting for sh, bash and zsh', () => {
    for (const shell of ['sh', 'bash', 'zsh', '/usr/bin/zsh']) {
      expect(droppedText(shell, "/work/it's \\ a\u2019b!.txt")).toBe("'/work/it'\\''s \\ a\u2019b!.txt' ")
    }
  })
})
