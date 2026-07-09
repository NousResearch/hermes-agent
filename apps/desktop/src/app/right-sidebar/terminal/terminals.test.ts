import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.terminals.v1'

async function loadTerminalStore() {
  vi.doMock('@/store/session', () => ({
    $currentCwd: atom('/workspace')
  }))

  return import('./terminals')
}

// Helper to test cleanReviveSnapshot logic (from use-terminal-session.ts)
function stripEscapeSequences(data: string): string {
  let index = 0
  let text = ''

  while (index < data.length) {
    if (data.charCodeAt(index) === 0x1b) {
      const sequence = readEscapeSequence(data, index)

      if (sequence) {
        index += sequence.length
        continue
      }
    }

    text += data[index]
    index += 1
  }

  return text
}

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

  // Character-set and other short ESC forms are three bytes (e.g. ESC ( B).
  if (['(', ')', '*', '+', '-', '.', '/'].includes(kind) && index + 2 < data.length) {
    return data.slice(index, index + 3)
  }

  return data.slice(index, Math.min(index + 2, data.length))
}

function cleanReviveSnapshot(serialized: string): string {
  const visible = (line: string) => stripEscapeSequences(line).replace(/[\s%]/g, '')
  const lines = serialized.split(/\r?\n/)

  while (lines.length && visible(lines[lines.length - 1]) === '') {
    lines.pop()
  }

  // If the buffer contains only a few visible lines (e.g., just a prompt without
  // any command output), treat it as an idle session and clear it entirely.
  // This prevents duplicate prompts from accumulating on every app relaunch.
  // The threshold of 3 accounts for single-command sessions that include the
  // prompt, command, and output.
  if (lines.length <= 3) {
    return ''
  }

  let lastBlank = -1

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (visible(lines[i]) === '') {
      lastBlank = i

      break
    }
  }

  // A prompt is a short block; a long tail after the blank is real output, leave it.
  if (lastBlank >= 0 && lines.length - 1 - lastBlank <= 3) {
    lines.length = lastBlank
  }

  return lines.join('\r\n')
}

describe('cleanReviveSnapshot', () => {
  it('clears idle buffer with just a prompt', () => {
    const idleBuffer = 'PS C:\\Users\\Aleksandr>'
    expect(cleanReviveSnapshot(idleBuffer)).toBe('')
  })

  it('clears buffer with only prompt and command output (≤3 lines)', () => {
    const cmdBuffer = 'PS C:\\Users\\Aleksandr> echo test\r\ntest\r\nPS C:\\Users\\Aleksandr>'
    expect(cleanReviveSnapshot(cmdBuffer)).toBe('')
  })

  it('preserves longer buffers (>3 lines)', () => {
    const longBuffer = 'PS C:\\Users\\Aleksandr> echo test\r\ntest\r\nPS C:\\Users\\Aleksandr> echo foo\r\nfoo\r\nPS C:\\Users\\Aleksandr>'
    const result = cleanReviveSnapshot(longBuffer)
    expect(result).toContain('echo test')
    expect(result).toContain('echo foo')
  })

  it('strips trailing blank lines before checking line count', () => {
    const bufferWithBlanks = 'PS C:\\Users\\Aleksandr>\r\n\r\n\r\n'
    expect(cleanReviveSnapshot(bufferWithBlanks)).toBe('')
  })

  it('handles empty buffer', () => {
    expect(cleanReviveSnapshot('')).toBe('')
  })
})

describe('terminal store persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores user tabs, active tab, and history on module load', async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        activeTerminalId: 'term-two',
        terminals: [
          { auto: false, cwd: '/repo/one', id: 'term-one', reviveBuffer: 'last output', title: 'zsh' },
          { auto: true, cwd: '/repo/two', id: 'term-two', title: 'Terminal' }
        ]
      })
    )

    const { $activeTerminalId, $terminals } = await loadTerminalStore()

    expect($activeTerminalId.get()).toBe('term-two')
    expect($terminals.get()).toEqual([
      { auto: false, cwd: '/repo/one', id: 'term-one', kind: 'user', reviveBuffer: 'last output', title: 'zsh' },
      { auto: true, cwd: '/repo/two', id: 'term-two', kind: 'user', title: 'Terminal' }
    ])
  })

  it('persists user tabs and history synchronously, skipping agent mirrors', async () => {
    const { createTerminal, ensureAgentTerminal, renameTerminal, selectTerminal, updateTerminalReviveBuffer } =
      await loadTerminalStore()

    const userId = createTerminal('/repo')
    renameTerminal(userId, 'server')
    updateTerminalReviveBuffer(userId, 'recent scrollback')
    ensureAgentTerminal('proc-1', 'background task')
    selectTerminal(userId)

    // No flush/tick: persistence is synchronous, so the snapshot is already on
    // disk (this is what makes app-quit restore reliable).
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')).toEqual({
      activeTerminalId: userId,
      terminals: [{ auto: false, cwd: '/repo', id: userId, reviveBuffer: 'recent scrollback', title: 'server' }]
    })
  })

  it('never attaches a revive buffer to an agent tab', async () => {
    const { $terminals, ensureAgentTerminal, updateTerminalReviveBuffer } = await loadTerminalStore()

    const agentId = ensureAgentTerminal('proc-1', 'background task')!
    updateTerminalReviveBuffer(agentId, 'should be ignored')

    expect($terminals.get().find(term => term.id === agentId)?.reviveBuffer).toBeUndefined()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('tail-trims an oversized revive buffer to stay under the storage budget', async () => {
    const { $terminals, createTerminal, updateTerminalReviveBuffer } = await loadTerminalStore()

    const userId = createTerminal('/repo')
    const huge = 'x'.repeat(60_000)
    updateTerminalReviveBuffer(userId, huge)

    const stored = $terminals.get().find(term => term.id === userId)?.reviveBuffer ?? ''
    expect(stored.length).toBe(48_000)
    expect(stored).toBe(huge.slice(-48_000))
  })

  it('clears remembered tabs when all terminals close', async () => {
    const { closeAllTerminals, createTerminal } = await loadTerminalStore()

    createTerminal('/repo')
    expect(window.localStorage.getItem(STORAGE_KEY)).not.toBeNull()

    closeAllTerminals()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})
