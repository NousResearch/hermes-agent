import type { HermesTerminalExit } from '../src/global'

import { browserMessage } from './messages'
import { gatewayWsUrl } from './transport'

interface Terminal {
  socket: WebSocket
  cwd: string
  buffer: string
  onData: Set<(data: string) => void>
  onExit: Set<(exit: HermesTerminalExit) => void>
}

const terminals = new Map<string, Terminal>()

export const terminal: Window['hermesDesktop']['terminal'] = {
  async start(options = {}) {
    const auth = await gatewayWsUrl()

    if (!auth.ok) {throw new Error(auth.error)}
    const url = new URL(auth.wsUrl)
    url.pathname = url.pathname.replace(/\/api\/ws$/, '/api/browser/terminal')

    if (options.cwd) {url.searchParams.set('cwd', options.cwd)}

    if (options.cols) {url.searchParams.set('cols', String(options.cols))}

    if (options.rows) {url.searchParams.set('rows', String(options.rows))}
    const socket = new WebSocket(url)
    const id = crypto.randomUUID()
    const state: Terminal = { socket, cwd: options.cwd || '', buffer: '', onData: new Set(), onExit: new Set() }

    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => { socket.close(); reject(new Error(browserMessage('terminalTimeout'))) }, 15_000)

      socket.onmessage = (event) => {
        const message: { type: string; cwd?: string; shell?: string; data?: string; error?: string } = JSON.parse(String(event.data))

        if (message.type === 'ready') {
          window.clearTimeout(timer)
          state.cwd = message.cwd || ''
          terminals.set(id, state)
          resolve({ id, cwd: state.cwd, shell: message.shell || '' })
        } else if (message.type === 'data' && message.data) {
          if (state.onData.size) {for (const listener of state.onData) {listener(message.data)}}
          else {state.buffer = (state.buffer + message.data).slice(-65536)}
        } else if (message.type === 'error') {
          window.clearTimeout(timer)
          socket.close()
          reject(new Error(message.error || browserMessage('terminalFailed')))
        }
      }

      socket.onerror = () => { window.clearTimeout(timer); reject(new Error(browserMessage('terminalFailed'))) }

      socket.onclose = () => {
        window.clearTimeout(timer)
        terminals.delete(id)

        for (const listener of state.onExit) {listener({ code: null, signal: null })}
        reject(new Error(browserMessage('terminalClosed')))
      }
    })
  },
  async attach(id) { return terminals.has(id) },
  async cwd(id) { return terminals.get(id)?.cwd || null },
  async dispose(id) { const entry = terminals.get(id); entry?.socket.close(); terminals.delete(id);

 return Boolean(entry) },
  async write(id, data) {
    const entry = terminals.get(id)

    if (entry?.socket.readyState !== WebSocket.OPEN) {return false}
    entry.socket.send(JSON.stringify({ type: 'write', data }))

    return true
  },
  async resize(id, { cols, rows }) {
    const entry = terminals.get(id)

    if (entry?.socket.readyState !== WebSocket.OPEN) {return false}
    entry.socket.send(JSON.stringify({ type: 'resize', cols, rows }))

    return true
  },
  onData(id, callback) {
    const entry = terminals.get(id)

    if (!entry) {return () => {}}
    entry.onData.add(callback)

    if (entry.buffer) { callback(entry.buffer); entry.buffer = '' }

    return () => { entry.onData.delete(callback) }
  },
  onExit(id, callback) {
    const entry = terminals.get(id)
    entry?.onExit.add(callback)

    return () => { entry?.onExit.delete(callback) }
  }
}
