/**
 * worker.mjs — the real Hermes backend under Pyodide.
 *
 * Inbound frames arrive on a SharedArrayBuffer ring written by the page
 * (postMessage can't reach this worker while Python is blocked in
 * Atomics.wait). Outbound frames go via postMessage — fire-and-forget is
 * fine in this direction.
 *
 * Ring layout (Int32 header over SAB):
 *   [0] seq doorbell — incremented by the writer, Atomics.wait target
 *   [1] write offset (bytes into data region, monotonic mod CAP)
 *   [2] read offset
 *   data region: CAP bytes of u32-length-prefixed UTF-8 frames, wrap-around
 *
 * Single producer (page) / single consumer (this worker).
 */
import { loadPyodide } from './pyodide/pyodide.mjs'

const RING_CAP = 4 * 1024 * 1024

let hdr        // Int32Array view over sab (header)
let data       // Uint8Array view over sab (data region)
let lastSeq = 0

function initRing(sab) {
  hdr = new Int32Array(sab, 0, 4)
  data = new Uint8Array(sab, 16)
}

const FRAG_MORE = 0x80000000
let fragBuf = null   // accumulated fragments of one logical frame

function ringDrain() {
  // Returns all complete frames currently in the ring. Records whose
  // length word carries FRAG_MORE are fragments — the single-producer
  // page writes them contiguously; a frame completes on the first
  // record without the flag.
  const out = []
  const w = Atomics.load(hdr, 1)
  let r = Atomics.load(hdr, 2)
  while (r !== w) {
    // The 4-byte length word wraps the ring end like any payload bytes.
    const h = data[(r) % RING_CAP] | (data[(r + 1) % RING_CAP] << 8) |
      (data[(r + 2) % RING_CAP] << 16) | (data[(r + 3) % RING_CAP] << 24)
    const raw = h >>> 0
    const more = (raw & FRAG_MORE) !== 0
    const len = raw & ~FRAG_MORE
    r = (r + 4) % RING_CAP
    const bytes = new Uint8Array(len)
    for (let i = 0; i < len; i++) bytes[i] = data[(r + i) % RING_CAP]
    r = (r + len) % RING_CAP
    if (more) {
      if (fragBuf) fragBuf.push(bytes); else fragBuf = [bytes]
      continue
    }
    let frame = bytes
    if (fragBuf) {
      fragBuf.push(bytes)
      const total = fragBuf.reduce((n, c) => n + c.length, 0)
      frame = new Uint8Array(total)
      let off = 0
      for (const c of fragBuf) { frame.set(c, off); off += c.length }
      fragBuf = null
    }
    out.push(new TextDecoder().decode(frame))
  }
  Atomics.store(hdr, 2, r)
  return out
}

const bridge = {
  // Called from inside Python while it blocks on a primitive. Atomics.wait
  // sleeps until the page bumps the doorbell; then we drain the ring and
  // return raw frame strings for Python to route.
  pump(ms) {
    // Doorbell-driven, but bounded: any wakeup path Python missed (asyncio
    // internals queuing work outside the ring) costs a poll cycle, never a
    // permanent stall. 500ms is invisible to users and idle-cheap.
    const seq = Atomics.load(hdr, 0)
    Atomics.wait(hdr, 0, seq, ms < 0 ? 500 : Math.min(ms, 500))
    lastSeq = Atomics.load(hdr, 0)
    return ringDrain()
  },
  emit(wsId, text) {
    self.postMessage({ type: 'ws-event', id: wsId, event: 'message', data: text })
  },
  wsAccepted(wsId) {
    self.postMessage({ type: 'ws-event', id: wsId, event: 'open' })
  },
  wsClosed(wsId, code, reason) {
    self.postMessage({ type: 'ws-event', id: wsId, event: 'close', code, reason })
  },
  restReply(id, status, headersJson, bodyB64) {
    // headersJson crosses the Python boundary as a string (PyProxy dicts
    // can't survive structured clone).
    self.postMessage({
      type: 'fetch-response', id, status,
      headers: JSON.parse(headersJson || '{}'),
      bodyB64,
    })
  },
  fetchRequest(id, url, method, headersJson, bodyB64) {
    self.postMessage({
      type: 'net-request', id, url, method,
      headers: JSON.parse(headersJson), bodyB64,
    })
  },
  log(level, msg) { console.log('[py]', msg) },
}

self.onmessage = async (ev) => {
  const msg = ev.data
  if (msg.type !== 'boot') return
  initRing(msg.sab)
  try {
    await main(msg)
  } catch (e) {
    self.postMessage({ type: 'boot-failed', error: String(e && e.message || e) })
    throw e
  }
}

async function main(msg) {
  const pyodide = await loadPyodide({ indexURL: msg.pyodideUrl })
  // Debug aid: page can write 2 to interruptSab[0] to SIGINT a wedged boot.
  if (msg.interruptSab) pyodide.setInterruptBuffer(new Uint8Array(msg.interruptSab))
  self.globalThis.hermesBridge = bridge
  pyodide.setStdout({ batched: (s) => self.postMessage({ type: 'log', stream: 'out', text: s }) })
  pyodide.setStderr({ batched: (s) => self.postMessage({ type: 'log', stream: 'err', text: s }) })
  pyodide.setStdin({ stdin: () => null })

  // Pyodide stdlib packages shipped with the distribution.
  await pyodide.loadPackage(['sqlite3', 'ssl'])

  // Preinstalled site-packages env (built by scripts/pack-env.mjs at
  // build time — no runtime PyPI dependency).
  const envZip = await (await fetch(msg.envZipUrl)).arrayBuffer()
  pyodide.FS.mkdirTree('/hermes-env')
  pyodide.unpackArchive(new Uint8Array(envZip), 'zip', { extractDir: '/hermes-env' })

  // Unpack the vendored upstream python tree.
  const zip = await (await fetch(msg.pyZipUrl)).arrayBuffer()
  pyodide.FS.mkdirTree('/hermes-py')
  pyodide.unpackArchive(new Uint8Array(zip), 'zip', { extractDir: '/hermes-py' })

  // Our python overlay modules (manifest-driven so assemble.mjs stays the
  // single source of file names).
  const overlay = await (await fetch(msg.overlayManifestUrl)).json()
  for (const f of overlay) {
    const bytes = await (await fetch(f.url)).arrayBuffer()
    const target = `/hermes-py/${f.name}`
    const parent = target.slice(0, target.lastIndexOf('/'))
    pyodide.FS.mkdirTree(parent)
    pyodide.FS.writeFile(target, new Uint8Array(bytes))
  }

  // Optional OPFS persistence for ~/.hermes (cross-origin isolated only).
  if (msg.persistHome && pyodide.mountNativeFS) {
    try {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('hermes', { create: true })
      pyodide.FS.mkdirTree('/hermes-home')
      await pyodide.mountNativeFS('/hermes-home', dir)
    } catch (e) {
      self.postMessage({ type: 'log', stream: 'err', text: `opfs mount failed: ${e.message}` })
    }
  }

  await pyodide.runPythonAsync(`
import sys
# hermes-env last: the zip contains pure-python stdlib shims (ssl.py etc.)
# captured from site-packages; they must NOT shadow the wasm stdlib.
sys.path.append('/hermes-env')
sys.path.insert(0, '/hermes-py')
import browser.gateway
browser.gateway.boot(${JSON.stringify(String(msg.sessionToken || ''))}, ${JSON.stringify(String(msg.publicHost || ''))})
`)

  const gw = pyodide.pyimport('browser.gateway')
  const runtime = pyodide.pyimport('browser.runtime')
  self.postMessage({ type: 'boot-ready' })

  // Main loop: block on the ring until frames arrive, route each into Python,
  // then drain Python-side work (loop tasks, thread reschedules) to quiescence.
  // Python calls that block (approvals, page fetches) re-enter bridge.pump
  // internally, so this loop only advances when the interpreter is idle.
  // bridge.pump is bounded at 500ms, so due timers fire even with no traffic.
  while (true) {
    const frames = bridge.pump(-1)
    for (const raw of frames) {
      try { gw.handle(raw) } catch (e) {
        self.postMessage({ type: 'log', stream: 'err', text: `handle: ${e.message}` })
      }
    }
    try {
      let pending = runtime.pump_once(0)
      let guard = 0
      while (pending > 0 && guard++ < 1000) pending = runtime.pump_once(0)
    } catch (e) {
      self.postMessage({ type: 'log', stream: 'err', text: `pump: ${e.message}` })
    }
  }
}
