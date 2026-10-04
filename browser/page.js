/**
 * page.js — runs BEFORE the Hermes webapp bundle.
 *
 * Installs the __HERMES_* globals the upstream browser bridge reads, then
 * replaces window.fetch and window.WebSocket so every same-origin /api/*
 * call is served by the in-page Pyodide backend worker (browser/worker.mjs).
 *
 * Transport: coincident (MIT) — synchronous worker->page proxy calls over
 * SharedArrayBuffer + Atomics. The worker's pump parks inside
 * proxy.pullInbound(ms) until this page hands over buffered inbound
 * frames; worker->page events remain plain postMessage (fire-and-forget
 * is fine in that direction). postMessage alone could never reach the
 * worker while Python blocks — coincident's sync channel carries both
 * directions.
 *
 * Requirements: cross-origin isolation (SharedArrayBuffer/Atomics), i.e.
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 */
(function () {
  'use strict'

  var BASE_PATH = ''
  var origin = window.location.origin

  // --- 1. Hermes webapp bootstrap globals -------------------------------
  window.__HERMES_UI_SURFACE__ = 'webapp'
  window.__HERMES_BASE_PATH__ = BASE_PATH
  window.__HERMES_AUTH_REQUIRED__ = false

  var sessionKey = 'hermes.webapp.session.v1:' + JSON.stringify([origin, BASE_PATH])
  var sessionToken
  try {
    sessionToken = sessionStorage.getItem(sessionKey)
    if (!sessionToken || !/^[A-Za-z0-9_-]{43}$/.test(sessionToken)) {
      var bytes = new Uint8Array(32)
      crypto.getRandomValues(bytes)
      sessionToken = btoa(String.fromCharCode.apply(null, bytes))
        .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '').slice(0, 43)
      sessionStorage.setItem(sessionKey, sessionToken)
    }
  } catch (e) {
    sessionToken = 'x'.repeat(43)
  }

  // --- 2. Inbound frame buffer to the backend worker ---------------------
  // The worker pulls: its pump is a sync proxy.pullInbound(ms) call that
  // parks until frames exist or the timeout elapses, then resolves with
  // the drained array (JSON strings — browser.gateway.handle parses each).
  var inbound = []
  var pullWaiter = null

  function postToWorker(msg) {
    inbound.push(msg)
    if (pullWaiter) {
      var r = pullWaiter
      pullWaiter = null
      r()
    }
  }

  // --- 3. Backend worker --------------------------------------------------
  var bootReady = false
  var bootWaiters = []

  var pendingFetch = {}
  var sockets = {}
  var nextId = 1

  function onWorkerMessage(ev) {
    var msg = ev.data
    if (msg.type === 'fetch-response' && pendingFetch[msg.id]) {
      var p = pendingFetch[msg.id]
      delete pendingFetch[msg.id]
      var bodyBytes = null
      if (msg.bodyB64) {
        var bin = atob(msg.bodyB64)
        bodyBytes = new Uint8Array(bin.length)
        for (var i = 0; i < bin.length; i++) bodyBytes[i] = bin.charCodeAt(i)
      }
      p.resolve(new Response(bodyBytes, { status: msg.status, headers: msg.headers }))
    } else if (msg.type === 'ws-event' && sockets[msg.id]) {
      sockets[msg.id]._onWorkerEvent(msg)
    } else if (msg.type === 'log') {
      (msg.stream === 'err' ? console.warn : console.log)('[backend]', msg.text)
    } else if (msg.type === 'boot-ready') {
      bootReady = true
      window.__HERMES_BACKEND_READY__ = true
      bootWaiters.splice(0).forEach(function (f) { f() })
    } else if (msg.type === 'boot-failed') {
      console.error('[hermes-browser] backend boot failed:', msg.error)
    }
  }

  // Worker-side timeout loops are gone (sync proxy calls return results
  // directly), so page handlers impose the deadlines themselves.
  function withTimeout(p, ms, label) {
    return new Promise(function (resolve) {
      var timer = setTimeout(function () {
        resolve({ error: label + ' timeout' })
      }, ms)
      Promise.resolve(p).then(function (r) {
        clearTimeout(timer)
        resolve(r)
      }, function (e) {
        clearTimeout(timer)
        resolve({ error: String(e).slice(0, 300) })
      })
    })
  }

  // coincident proxy handlers — the worker's synchronous calls land here
  // and the returned promise's resolution is handed back on the parked
  // thread.
  function registerProxyHandlers(w) {
    w.proxy.pullInbound = function (ms) {
      return new Promise(function (resolve) {
        var drain = function () {
          pullWaiter = null
          resolve(inbound.splice(0).map(JSON.stringify))
        }
        if (inbound.length) return drain()
        pullWaiter = drain
        if (ms > 0 && ms < 2147483647) {
          setTimeout(function () {
            if (pullWaiter === drain) drain()
          }, ms)
        }
      })
    }
    w.proxy.fetchRequest = function (url, method, headersJson, bodyB64) {
      return fetchRequestDirect(url, method, headersJson, bodyB64)
    }
  }

  // The backend's Python httpx/urllib3 transports ask the page to perform
  // real fetches (TLS + CORS belong to the browser; the worker has no
  // sockets). The synchronous proxy call resolves with the response.
  function fetchRequestDirect(url, method, headersJson, bodyB64) {
    var headers = {}
    try { headers = JSON.parse(headersJson || '{}') } catch (e) {}
    // Bare-browser fetch is subject to each provider's CORS header
    // allowlist; headers outside it fail preflight. x-stainless-* are
    // OpenAI-SDK build diagnostics with no request semantics; upstream's
    // X-OpenRouter-Cache(-TTL) response-cache hints are absent from
    // openrouter.ai's Access-Control-Allow-Headers (they degrade to
    // uncached responses, the request still succeeds).
    var HOST_HEADER_DENY = {
      'openrouter.ai': ['x-openrouter-cache', 'x-openrouter-cache-ttl'],
    }
    var hostDeny = null
    try {
      hostDeny = HOST_HEADER_DENY[new URL(url).hostname] || null
    } catch (e) {}
    Object.keys(headers).forEach(function (k) {
      var kl = k.toLowerCase()
      if (kl.indexOf('x-stainless-') === 0) delete headers[k]
      else if (hostDeny && hostDeny.indexOf(kl) !== -1) delete headers[k]
    })
    var init = { method: method, headers: headers }
    if (bodyB64) init.body = Uint8Array.from(atob(bodyB64), function (c) { return c.charCodeAt(0) })
    var urlTag = url.split('?')[0]
    console.log('[net] -> ' + method + ' ' + urlTag)
    return withTimeout(realFetch(url, init).then(function (resp) {
      console.log('[net] ' + method + ' ' + urlTag + ' -> ' + resp.status)
      return resp.arrayBuffer().then(function (ab) {
        var bytes = new Uint8Array(ab)
        // Chunked base64: byte-at-a-time concat is O(n^2) and a ~30MB catalog
        // response wedges the page main thread for minutes.
        var bin = ''
        var CH = 0x8000
        for (var i = 0; i < bytes.length; i += CH)
          bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH))
        var hs = {}
        resp.headers.forEach(function (v, k) { hs[k] = v })
        return { status: resp.status, headers: hs, body: btoa(bin) }
      })
    }).catch(function (e) {
      console.log('[net] ' + method + ' ' + urlTag + ' -> ERR ' + String(e).slice(0, 120))
      return { status: 599, headers: {}, body: '', error: String(e) }
    }), 120000, 'fetch').then(function (r) {
      if (r && r.error === 'fetch timeout') {
        return { status: 599, headers: {}, body: '', error: 'fetch timeout' }
      }
      return r
    })
  }

  // --- 4. fetch shim ------------------------------------------------------
  var realFetch = window.fetch
  window.fetch = function (input, init) {
    var url
    try {
      var href = input instanceof Request ? input.url : String(input)
      url = new URL(href, origin)
    } catch (e) {
      return realFetch.apply(this, arguments)
    }
    if (url.origin !== origin || !url.pathname.startsWith('/api/')) {
      return realFetch.apply(this, arguments)
    }
    var method = (init && init.method) || (input && input.method) || 'GET'
    return new Promise(function (resolve, reject) {
      var id = nextId++
      pendingFetch[id] = { resolve: resolve, reject: reject }
      var bodyPromise
      if (init && init.body !== undefined && init.body !== null) {
        if (typeof init.body === 'string') bodyPromise = Promise.resolve({ text: init.body })
        else if (init.body instanceof Blob) {
          bodyPromise = init.body.arrayBuffer().then(function (ab) {
            var b = new Uint8Array(ab), s = ''
            for (var i = 0; i < b.length; i++) s += String.fromCharCode(b[i])
            return { b64: btoa(s), type: init.body.type }
          })
        } else bodyPromise = Promise.resolve({ text: String(init.body) })
      } else if (input instanceof Request && input.method !== 'GET' && input.method !== 'HEAD') {
        bodyPromise = input.arrayBuffer().then(function (ab) {
          var b = new Uint8Array(ab), s = ''
          for (var i = 0; i < b.length; i++) s += String.fromCharCode(b[i])
          return { b64: btoa(s) }
        })
      } else bodyPromise = Promise.resolve(null)
      bodyPromise.then(function (body) {
        // A real browser request carries Host/Origin; the shim synthesizes
        // them so the app's DNS-rebinding middleware sees the true surface.
        // Headers may arrive as a Headers instance, a Request, a pair array,
        // or a plain object — normalize through Headers, then flatten.
        var hdrs = {}
        var merge = function (h) {
          if (!h) return
          new Headers(h).forEach(function (v, k) { hdrs[k] = v })
        }
        merge(input instanceof Request ? input.headers : null)
        merge(init && init.headers)
        if (!hdrs.host) hdrs.host = url.host
        // Ambient private-session auth — the same credential upstream injects
        // into index.html. Callers that already set the header keep theirs.
        if (!hdrs['x-hermes-session-token'] && !hdrs['authorization']) {
          hdrs['x-hermes-session-token'] = sessionToken
        }
        postToWorker({
          t: 'rest', id: id, method: method,
          path: url.pathname + url.search,
          headers: hdrs,
          body: body,
        })
      }).catch(reject)
    })
  }

  // --- 5. WebSocket shim --------------------------------------------------
  var RealWebSocket = window.WebSocket

  function LocalSocket(url, protocols) {
    var u = new URL(url, origin)
    this.url = url
    this.readyState = 0
    this.onopen = null; this.onmessage = null; this.onclose = null; this.onerror = null
    this._id = nextId++
    this._listeners = {}
    this._queue = []
    sockets[this._id] = this
    var wsHeaders = { host: u.host, origin: window.location.origin }
    if (protocols) {
      wsHeaders['sec-websocket-protocol'] =
        Array.isArray(protocols) ? protocols.join(', ') : String(protocols)
    }
    postToWorker({ t: 'ws-open', id: this._id, path: u.pathname + u.search, headers: wsHeaders })
    this._path = u.pathname + u.search
  }
  LocalSocket.CONNECTING = 0; LocalSocket.OPEN = 1
  LocalSocket.CLOSING = 2; LocalSocket.CLOSED = 3
  LocalSocket.prototype = {
    get CONNECTING() { return 0 }, get OPEN() { return 1 },
    get CLOSING() { return 2 }, get CLOSED() { return 3 },
    send: function (d) {
      if (this.readyState === 0) { this._queue.push(d); return }
      if (this.readyState !== 1) { throw new Error('WebSocket is not open') }
      postToWorker({ t: 'ws-send', id: this._id, data: d })
    },
    close: function (code, reason) {
      if (this.readyState >= 2) return
      this.readyState = 3
      postToWorker({ t: 'ws-close', id: this._id, code: code, reason: reason })
      this._fire('close', { code: code || 1000, reason: reason || '', wasClean: true })
    },
    addEventListener: function (t, f) { (this._listeners[t] = this._listeners[t] || []).push(f) },
    removeEventListener: function (t, f) {
      var l = this._listeners[t] || []
      var i = l.indexOf(f); if (i >= 0) l.splice(i, 1)
    },
    _fire: function (t, ev) {
      ev = ev || {}; ev.type = t; if (ev.target === undefined) ev.target = this
      if (typeof this['on' + t] === 'function') this['on' + t](ev)
      var l = this._listeners[t] || []
      for (var i = 0; i < l.length; i++) l[i].call(this, ev)
    },
    _onWorkerEvent: function (msg) {
      if (msg.event === 'open') {
        this.readyState = 1
        this._fire('open')
        for (var i = 0; i < this._queue.length; i++) this.send(this._queue[i])
        this._queue = []
      } else if (msg.event === 'message') {
        this._fire('message', { data: msg.data })
      } else if (msg.event === 'close') {
        this.readyState = 3
        this._fire('close', { code: msg.code || 1000, reason: msg.reason || '', wasClean: true })
      } else if (msg.event === 'error') {
        this._fire('error', {})
      }
    },
  }

  window.WebSocket = function (url, protocols) {
    try {
      var u = new URL(url, origin)
      if (u.host === window.location.host && u.pathname.startsWith('/api/')) {
        return new LocalSocket(url, protocols)
      }
    } catch (e) { /* fall through */ }
    return protocols !== undefined ? new RealWebSocket(url, protocols) : new RealWebSocket(url)
  }
  window.WebSocket.CONNECTING = 0; window.WebSocket.OPEN = 1
  window.WebSocket.CLOSING = 2; window.WebSocket.CLOSED = 3
  window.WebSocket.prototype = RealWebSocket.prototype

  // --- 6. Boot ------------------------------------------------------------
  var interruptSab = new SharedArrayBuffer(8)
  window.__HERMES_INTERRUPT__ = new Int32Array(interruptSab)

  // coincident wraps Worker: worker code calls proxy.<fn> synchronously;
  // the handlers registered above answer them. `native` false means no
  // cross-origin isolation — the backend cannot park, so boot fails loudly
  // instead of hanging.
  import('./vendor/coincident-main.js').then(function (mod) {
    var co = mod.default()
    var w = new co.Worker('./worker.mjs', { type: 'module' })
    registerProxyHandlers(w)
    w.onmessage = onWorkerMessage
    w.postMessage({
      type: 'boot',
      interruptSab: interruptSab,
      pyodideUrl: './pyodide/',
      pyZipUrl: './hermes-py.zip',
      envZipUrl: './hermes-env.zip',
      overlayManifestUrl: './overlay/manifest.json',
      persistHome: true,
      sessionToken: sessionToken,
      publicHost: window.location.hostname,
    })
  }).catch(function (e) {
    console.error('[hermes-browser] transport init failed:', e)
  })

  window.__HERMES_BOOTSTRAP__ = { sessionToken: sessionToken, intercepted: true }
  console.log('[hermes-browser] bootstrap installed')
})()
