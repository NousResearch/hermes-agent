/**
 * Helpers for Electron net.request calls that ride the OAuth session partition.
 *
 * Electron's ClientRequest forbids app-set restricted headers such as
 * Content-Length. Let Chromium frame the body itself, and only set an entity
 * Content-Type when a request actually has a body.
 */

function serializeJsonBody(body) {
  return body === undefined ? undefined : Buffer.from(JSON.stringify(body))
}

function setRequestContentType(request, contentType = 'application/json') {
  request.setHeader('Content-Type', contentType)
}

function parseJsonResponse(url, statusCode, headers, text) {
  if (statusCode >= 400) {
    const err = new Error(`${statusCode}: ${text || ''}`) as any
    err.statusCode = statusCode

    throw err
  }

  if (!text) {
    return null
  }

  const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
  const contentType = String(headers['content-type'] || headers['Content-Type'] || '')

  if (looksHtml || contentType.includes('text/html')) {
    throw new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`)
  }

  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`)
  }
}

function fetchJsonViaOauthSession(url, options: any = {}, deps: any = {}) {
  return new Promise((resolve, reject) => {
    const sess = deps.session

    if (!sess) {
      reject(new Error('OAuth session partition is unavailable.'))

      return
    }

    let parsed

    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`))

      return
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

      return
    }

    const multipart = options.upload ? deps.multipartBody(options.upload) : null
    const body = multipart ? multipart.body : serializeJsonBody(options.body)
    const contentType = multipart ? multipart.contentType : body ? 'application/json' : null
    const timeoutMs = deps.resolveTimeoutMs(options.timeoutMs, deps.defaultTimeoutMs)

    const request = deps.net.request({
      method: options.method || 'GET',
      url,
      session: sess,
      useSessionCookies: true,
      redirect: 'follow'
    } as any)

    if (contentType) {
      setRequestContentType(request, contentType)
    }

    let settled = false

    const settle = (fn, value) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      fn(value)
    }

    const timer = setTimeout(() => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      const timeoutError = new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`)

      try {
        request.abort()
      } catch {
        // already finished
      }

      reject(timeoutError)
    }, timeoutMs)

    request.on('response', res => {
      const chunks = []

      res.on('data', chunk => chunks.push(Buffer.from(chunk)))
      res.on('error', error => settle(reject, error))
      res.on('end', () => {
        if (settled) {
          return
        }

        const text = Buffer.concat(chunks).toString('utf8')
        const statusCode = res.statusCode || 500

        try {
          settle(resolve, parseJsonResponse(url, statusCode, res.headers || {}, text))
        } catch (error) {
          settle(reject, error)
        }
      })
    })
    request.on('error', error => settle(reject, error))

    if (body) {
      request.write(body)
    }

    request.end()
  })
}

export { fetchJsonViaOauthSession, serializeJsonBody, setRequestContentType }
