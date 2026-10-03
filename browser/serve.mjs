#!/usr/bin/env node
// Dev server for the built site dir with the COOP/COEP headers a browser
// runtime host must send (SharedArrayBuffer requires cross-origin isolation).
// Usage: node browser/serve.mjs [siteDir] [port]
import { createReadStream, existsSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { extname, join } from 'node:path'

const ROOT = process.argv[2] || 'dist'
const PORT = Number(process.argv[3] || 8471)

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.zip': 'application/zip', '.wasm': 'application/wasm',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.css': 'text/css', '.ico': 'image/x-icon',
  '.py': 'text/x-python', '.webmanifest': 'application/manifest+json',
}

// /mock-llm — OpenAI-compatible dev endpoint for smoke tests. Returns a
// canned completion echoing the last user message.
function mockLlm(req, res) {
  // The provider's endpoint-discovery sweep hits several API shapes
  // (OpenAI /v1/* and Ollama /api/*) — answer the ones it probes and
  // allow cross-origin since localhost is normalized to 127.0.0.1.
  if (req.url.startsWith('/mock-llm/')) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, x-stainless-*')
    if (req.method === 'OPTIONS') { res.end(); return true }
  }
  const openaiModels = () => JSON.stringify({
    object: 'list', data: [{ id: 'mock-1', object: 'model', created: 0, owned_by: 'mock' }],
  })
  const path = req.url.split('?')[0]
  if (req.method === 'GET' && /\/models(\/[\w.-]+)?$/.test(path)) {
    res.setHeader('Content-Type', 'application/json')
    res.end(path.endsWith('/mock-1')
      ? JSON.stringify({ id: 'mock-1', object: 'model', created: 0, owned_by: 'mock' })
      : openaiModels())
    return true
  }
  // Ollama-flavored probes.
  if (req.method === 'GET' && path.endsWith('/api/tags')) {
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ models: [{ name: 'mock-1', model: 'mock-1' }] }))
    return true
  }
  if ((req.method === 'POST' && path.endsWith('/api/show')) ||
      (req.method === 'GET' && (path.endsWith('/version') || path.endsWith('/props')))) {
    res.setHeader('Content-Type', 'application/json')
    res.end(path.endsWith('/show')
      ? JSON.stringify({ model_info: { 'llama.context_length': 8192 } })
      : JSON.stringify({ version: 'mock' }))
    return true
  }
  if (req.url === '/mock-llm/v1/chat/completions' && req.method === 'POST') {
    let body = ''
    req.on('data', (c) => body += c)
    req.on('end', () => {
      let echo = ''
      try {
        const msgs = JSON.parse(body).messages || []
        echo = (msgs[msgs.length - 1] && msgs[msgs.length - 1].content) || ''
        if (typeof echo !== 'string') echo = JSON.stringify(echo)
      } catch (e) { /* ignore */ }
      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({
        id: 'chatcmpl-mock', object: 'chat.completion', created: Math.floor(Date.now() / 1000),
        model: 'mock-1',
        choices: [{ index: 0, message: { role: 'assistant', content: 'MOCK-LLM-REPLY: ' + echo.slice(0, 200) }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }))
    })
    return true
  }
  return false
}

createServer((req, res) => {
  if (mockLlm(req, res)) return
  let p = join(ROOT, decodeURIComponent(req.url.split('?')[0]))
  if (p.endsWith('/')) p += 'index.html'
  if (!existsSync(p) || statSync(p).isDirectory()) p = join(ROOT, 'index.html')
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin')
  res.setHeader('Content-Type', MIME[extname(p)] || 'application/octet-stream')
  createReadStream(p).pipe(res)
}).listen(PORT, () => console.log(`serving ${ROOT} on :${PORT}`))
