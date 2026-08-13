#!/usr/bin/env node
/**
 * Launch the desktop app with a mock inference provider — no real API
 * keys needed. Starts a local OpenAI-compatible server that returns a
 * canned reply, writes an isolated config.yaml + .env, and launches the
 * built Electron app against them.
 *
 * This reuses the same mock-server and config format as the E2E fixtures
 * (apps/desktop/e2e/mock-server.ts + fixtures.ts), so local dev and CI
 * test the same chain.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 *
 * Usage:
 *   node scripts/dev-mock.mjs
 *   npm run dev:mock
 *
 * The mock server listens on an ephemeral port and replies to every
 * chat completion with:
 *   "Hello from the mock inference server! The full boot chain is working."
 */

import http from 'node:http'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawn, spawnSync } from 'node:child_process'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

// ── Canned reply ───────────────────────────────────────────────────────

const CANNED_REPLY =
  'Hello from the mock inference server! The full boot chain is working.'

// ── Mock server (mirrors e2e/mock-server.ts) ───────────────────────────

function startMockServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

      if (req.method === 'OPTIONS') {
        res.writeHead(204)
        res.end()
        return
      }

      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            object: 'list',
            data: [{ id: 'mock-model', object: 'model', created: 0, owned_by: 'mock' }],
          }),
        )
        return
      }

      if (req.method === 'POST' && req.url?.startsWith('/v1/chat/completions')) {
        let body = ''
        req.on('data', (chunk) => { body += chunk.toString() })
        req.on('end', () => {
          let parsed = {}
          try { parsed = JSON.parse(body) } catch { /* non-streaming */ }

          const stream = parsed.stream === true
          const model = parsed.model || 'mock-model'

          if (stream) {
            res.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
            })
            const words = CANNED_REPLY.split(' ')
            let i = 0
            const sendChunk = () => {
              if (i >= words.length) {
                res.write(
                  `data: ${JSON.stringify({
                    id: 'mock-completion', object: 'chat.completion.chunk',
                    created: 0, model,
                    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                  })}\n\n`,
                )
                res.write('data: [DONE]\n\n')
                res.end()
                return
              }
              const word = i === 0 ? words[i] : ' ' + words[i]
              res.write(
                `data: ${JSON.stringify({
                  id: 'mock-completion', object: 'chat.completion.chunk',
                  created: 0, model,
                  choices: [{ index: 0, delta: { content: word }, finish_reason: null }],
                })}\n\n`,
              )
              i++
              setTimeout(sendChunk, 20)
            }
            sendChunk()
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' })
            res.end(
              JSON.stringify({
                id: 'mock-completion', object: 'chat.completion',
                created: 0, model,
                choices: [{
                  index: 0,
                  message: { role: 'assistant', content: CANNED_REPLY },
                  finish_reason: 'stop',
                }],
                usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
              }),
            )
          }
        })
        req.on('error', () => { res.writeHead(400); res.end('Bad request') })
        return
      }

      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Not found' }))
    })

    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (addr === null || typeof addr === 'string') {
        reject(new Error('Failed to get server address'))
        return
      }
      resolve({ port: addr.port, url: `http://127.0.0.1:${addr.port}`, close: () => server.close() })
    })
  })
}

// ── Config + env writing (mirrors e2e/fixtures.ts) ─────────────────────

function createSandbox() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-dev-mock-${Date.now()}`))
  const hermesHome = path.join(root, 'hermes-home')
  const userDataDir = path.join(root, 'electron-user-data')
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.mkdirSync(userDataDir, { recursive: true })
  return { root, hermesHome, userDataDir, cleanup: () => fs.rmSync(root, { recursive: true, force: true }) }
}

function writeMockConfig(hermesHome, mockUrl) {
  fs.writeFileSync(
    path.join(hermesHome, 'config.yaml'),
    `# Auto-generated by dev-mock.mjs
model:
  default: mock-model
  provider: mock
providers:
  mock:
    api: ${mockUrl}/v1
    name: Mock
    api_mode: chat_completions
    key_env: MOCK_API_KEY
    models:
      mock-model: {}
    context_length: 4096
`,
    'utf8',
  )
  fs.writeFileSync(path.join(hermesHome, '.env'), 'MOCK_API_KEY=e2e-mock-key\n', 'utf8')
}

// ── Electron launch ────────────────────────────────────────────────────

function findElectron() {
  const local = path.join(REPO_ROOT, 'node_modules', 'electron', 'dist', 'electron')
  if (fs.existsSync(local)) return local
  const r = spawnSync('which', ['electron'], { encoding: 'utf8' })
  if (r.status === 0 && r.stdout.trim()) return r.stdout.trim()
  throw new Error('Electron binary not found. Run "npm install" from the repo root.')
}

function assertDistBuilt() {
  const electronMain = path.join(DESKTOP_ROOT, 'dist', 'electron-main.mjs')
  const indexHtml = path.join(DESKTOP_ROOT, 'dist', 'index.html')
  if (!fs.existsSync(electronMain) || !fs.existsSync(indexHtml)) {
    throw new Error(
      `Desktop dist not built. Run 'cd apps/desktop && npm run build' first.\n` +
      `Missing: ${electronMain}`,
    )
  }
}

// ── Main ───────────────────────────────────────────────────────────────

async function main() {
  assertDistBuilt()

  console.log('Starting mock inference server...')
  const mock = await startMockServer()
  console.log(`  Mock server: ${mock.url}`)

  const sandbox = createSandbox()
  writeMockConfig(sandbox.hermesHome, mock.url)
  console.log(`  HERMES_HOME: ${sandbox.hermesHome}`)

  const electronBin = findElectron()

  const env = {
    ...process.env,
    HERMES_HOME: sandbox.hermesHome,
    HERMES_DESKTOP_USER_DATA_DIR: sandbox.userDataDir,
    HERMES_DESKTOP_IGNORE_EXISTING: '1',
    HERMES_DESKTOP_HERMES_ROOT: REPO_ROOT,
    HERMES_DESKTOP_APP_NAME: `HermesDevMock-${Date.now()}`,
  }

  console.log('Launching Electron...')
  const child = spawn(electronBin, [DESKTOP_ROOT, '--disable-gpu', '--no-sandbox'], {
    env,
    cwd: DESKTOP_ROOT,
    stdio: 'inherit',
  })

  child.on('exit', (code) => {
    mock.close()
    sandbox.cleanup()
    process.exit(code ?? 0)
  })
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
