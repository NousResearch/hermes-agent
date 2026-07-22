/**
 * Minimal OpenAI-compatible mock inference server for E2E tests.
 *
 * Implements just enough of the /v1/* surface for `hermes serve` to resolve a
 * provider, list models, and stream a canned chat completion back to the
 * desktop app — without any real LLM.
 *
 * Endpoints:
 *   GET  /v1/models             → { data: [{ id, ... }] }
 *   POST /v1/chat/completions   → streaming (SSE) or non-streaming response
 *
 * The canned response is a short, deterministic assistant message. Tool-call
 * requests are not simulated — the E2E tests only need the chat surface to
 * prove the full boot → gateway → inference → renderer chain works.
 */

import http from 'node:http'

/** A canned assistant reply used for every chat completion request. */
const CANNED_REPLY = 'Hello from the mock inference server! The full boot chain is working.'

/**
 * Start the mock server on an ephemeral port.
 *
 * @returns a handle with `port`, `url`, and `close()`.
 */
export function startMockServer(): Promise<{ port: number; url: string; close: () => Promise<void> }> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      // CORS headers — the Electron renderer doesn't need them, but they
      // don't hurt and make the server usable from a browser context too.
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

      if (req.method === 'OPTIONS') {
        res.writeHead(204)
        res.end()
        return
      }

      // GET /v1/models — return a single fake model.
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            object: 'list',
            data: [
              {
                id: 'mock-model',
                object: 'model',
                created: 0,
                owned_by: 'mock',
              },
            ],
          }),
        )
        return
      }

      // POST /v1/chat/completions — return a canned response.
      if (req.method === 'POST' && req.url?.startsWith('/v1/chat/completions')) {
        let body = ''

        req.on('data', (chunk: Buffer) => {
          body += chunk.toString()
        })

        req.on('end', () => {
          let parsed: any = {}

          try {
            parsed = JSON.parse(body)
          } catch {
            // malformed JSON — treat as non-streaming with defaults
          }

          const stream = parsed.stream === true
          const model = parsed.model || 'mock-model'

          if (stream) {
            res.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
            })

            // Send the content in a few chunks to simulate streaming.
            const words = CANNED_REPLY.split(' ')
            let i = 0

            const sendChunk = () => {
              if (i >= words.length) {
                // Final chunk with finish_reason
                res.write(
                  `data: ${JSON.stringify({
                    id: 'mock-completion',
                    object: 'chat.completion.chunk',
                    created: 0,
                    model,
                    choices: [
                      {
                        index: 0,
                        delta: {},
                        finish_reason: 'stop',
                      },
                    ],
                  })}\n\n`,
                )
                res.write('data: [DONE]\n\n')
                res.end()
                return
              }

              const word = i === 0 ? words[i] : ' ' + words[i]
              res.write(
                `data: ${JSON.stringify({
                  id: 'mock-completion',
                  object: 'chat.completion.chunk',
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: { content: word },
                      finish_reason: null,
                    },
                  ],
                })}\n\n`,
              )
              i++
              // Small delay between chunks to simulate real streaming.
              setTimeout(sendChunk, 20)
            }

            sendChunk()
          } else {
            // Non-streaming response
            res.writeHead(200, { 'Content-Type': 'application/json' })
            res.end(
              JSON.stringify({
                id: 'mock-completion',
                object: 'chat.completion',
                created: 0,
                model,
                choices: [
                  {
                    index: 0,
                    message: { role: 'assistant', content: CANNED_REPLY },
                    finish_reason: 'stop',
                  },
                ],
                usage: {
                  prompt_tokens: 10,
                  completion_tokens: 20,
                  total_tokens: 30,
                },
              }),
            )
          }
        })

        req.on('error', () => {
          res.writeHead(400)
          res.end('Bad request')
        })
        return
      }

      // Fallback — 404 for anything else
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

      const port = addr.port
      const url = `http://127.0.0.1:${port}`

      resolve({
        port,
        url,
        close: () =>
          new Promise((resolveClose, rejectClose) => {
            server.close((err) => {
              if (err) {
                rejectClose(err)
              } else {
                resolveClose()
              }
            })
          }),
      })
    })
  })
}
