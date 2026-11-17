import { execFileSync } from 'node:child_process'
import { createHmac } from 'node:crypto'

import { describe, expect, it } from 'vitest'

import { WEBHOOK_SECRET_PLACEHOLDER, WEBHOOK_TEST_BODY, webhookTestCurl, type WebhookTestSignature } from '../webhook-test-curl'

const SECRET = "s3cr'et with space"
const URL = 'http://localhost:8644/webhooks/deploy'

/** Run the copy-paste command in a real shell with `curl` stubbed to print its argv. */
function runCurl(signature: WebhookTestSignature | null): { body: string; headers: Map<string, string> } {
  const script = `curl() { printf '%s\\n' "$@"; }\n${webhookTestCurl({ secret: SECRET, signature, url: URL })}`
  const argv = execFileSync('bash', ['-c', script], { encoding: 'utf8' }).trimEnd().split('\n')
  const headers = new Map<string, string>()

  argv.forEach((arg, i) => {
    if (argv[i - 1] === '-H') {
      const [name, ...rest] = arg.split(': ')
      headers.set(name.toLowerCase(), rest.join(': '))
    }
  })
  expect(argv.slice(0, 4)).toEqual(['-sS', '-X', 'POST', URL])

  return { body: argv[argv.indexOf('-d') + 1], headers }
}

const hmac = (input: string) => createHmac('sha256', SECRET).update(input).digest('hex')

describe('webhookTestCurl', () => {
  it.each([null, 'generic'] as const)('signs %s routes with timestamp-bound V2', signature => {
    const { body, headers } = runCurl(signature)
    const ts = headers.get('x-webhook-timestamp') ?? ''

    expect(body).toBe(WEBHOOK_TEST_BODY)
    expect(Math.abs(Number(ts) - Date.now() / 1000)).toBeLessThan(60)
    expect(headers.get('x-webhook-signature-v2')).toBe(hmac(`${ts}.${body}`))
  })

  it('signs github routes the way GitHub does', () => {
    const { body, headers } = runCurl('github')

    expect(headers.get('x-hub-signature-256')).toBe(`sha256=${hmac(body)}`)
  })

  it('signs linear and gitlab routes with their own headers', () => {
    expect(runCurl('linear').headers.get('linear-signature')).toBe(hmac(WEBHOOK_TEST_BODY))
    expect(runCurl('gitlab').headers.get('x-gitlab-token')).toBe(SECRET)
  })

  it('gives every run its own delivery id', () => {
    const first = runCurl('generic').headers.get('x-request-id')

    expect(first).toMatch(/^test-\d+-[0-9a-f]{12}$/)
    expect(runCurl('generic').headers.get('x-request-id')).not.toBe(first)
  })

  it('uses a placeholder when the secret is not known', () => {
    expect(webhookTestCurl({ url: URL })).toContain(`SECRET='${WEBHOOK_SECRET_PLACEHOLDER}'`)
  })
})
