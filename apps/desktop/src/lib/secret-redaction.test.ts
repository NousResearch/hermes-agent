import { describe, expect, it } from 'vitest'

import { redactSensitiveText, redactSensitiveValue } from './secret-redaction'

const FAKE_TOKEN = 'fake_plex_token_for_regression_123456789'

describe('secret redaction', () => {
  it('redacts Plex URL tokens in shell commands', () => {
    const text = `curl 'http://localhost:32400/library/sections?X-Plex-Token=${FAKE_TOKEN}'`
    const redacted = redactSensitiveText(text)

    expect(redacted).not.toContain(FAKE_TOKEN)
    expect(redacted).toContain('X-Plex-Token=[REDACTED]')
  })

  it('redacts env assignments and bearer headers', () => {
    const text = `PLEX_TOKEN='${FAKE_TOKEN}' curl -H "Authorization: Bearer ${FAKE_TOKEN}" http://localhost`
    const redacted = redactSensitiveText(text)

    expect(redacted).not.toContain(FAKE_TOKEN)
    expect(redacted).toContain("PLEX_TOKEN='[REDACTED]'")
    expect(redacted).toContain('Authorization: Bearer [REDACTED]')
  })

  it('redacts nested structured values', () => {
    const redacted = redactSensitiveValue({
      command: `TOKEN=${FAKE_TOKEN} python /tmp/check.py`,
      nested: {
        token: FAKE_TOKEN,
        url: `http://localhost:32400/?X-Plex-Token=${FAKE_TOKEN}`
      }
    })

    expect(JSON.stringify(redacted)).not.toContain(FAKE_TOKEN)
    expect(redacted).toMatchObject({
      command: 'TOKEN=[REDACTED] python /tmp/check.py',
      nested: {
        token: '[REDACTED]',
        url: 'http://localhost:32400/?X-Plex-Token=[REDACTED]'
      }
    })
  })
})
