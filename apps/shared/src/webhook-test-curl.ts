/** A copy-paste `curl` that sends one signed test event to a webhook route.
 *
 * The shell computes the signature when the command runs (openssl), so a timestamped V2 signature is
 * never stale and the secret is only ever in the user's own terminal. Without the secret (it is shown
 * once, at creation) the command carries a placeholder to replace. Each run sends a fresh
 * `X-Request-ID`, which the receiver records as the delivery id, so a re-run is a new delivery rather
 * than a duplicate. Mirrors `hermes webhook test` (hermes_cli/webhook.py::_test_signature_headers). */

export type WebhookTestSignature = 'generic' | 'github' | 'gitlab' | 'linear'

export interface WebhookTestCurlInput {
  /** Omitted: a placeholder the user replaces with the secret shown at creation. */
  secret?: null | string
  /** The route's pin; null (unpinned) signs like generic. */
  signature?: null | WebhookTestSignature
  url: string
}

export const WEBHOOK_TEST_BODY = '{"event_type":"test","text":"hello"}'
export const WEBHOOK_SECRET_PLACEHOLDER = 'paste-the-secret-shown-at-creation'

const MAC = (input: string) => `SIG=$(printf '%s' ${input} | openssl dgst -sha256 -hmac "$SECRET" | sed 's/^.* //')`

const SIGNING: Record<WebhookTestSignature, { headers: string[]; mac?: string }> = {
  generic: {
    mac: MAC('"$TS.$BODY"'),
    headers: ['X-Webhook-Timestamp: $TS', 'X-Webhook-Signature-V2: $SIG']
  },
  github: { mac: MAC('"$BODY"'), headers: ['X-GitHub-Event: test', 'X-Hub-Signature-256: sha256=$SIG'] },
  gitlab: { headers: ['X-Gitlab-Event: test', 'X-Gitlab-Token: $SECRET'] },
  linear: { mac: MAC('"$BODY"'), headers: ['Linear-Signature: $SIG'] }
}

/** POSIX single-quoting: safe for any secret or URL a user pasted. */
export function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`
}

export function webhookTestCurl({ secret, signature, url }: WebhookTestCurlInput): string {
  const scheme = SIGNING[signature ?? 'generic'] ?? SIGNING.generic

  const headers = ['Content-Type: application/json', 'X-Request-ID: test-$TS-$NONCE', ...scheme.headers]

  return [
    `SECRET=${shellQuote(secret || WEBHOOK_SECRET_PLACEHOLDER)}`,
    `BODY=${shellQuote(WEBHOOK_TEST_BODY)}`,
    'TS=$(date +%s) NONCE=$(openssl rand -hex 6)',
    ...(scheme.mac ? [scheme.mac] : []),
    `curl -sS -X POST ${shellQuote(url)} \\`,
    ...headers.map(header => `  -H "${header}" \\`),
    '  -d "$BODY"'
  ].join('\n')
}
