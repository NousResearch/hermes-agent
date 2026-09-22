/**
 * gateway-probe-errors.ts (ForgeGuard fork)
 *
 * Plain-language text for gateway probe failures whose raw form points the
 * user at the wrong thing. Kept free of `electron` imports for unit tests.
 *
 * TLS alert 112 (`unrecognized_name`) is the one that matters in practice: it
 * reads like a certificate problem — and the connection dialog shows it right
 * above "Allow self-signed certificate" — but it means the server refused the
 * HOSTNAME before any certificate was sent. That is almost always a reverse
 * proxy with no site configured for the name (nginx's `ssl_reject_handshake`,
 * which Nginx Proxy Manager uses for unknown hosts), or DNS sending the name
 * to the wrong proxy. The self-signed toggle cannot fix it.
 */

function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).hostname || baseUrl
  } catch {
    return baseUrl
  }
}

// Matched on the message, not `code`: Node reports this alert as a bare
// EPROTO. BoringSSL (Electron) says `TLSV1_ALERT_UNRECOGNIZED_NAME`, OpenSSL
// (plain Node) says `tlsv1 unrecognized name`; both end "SSL alert number 112".
export function describeGatewayProbeError(error: unknown, baseUrl: string): string {
  const raw = (error instanceof Error ? error.message : String(error)).trim()

  if (/unrecognized[_ ]name/i.test(raw)) {
    const host = hostOf(baseUrl)

    return (
      `The server answering for ${host} does not recognise that hostname (TLS "unrecognized name"). ` +
      `This is not a certificate problem: the reverse proxy in front of the gateway has no site ` +
      `configured for ${host}, or DNS points the name at the wrong proxy. (${raw})`
    )
  }

  return raw
}
