import { describe, expect, it } from 'vitest'

import { describeGatewayProbeError } from './gateway-probe-errors'

// ForgeGuard fork — the raw text of a rejected SNI reads like a certificate
// problem and sits right above "Allow self-signed certificate", which cannot
// fix it.

describe('describeGatewayProbeError', () => {
  // Both texts captured from a real handshake refused with alert 112 — the
  // first from Electron 40's Node (BoringSSL, as the desktop app runs), the
  // second from plain Node 22 (OpenSSL). Node reports both as code EPROTO.
  it.each([
    'write EPROTO 22299470794240:error:10000458:SSL routines:OPENSSL_internal:TLSV1_ALERT_UNRECOGNIZED_NAME:../../third_party/boringssl/src/ssl/tls_record.cc:486:SSL alert number 112\n',
    'write EPROTO C0CC064D017F0000:error:0A000458:SSL routines:ssl3_read_bytes:tlsv1 unrecognized name:../deps/openssl/openssl/ssl/record/rec_layer_s3.c:918:SSL alert number 112\n'
  ])('explains a rejected hostname instead of echoing the TLS library: %s', raw => {
    const text = describeGatewayProbeError(
      Object.assign(new Error(raw), { code: 'EPROTO' }),
      'https://edi.example.test/hermes'
    )

    expect(text).toContain('edi.example.test does not recognise that hostname')
    expect(text).toContain('not a certificate problem')
    expect(text).toContain(`(${raw.trim()})`)
  })

  it('passes every other failure through untouched', () => {
    expect(describeGatewayProbeError(new Error('connect ECONNREFUSED 10.0.0.1:443'), 'https://x.test')).toBe(
      'connect ECONNREFUSED 10.0.0.1:443'
    )
    expect(describeGatewayProbeError('plain string', 'not a url')).toBe('plain string')
  })
})
