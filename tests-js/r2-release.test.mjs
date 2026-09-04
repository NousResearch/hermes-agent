/**
 * scripts/r2-release.mjs — SigV4 signer pinned against AWS botocore (the
 * reference implementation, 1.43.81) at a FIXED timestamp/creds, so the
 * expected values are reproducible fixtures rather than self-consistency.
 *
 * Vectors were generated with the venv script at
 *   sigv4-venv/gen_vectors.py (botocore.auth.SigV4Auth / S3SigV4Auth,
 *   timestamp 20150830T123600Z, creds AKIDEXAMPLE):
 *   - get-vanilla        generic signer (no x-amz-content-sha256), example.com
 *   - r2-put-payload     S3 signer, payload hash signed, region auto
 *   - r2-list            S3 signer, ListObjectsV2 with query params
 *   - r2-delete          S3 signer, region auto
 * The get-vanilla case also reproduces the public aws-sig-v4-test-suite
 * request shape (verified by independent spec computation).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  authHeader,
  canonicalQuery,
  canonicalRequest,
  channelForTag,
  contentTypeFor,
  encodeKeyPath,
  feedDirFor,
  mergeFeedYmls,
  canaryDoomedKeys,
  rewriteFeedPaths,
  stagingKeyFor,
  parseListXml,
  rfc3986Encode,
} from '../scripts/r2-release.mjs'

const AKID = 'AKIDEXAMPLE'
const SECRET = 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'
const NOW = '20150830T123600Z'
const EMPTY_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

test('get-vanilla matches the AWS test-suite vector', () => {
  const authz = authHeader({
    method: 'GET',
    host: 'example.com',
    path: '/',
    query: '',
    headers: { host: 'example.com', 'x-amz-date': NOW },
    payloadHash: EMPTY_SHA,
    accessKeyId: AKID,
    secretKey: SECRET,
    now: NOW,
    region: 'us-east-1',
    service: 'service',
  })
  assert.equal(
    authz,
    'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, ' +
      'SignedHeaders=host;x-amz-date, ' +
      'Signature=33399fd3d4a9d6104710c7c04005f7c959f8b1f8bf41b823587ed36b079e453f',
  )
})

test('r2-put-payload matches botocore (S3 signer, payload hash, region auto)', () => {
  const bodyHash = '44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072' // sha256("Welcome to Amazon S3.")
  const host = 'abc123.r2.cloudflarestorage.com'
  const authz = authHeader({
    method: 'PUT',
    host,
    path: '/hermes-releases/HermesBundled-0.28.0-win-x64.msix',
    query: '',
    headers: { host, 'x-amz-date': NOW, 'x-amz-content-sha256': bodyHash },
    payloadHash: bodyHash,
    accessKeyId: AKID,
    secretKey: SECRET,
    now: NOW,
  })
  assert.equal(
    authz,
    'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, ' +
      'SignedHeaders=host;x-amz-content-sha256;x-amz-date, ' +
      'Signature=05ba50acfb54042fac330848af50877e5fb477c4f2063c2f77f9cc80855eb1e9',
  )
})

test('r2-list matches botocore (canonical query sorted, empty payload hash)', () => {
  const query = canonicalQuery({ 'list-type': '2', prefix: 'HermesBundled-0.28.0-', 'max-keys': '1000' })
  assert.equal(query, 'list-type=2&max-keys=1000&prefix=HermesBundled-0.28.0-')
  const host = 'abc123.r2.cloudflarestorage.com'
  const authz = authHeader({
    method: 'GET',
    host,
    path: '/hermes-releases',
    query,
    headers: { host, 'x-amz-date': NOW, 'x-amz-content-sha256': EMPTY_SHA },
    payloadHash: EMPTY_SHA,
    accessKeyId: AKID,
    secretKey: SECRET,
    now: NOW,
  })
  assert.equal(
    authz,
    'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, ' +
      'SignedHeaders=host;x-amz-content-sha256;x-amz-date, ' +
      'Signature=3ec423c452a318664c85fbcc25667ad07201aedce688e3bb6b345b4baaa39d90',
  )
})

test('r2-delete matches botocore', () => {
  const host = 'abc123.r2.cloudflarestorage.com'
  const authz = authHeader({
    method: 'DELETE',
    host,
    path: '/hermes-releases/HermesBundled-0.28.0-canary.20260818-win-arm64.msix',
    query: '',
    headers: { host, 'x-amz-date': NOW, 'x-amz-content-sha256': EMPTY_SHA },
    payloadHash: EMPTY_SHA,
    accessKeyId: AKID,
    secretKey: SECRET,
    now: NOW,
  })
  assert.equal(
    authz,
    'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, ' +
      'SignedHeaders=host;x-amz-content-sha256;x-amz-date, ' +
      'Signature=ec5ccb76f701193b28aaca052cabbf2f084e9c71ffc64b872fa51d4e70dc6e55',
  )
})

test('rfc3986Encode escapes the AWS reserved set, keeps unreserved', () => {
  assert.equal(rfc3986Encode('HermesBundled-0.28.0-win-x64.msix'), 'HermesBundled-0.28.0-win-x64.msix')
  assert.equal(rfc3986Encode("a b!'()*c"), 'a%20b%21%27%28%29%2Ac')
})

test('encodeKeyPath encodes segment-wise, preserves separators', () => {
  assert.equal(encodeKeyPath('HermesBundled-0.28.0-win-x64.msix'), 'HermesBundled-0.28.0-win-x64.msix')
  assert.equal(encodeKeyPath('a b/c d'), 'a%20b/c%20d')
})

test('parseListXml extracts keys, truncation, continuation token, entities', () => {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>hermes-releases</Name>
  <Prefix></Prefix>
  <KeyCount>3</KeyCount>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>true</IsTruncated>
  <Contents><Key>HermesBundled-0.28.0-win-x64.msix</Key><LastModified>2026-08-18T00:00:00Z</LastModified><Size>123</Size></Contents>
  <Contents><Key>a&amp;b.msix</Key><LastModified>2026-08-18T00:00:00Z</LastModified><Size>1</Size></Contents>
  <Contents><Key>latest.yml</Key><LastModified>2026-08-18T00:00:00Z</LastModified><Size>2</Size></Contents>
  <NextContinuationToken>abc+def/=</NextContinuationToken>
</ListBucketResult>`
  const parsed = parseListXml(xml)
  assert.deepEqual(parsed.keys, ['HermesBundled-0.28.0-win-x64.msix', 'a&b.msix', 'latest.yml'])
  assert.equal(parsed.truncated, true)
  assert.equal(parsed.nextToken, 'abc+def/=')
})

test('canaryDoomedKeys dates by the key suffix, ignores stable artifacts', () => {
  const keys = [
    'releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix', // stable — never doomed
    'releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix',
    'releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix', // today — kept
    'releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix.blockmap',
    'latest.yml',
    'canary.yml',
  ]
  assert.deepEqual(canaryDoomedKeys(keys, '20260814'), [
    'releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix',
    'releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix.blockmap',
  ])
})

test('channelForTag maps stable vs canary', () => {
  assert.equal(channelForTag('v0.28.0'), 'stable')
  assert.equal(channelForTag('v0.28.0-canary.20260818101010'), 'canary')
  assert.equal(channelForTag('v0.28.0-canary.20260818'), 'canary')
})

test('stagingKeyFor + feedDirFor produce the layout keys', () => {
  assert.equal(stagingKeyFor('v0.28.0', 'HermesBundled-0.28.0-win-x64.msix'),
    'releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix')
  assert.equal(feedDirFor('win32', 'stable'), 'releases/win32/stable')
  assert.equal(feedDirFor('darwin', 'canary'), 'releases/darwin/canary')
})

test('contentTypeFor maps MSIX / App Installer artifacts to their MIME types', () => {
  assert.equal(contentTypeFor('HermesBundled-0.28.0-win-x64.msix'), 'application/msix')
  assert.equal(contentTypeFor('HermesBundled-0.28.0-win.msixbundle'), 'application/msixbundle')
  assert.equal(contentTypeFor('stable.appinstaller'), 'application/appinstaller')
  assert.equal(contentTypeFor('HermesBundled-0.28.0-mac-x64.dmg'), undefined)
  assert.equal(contentTypeFor('latest-mac.yml'), undefined)
  // Case-insensitive on the suffix.
  assert.equal(contentTypeFor('X.APPINSTALLER'), 'application/appinstaller')
})

test('canonicalRequest reads mixed-case header values (Content-Type)', () => {
  // Regression: canonicalRequest lowercased the header NAME for the canonical
  // line but read the value with `headers[lowerName]` — a 'Content-Type'
  // value came out as 'undefined' while R2 canonicalized 'application/msix',
  // so every signed msix PUT (the only artifact with Content-Type) failed
  // with SignatureDoesNotMatch / 403. Value lookup must be case-insensitive.
  const host = 'abc123.r2.cloudflarestorage.com'
  const now = '20150830T123600Z'
  const bodyHash = '44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072'
  const headers = {
    host,
    'x-amz-date': now,
    'x-amz-content-sha256': bodyHash,
    'Content-Type': 'application/msix'
  }
  const canon = canonicalRequest('PUT', '/hermes-releases/HermesBundled-0.28.0-win-x64.msix', '', headers, bodyHash)
  assert.ok(canon.includes('content-type:application/msix'), 'content-type value must survive canonicalization')
  assert.ok(!canon.includes('undefined'), 'no undefined values leaked into the canonical request')
  // The canonical line ordering + header list match what SigV4/R2 recompute.
  assert.ok(canon.includes('content-type;host;x-amz-content-sha256;x-amz-date'))
})

test('rewriteFeedPaths rewrites path:/url: to absolute /releases/tag keys, idempotent', () => {
  const absKey = (f) => `/releases/tag/v0.28.0/${f}`
  const yml = `version: 0.28.0
files:
  - url: HermesBundled-0.28.0-mac-x64.zip
    sha512: abc
    size: 1
  - url: HermesBundled-0.28.0-mac-x64.dmg
    sha512: def
    size: 2
path: HermesBundled-0.28.0-mac-x64.zip
sha512: ghi
releaseDate: '2026-08-18T00:00:00.000Z'
`
  const once = rewriteFeedPaths(yml, absKey)
  assert.ok(once.includes('url: /releases/tag/v0.28.0/HermesBundled-0.28.0-mac-x64.zip'))
  assert.ok(once.includes('url: /releases/tag/v0.28.0/HermesBundled-0.28.0-mac-x64.dmg'))
  assert.ok(once.includes('path: /releases/tag/v0.28.0/HermesBundled-0.28.0-mac-x64.zip'))
  assert.ok(once.includes('sha512: abc')) // artifact hashes untouched
  // Already-absolute values are left alone (a re-finalize must not double-prefix).
  assert.equal(rewriteFeedPaths(once, absKey), once)
})

test('mergeFeedYmls concatenates files[] lists, dedupes, keeps head', () => {
  const x64 = `version: 0.28.0
files:
  - url: HermesBundled-0.28.0-mac-x64.zip
    sha512: abc
    size: 1
  - url: HermesBundled-0.28.0-mac-x64.dmg
    sha512: def
    size: 2
path: HermesBundled-0.28.0-mac-x64.zip
sha512: ghi
releaseDate: '2026-08-18T00:00:00.000Z'
`
  const arm64 = `version: 0.28.0
files:
  - url: HermesBundled-0.28.0-mac-arm64.zip
    sha512: jkl
    size: 3
  - url: HermesBundled-0.28.0-mac-arm64.dmg
    sha512: mno
    size: 4
path: HermesBundled-0.28.0-mac-arm64.zip
sha512: pqr
releaseDate: '2026-08-18T00:00:00.000Z'
`
  const merged = mergeFeedYmls([x64, arm64])
  assert.ok(merged.includes('url: HermesBundled-0.28.0-mac-x64.zip'))
  assert.ok(merged.includes('url: HermesBundled-0.28.0-mac-arm64.zip'))
  assert.ok(merged.includes('releaseDate'))
  // Idempotent: merging the merged output adds nothing new.
  assert.equal(mergeFeedYmls([merged, arm64]), merged)
})
