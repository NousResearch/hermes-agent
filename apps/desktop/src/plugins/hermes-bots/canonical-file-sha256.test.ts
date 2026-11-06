import { createHash, webcrypto } from 'node:crypto'

import { afterEach, expect, it, vi } from 'vitest'

import { canonicalFileSha256 } from './canonical-file-sha256'

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it.each([
  ['', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'],
  ['abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'],
  ['abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq', '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'],
  ['a'.repeat(1_000_000), 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0']
])('verifies SHA-256 vector %# without SubtleCrypto', async (input, expected) => {
  vi.stubGlobal('crypto', { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) })
  expect(await canonicalFileSha256(new TextEncoder().encode(input))).toBe(expected)
})

it.each([1, 55, 56, 63, 64, 65, 127, 128, 15_000_000])('verifies %i binary bytes at padding boundaries and the file limit over HTTP', async length => {
  vi.stubGlobal('crypto', { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) })
  const storage = Uint8Array.from({ length: length + 2 }, (_, index) => index % 256)
  const bytes = storage.subarray(1, length + 1)
  expect(await canonicalFileSha256(bytes)).toBe(createHash('sha256').update(bytes).digest('hex'))
})

it('uses Web Crypto when available', async () => {
  vi.stubGlobal('crypto', webcrypto)
  const digest = vi.spyOn(webcrypto.subtle, 'digest')
  const bytes = new TextEncoder().encode('abc')
  expect(await canonicalFileSha256(bytes)).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
  expect(digest).toHaveBeenCalledWith('SHA-256', bytes)
})
