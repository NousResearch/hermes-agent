import { expect, test } from 'vitest'

import { trustedRendererOrigin } from './renderer-origin'

test('uses only the trusted renderer origin, without path or query', () => {
  expect(trustedRendererOrigin('http://127.0.0.1:47891/chat?profile=x')).toBe('http://127.0.0.1:47891')
  expect(trustedRendererOrigin('file:///Applications/Hermes.app/index.html')).toBeUndefined()
  expect(trustedRendererOrigin('https://attacker.example/path')).toBeUndefined()
  expect(trustedRendererOrigin('http://127.0.0.1/path')).toBeUndefined()
  expect(trustedRendererOrigin('http://localhost:47891/path')).toBeUndefined()
})
