import { describe, expect, it } from 'vitest'

import {
  generatedViewApprovalSource,
  generatedViewEntryPath,
  generatedViewPathIsContained,
  parseGeneratedViewManifest,
  validateGeneratedViewManifest
} from './manifest'

const valid = {
  version: 1,
  id: 'usage-monitor',
  title: 'Usage Monitor',
  entry: 'index.html',
  capabilities: ['theme:read', 'state:persist'],
  bindings: ['hermes:usage-30d']
}

describe('generated-view manifest', () => {
  it('accepts the v1 document and normalizes duplicate authority declarations', () => {
    expect(validateGeneratedViewManifest({ ...valid, capabilities: ['theme:read', 'theme:read'] })).toEqual({
      ...valid,
      capabilities: ['theme:read']
    })
  })

  it.each([
    ['directory mismatch', { ...valid, id: 'other' }],
    ['uppercase id', { ...valid, id: 'Usage' }],
    ['traversal', { ...valid, entry: '../index.html' }],
    ['absolute path', { ...valid, entry: '/tmp/index.html' }],
    ['Windows absolute path', { ...valid, entry: 'C:\\tmp\\index.html' }],
    ['non-HTML entry', { ...valid, entry: 'index.js' }],
    ['unknown capability', { ...valid, capabilities: ['network:read'] }],
    ['unknown binding', { ...valid, bindings: ['gateway:anything'] }],
    ['unknown field', { ...valid, approved: true }]
  ])('rejects %s', (_label, manifest) => {
    expect(() => validateGeneratedViewManifest(manifest, 'usage-monitor')).toThrow()
  })

  it('fails closed on malformed and oversized JSON', () => {
    expect(() => parseGeneratedViewManifest('{')).toThrow(/valid JSON/)
    expect(() => parseGeneratedViewManifest(JSON.stringify({ ...valid, title: 'x'.repeat(33_000) }))).toThrow(/exceeds/)
  })

  it('joins and contains POSIX and Windows paths without collisions', () => {
    expect(generatedViewEntryPath('/home/me/.hermes/generated-views/demo', 'pages/index.html')).toBe(
      '/home/me/.hermes/generated-views/demo/pages/index.html'
    )
    expect(generatedViewPathIsContained('/home/me/views/demo', '/home/me/views/demo/index.html')).toBe(true)
    expect(generatedViewPathIsContained('/home/me/views/demo', '/home/me/views/demo-evil/index.html')).toBe(false)
    expect(generatedViewPathIsContained('C:\\Users\\Me\\views\\demo', 'c:\\users\\me\\views\\demo\\index.html')).toBe(
      true
    )
  })

  it('binds capabilities and bindings into approval input', () => {
    const manifest = validateGeneratedViewManifest(valid)

    expect(generatedViewApprovalSource(manifest, '<p>ok</p>')).not.toBe(
      generatedViewApprovalSource({ ...manifest, capabilities: [] }, '<p>ok</p>')
    )
    expect(generatedViewApprovalSource(manifest, '<p>ok</p>')).not.toBe(
      generatedViewApprovalSource({ ...manifest, bindings: [] }, '<p>ok</p>')
    )
  })
})
