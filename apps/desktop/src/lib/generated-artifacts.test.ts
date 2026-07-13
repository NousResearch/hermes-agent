import { describe, expect, it } from 'vitest'

import {
  extractGeneratedArtifactTargetsFromText,
  extractGeneratedArtifactTargetsFromToolPayload,
  isGeneratedArtifactTarget
} from './generated-artifacts'

describe('generated artifact detection', () => {
  it('extracts generated file paths from assistant text', () => {
    expect(
      extractGeneratedArtifactTargetsFromText(
        'ZIP created: /tmp/test-backup.zip\nPDF: C:\\Users\\me\\Documents\\report.pdf\nSee https://example.com/docs'
      )
    ).toEqual(['/tmp/test-backup.zip', 'C:\\Users\\me\\Documents\\report.pdf'])
  })

  it('extracts nested tool result paths and JSON-encoded tool output', () => {
    expect(
      extractGeneratedArtifactTargetsFromToolPayload({
        args: { path: './dist/index.html' },
        result: {
          message: '{"download_path":"/tmp/report.pdf","ignored":"plain text"}',
          output_file: '/tmp/archive.zip'
        }
      })
    ).toEqual(['./dist/index.html', '/tmp/report.pdf', '/tmp/archive.zip'])
  })

  it('keeps localhost previews but rejects ordinary non-artifact links', () => {
    expect(isGeneratedArtifactTarget('http://localhost:5173/')).toBe(true)
    expect(isGeneratedArtifactTarget('https://example.com/report.pdf')).toBe(true)
    expect(isGeneratedArtifactTarget('https://example.com/docs')).toBe(false)
  })
})
