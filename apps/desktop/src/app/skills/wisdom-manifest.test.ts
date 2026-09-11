import { describe, expect, it } from 'vitest'

import {
  parseWisdomManifest,
  parseWisdomSystemSpecification,
  wisdomManifestValidationError,
  wisdomSystemSpecificationValidationError
} from './wisdom-manifest'

const specification = {
  hermes: { minimum_version: '0.20.5' },
  platforms: ['macOS'],
  architectures: ['arm64'],
  model: { capabilities: ['vision'], minimum_context_window: 32_000 },
  tools: [{ name: 'browser', minimum_version: null, auto_install: false as const, requires_admin: false }],
  plugins: [{ id: 'research-tools', minimum_version: '1.2.0', required: true }],
  credentials: ['portal'],
  connections: ['gateway'],
  filesystem: { read: ['/workspace'], write: ['/tmp'] },
  network: { destinations: ['portal.nousresearch.com'] },
  runtime: { shell: false, browser: true, code: false, sandbox: true },
  hardware: [],
  known_limitations: ['Requires an authenticated Portal session.']
}

describe('Desktop Wisdom manifest forms', () => {
  it('parses the complete V1 contract without dropping structured fields', () => {
    expect(parseWisdomSystemSpecification(specification)).toEqual(specification)
    expect(
      parseWisdomManifest(JSON.stringify({ schema_version: 1, name: 'research-helper', requirements: specification }))
    ).toEqual({ schema_version: 1, name: 'research-helper', requirements: specification })
  })

  it('fails closed on unsupported fields and automatic dependency installation', () => {
    expect(() => parseWisdomSystemSpecification({ ...specification, surprise: true })).toThrow(/unsupported fields/)
    expect(() =>
      parseWisdomSystemSpecification({
        ...specification,
        tools: [{ name: 'browser', minimum_version: null, auto_install: true, requires_admin: false }]
      })
    ).toThrow(/cannot request automatic installation/)
  })

  it('reports form-level validation errors before a revision is submitted', () => {
    expect(wisdomSystemSpecificationValidationError({ ...specification, platforms: [''] })).toMatch(/required/)
    expect(
      wisdomManifestValidationError(
        JSON.stringify({ schema_version: 1, name: '', requirements: specification })
      )
    ).toBe('Skill name is required.')
  })
})
