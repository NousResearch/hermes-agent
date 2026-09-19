// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { SANDBOXED_FRAME_DEFAULT_SANDBOX, SandboxedFrame, sanitizeFrameSandbox } from './sandboxed-frame'

afterEach(cleanup)

describe('sanitizeFrameSandbox', () => {
  it('defaults to the opaque-origin posture for an empty or missing sandbox', () => {
    expect(sanitizeFrameSandbox(undefined)).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    expect(sanitizeFrameSandbox('')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    expect(sanitizeFrameSandbox('   ')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
  })

  it('keeps safe tokens and dedupes them', () => {
    expect(sanitizeFrameSandbox('allow-scripts allow-forms')).toBe('allow-scripts allow-forms')
    expect(sanitizeFrameSandbox('allow-scripts allow-scripts allow-forms')).toBe('allow-scripts allow-forms')
  })

  it('is case-insensitive, because the attribute is (a mixed-case escape used to pass)', () => {
    // HTML: "an unordered set of unique space-separated tokens that are ASCII
    // case-insensitive", and Chromium lower-cases each token before matching.
    expect(sanitizeFrameSandbox('ALLOW-SAME-ORIGIN allow-scripts')).toBe('allow-scripts')
    expect(sanitizeFrameSandbox('Allow-Top-Navigation allow-forms')).toBe('allow-forms')
    expect(sanitizeFrameSandbox('ALLOW-SCRIPTS Allow-Forms')).toBe('allow-scripts allow-forms')
  })

  it('drops a token it does not know instead of forwarding it', () => {
    // An allowlist, not a blocklist: a token nobody has heard of (or a future
    // one) must not reach the attribute on the strength of being unlisted.
    expect(sanitizeFrameSandbox('allow-scripts allow-invented-thing')).toBe('allow-scripts')
    expect(sanitizeFrameSandbox('allow-invented-thing')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
  })

  it('allows the safe tokens a real embed asks for', () => {
    expect(sanitizeFrameSandbox('allow-downloads allow-forms allow-presentation')).toBe(
      'allow-downloads allow-forms allow-presentation'
    )
  })

  it('strips every realm-escaping token, falling back to the default when nothing is left', () => {
    expect(sanitizeFrameSandbox('allow-scripts allow-same-origin')).toBe('allow-scripts')
    expect(sanitizeFrameSandbox('allow-top-navigation allow-popups')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    expect(sanitizeFrameSandbox('allow-same-origin')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    expect(sanitizeFrameSandbox('allow-modals allow-storage-access-by-user-activation')).toBe(
      SANDBOXED_FRAME_DEFAULT_SANDBOX
    )
  })
})

describe('SandboxedFrame', () => {
  it('renders a sandboxed, no-referrer, lazily-loaded iframe', () => {
    const { container } = render(<SandboxedFrame src="https://example.com/feed" title="Feed" />)
    const frame = container.querySelector('iframe')!

    expect(frame.getAttribute('sandbox')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer')
    expect(frame.getAttribute('loading')).toBe('lazy')
    expect(frame.getAttribute('src')).toBe('https://example.com/feed')
    expect(frame.getAttribute('title')).toBe('Feed')
  })

  it('keeps its no-referrer / lazy posture when a caller tries to override it', () => {
    const { container } = render(
      <SandboxedFrame loading="eager" referrerPolicy="unsafe-url" src="https://example.com" title="Feed" />
    )

    const frame = container.querySelector('iframe')!

    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer')
    expect(frame.getAttribute('loading')).toBe('lazy')
  })

  it('refuses an allow-same-origin / popup escape even when asked for one', () => {
    const { container } = render(
      <SandboxedFrame sandbox="allow-scripts allow-same-origin allow-popups" src="https://example.com" title="Feed" />
    )

    expect(container.querySelector('iframe')!.getAttribute('sandbox')).toBe('allow-scripts')
  })
})
