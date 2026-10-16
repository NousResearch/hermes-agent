// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SANDBOXED_FRAME_DEFAULT_SANDBOX, SandboxedFrame, sanitizeFrameSandbox } from './sandboxed-frame'

afterEach(cleanup)

describe('sanitizeFrameSandbox', () => {
  it('strips every realm-escaping or unknown token, case-insensitively, and never emits an empty attribute', () => {
    // HTML: sandbox is "an unordered set of unique space-separated tokens that
    // are ASCII case-insensitive"; Chromium lower-cases each token before
    // matching, so `ALLOW-SAME-ORIGIN` once sailed past a case-sensitive check.
    expect(sanitizeFrameSandbox('allow-scripts allow-same-origin')).toBe('allow-scripts')
    expect(sanitizeFrameSandbox('ALLOW-SAME-ORIGIN Allow-Top-Navigation allow-scripts')).toBe('allow-scripts')
    expect(sanitizeFrameSandbox('allow-scripts allow-invented-thing')).toBe('allow-scripts')
    // A frame with NO sandbox attribute is fully privileged — an emptied set
    // must fall back to the default posture, not to nothing.

    for (const escape of [
      undefined,
      '   ',
      'allow-same-origin',
      'allow-top-navigation allow-top-navigation-by-user-activation allow-popups allow-popups-to-escape-sandbox',
      'allow-modals allow-storage-access-by-user-activation'
    ]) {
      expect(sanitizeFrameSandbox(escape)).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
    }
  })

  it('keeps the allowlisted tokens a real embed asks for, deduped', () => {
    expect(sanitizeFrameSandbox('allow-scripts allow-scripts allow-forms')).toBe('allow-scripts allow-forms')
    expect(sanitizeFrameSandbox('allow-downloads allow-forms allow-presentation')).toBe(
      'allow-downloads allow-forms allow-presentation'
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

  it('keeps its posture when a caller tries to re-open it through props', () => {
    // The props type is an explicit allowlist, so a plugin has to cast to get
    // these past the compiler — and the DOM must still never see them:
    // `allow` delegates Permissions-Policy (the electron permission handlers
    // grant media without checking the frame's origin), `srcdoc` overrides
    // `src`, `name` makes the frame a navigation target.
    const hostile = {
      allow: 'camera; microphone',
      allowFullScreen: true,
      loading: 'eager',
      name: 'target',
      referrerPolicy: 'unsafe-url',
      srcDoc: '<script>1</script>'
    } as Record<string, unknown>

    const { container } = render(
      <SandboxedFrame
        {...hostile}
        sandbox="allow-scripts allow-same-origin allow-popups"
        src="https://example.com"
        title="Feed"
      />
    )

    const frame = container.querySelector('iframe')!

    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer')
    expect(frame.getAttribute('loading')).toBe('lazy')

    for (const attr of ['allow', 'allowfullscreen', 'name', 'srcdoc']) {
      expect(frame.hasAttribute(attr), attr).toBe(false)
    }
  })

  it('renders nothing for a src outside http(s)/data and says why', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    for (const src of ['file:///etc/passwd', 'blob:https://example.com/x', 'javascript:1', '/relative']) {
      const { container, unmount } = render(<SandboxedFrame src={src} title="Feed" />)

      expect(container.querySelector('iframe'), src).toBeNull()
      unmount()
    }

    expect(warn).toHaveBeenCalledTimes(4)
    warn.mockRestore()
  })
})
