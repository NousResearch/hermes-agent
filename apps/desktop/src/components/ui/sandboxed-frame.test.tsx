// @vitest-environment jsdom
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { Activity, createRef, type SyntheticEvent } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SANDBOXED_FRAME_DEFAULT_SANDBOX, SandboxedFrame, sanitizeFrameSandbox } from './sandboxed-frame'

afterEach(() => {
  cleanup()
  window.document.documentElement.removeAttribute('data-hermes-desktop-host')
  vi.unstubAllGlobals()
})

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

describe('SandboxedFrame in the Webapp', () => {
  const guest = `data:text/html;charset=utf-8,${encodeURIComponent('<script>parent.postMessage("ready", "*")</script>')}`

  function browserHost() {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal('__HERMES_BASE_PATH__', '/hermes')
    vi.stubGlobal('__HERMES_UI_SURFACE__', 'webapp')
  }

  function guestLoaded(frame: HTMLIFrameElement) {
    act(() => {
      window.dispatchEvent(
        new MessageEvent('message', { data: { type: 'hermes-guest-loaded' }, source: frame.contentWindow })
      )
    })
  }

  it.each([false, true])('dispatches fresh guest loads with normal propagation (stopPropagation=%s)', stop => {
    browserHost()
    const order: string[] = []
    const events: SyntheticEvent<HTMLIFrameElement>[] = []
    const targets: (EventTarget | null)[] = []

    const onLoad = (event: SyntheticEvent<HTMLIFrameElement>) => {
      order.push('sdk')
      events.push(event)
      targets.push(event.currentTarget, event.nativeEvent.target)
      event.currentTarget.contentWindow!.postMessage('initialize', '*')

      if (stop) {
        event.stopPropagation()
      }
    }

    const { container } = render(
      <div onLoad={() => order.push('ancestor')}>
        <SandboxedFrame onLoad={onLoad} src={guest} title="Widget" />
      </div>
    )

    const frame = container.querySelector('iframe')!

    frame.addEventListener('load', () => order.push('native'))
    fireEvent.load(frame)
    expect(order).toEqual([])

    for (let count = 1; count <= 3; count += 1) {
      guestLoaded(frame)
      expect(order.splice(0)).toEqual(stop ? ['sdk', 'native'] : ['sdk', 'ancestor', 'native'])
      expect(targets.splice(0)).toEqual([frame, frame])
      expect(new Set(events).size).toBe(count)
      expect(new Set(events.map(event => event.nativeEvent)).size).toBe(count)
    }
  })

  it('remounts only when the sanitized sandbox changes', () => {
    browserHost()
    const { container, rerender } = render(<SandboxedFrame sandbox="allow-scripts" src={guest} title="Widget" />)
    const initial = container.querySelector('iframe')!

    rerender(<SandboxedFrame sandbox="ALLOW-SCRIPTS allow-same-origin" src={guest} title="Widget" />)
    expect(container.querySelector('iframe')).toBe(initial)
    rerender(<SandboxedFrame sandbox="allow-scripts allow-forms" src={guest} title="Widget" />)
    const replacement = container.querySelector('iframe')!

    expect(replacement).not.toBe(initial)
    expect(replacement.getAttribute('sandbox')).toBe('allow-scripts allow-forms')
    rerender(<SandboxedFrame sandbox="allow-scripts allow-forms" src={guest} title="Widget" />)
    expect(container.querySelector('iframe')).toBe(replacement)
  })

  it('preserves an unchanged document across Activity reveals but sends changes and shell reloads', () => {
    browserHost()

    const view = (mode: 'hidden' | 'visible', src = guest) => (
      <Activity mode={mode}>
        <SandboxedFrame src={src} title="Widget" />
      </Activity>
    )

    const { container, rerender } = render(view('visible'))
    const frame = container.querySelector('iframe')!
    const send = vi.spyOn(frame.contentWindow!, 'postMessage')

    fireEvent.load(frame)
    expect(send).toHaveBeenCalledTimes(1)
    rerender(view('hidden'))
    rerender(view('visible'))
    expect(container.querySelector('iframe')).toBe(frame)
    expect(send).toHaveBeenCalledTimes(1)

    const changed = 'data:text/html,<p>changed</p>'

    rerender(view('visible', changed))
    expect(send).toHaveBeenCalledTimes(2)
    expect(send).toHaveBeenLastCalledWith(
      { type: 'hermes-preview-document', html: '<p>changed</p>', sandbox: 'allow-scripts' },
      '*'
    )
    rerender(view('hidden', guest))
    rerender(view('visible'))
    expect(send).toHaveBeenCalledTimes(3)
    fireEvent.load(frame)
    expect(send).toHaveBeenCalledTimes(4)
  })

  it('sends the document for a shell that loads while hidden, and reports a guest that loads while hidden', () => {
    browserHost()
    const onLoad = vi.fn()

    const view = (mode: 'hidden' | 'visible') => (
      <Activity mode={mode}>
        <SandboxedFrame onLoad={onLoad} src={guest} title="Widget" />
      </Activity>
    )

    const { container, rerender } = render(view('visible'))
    const frame = container.querySelector('iframe')!
    const send = vi.spyOn(frame.contentWindow!, 'postMessage')

    rerender(view('hidden'))
    fireEvent.load(frame)
    expect(send).toHaveBeenCalledExactlyOnceWith(
      {
        type: 'hermes-preview-document',
        html: '<script>parent.postMessage("ready", "*")</script>',
        sandbox: 'allow-scripts'
      },
      '*'
    )
    rerender(view('visible'))
    rerender(view('hidden'))
    guestLoaded(frame)
    expect(onLoad).toHaveBeenCalledOnce()
    rerender(view('visible'))
    expect(send).toHaveBeenCalledOnce()
  })

  it('starts a frame inside a shadow root, where the shell load never reaches the document', () => {
    browserHost()
    const onLoad = vi.fn()
    const shell = { postMessage: vi.fn() }
    const host = window.document.createElement('div')
    const mount = window.document.createElement('div')

    window.document.body.append(host)
    host.attachShadow({ mode: 'open' }).append(mount)
    // jsdom gives an iframe inside a shadow root no window.
    const windows = vi
      .spyOn(HTMLIFrameElement.prototype, 'contentWindow', 'get')
      .mockReturnValue(shell as unknown as Window)

    try {
      const { unmount } = render(<SandboxedFrame onLoad={onLoad} src={guest} title="Widget" />, { container: mount })

      fireEvent.load(mount.querySelector('iframe')!)
      expect(shell.postMessage).toHaveBeenCalledOnce()
      expect(onLoad).not.toHaveBeenCalled()
      act(() => {
        window.dispatchEvent(
          new MessageEvent('message', { data: { type: 'hermes-guest-loaded' }, source: shell as unknown as Window })
        )
      })
      expect(onLoad).toHaveBeenCalledOnce()
      unmount()
    } finally {
      windows.mockRestore()
      host.remove()
    }
  })

  it.each([false, true])(
    'starts a frame mounted hidden on reveal, even under a cancelling ancestor capture (cancel=%s)',
    cancel => {
      browserHost()
      const onLoad = vi.fn()
      const capture = vi.fn((event: SyntheticEvent) => cancel && event.stopPropagation())

      const view = (mode: 'hidden' | 'visible') => (
        <div onLoadCapture={capture}>
          <Activity mode={mode}>
            <SandboxedFrame onLoad={onLoad} src={guest} title="Widget" />
          </Activity>
        </div>
      )

      const { container, rerender } = render(view('hidden'))
      const frame = container.querySelector('iframe')!
      const send = vi.spyOn(frame.contentWindow!, 'postMessage')

      fireEvent.load(frame)
      expect(capture).not.toHaveBeenCalled()
      rerender(view('visible'))
      expect(send).toHaveBeenCalledOnce()
      rerender(view('visible'))
      expect(send).toHaveBeenCalledOnce()
      guestLoaded(frame)
      expect(capture).toHaveBeenCalledOnce()
      expect(onLoad).toHaveBeenCalledTimes(cancel ? 0 : 1)
    }
  )

  it('keeps the shell load from ancestor capture handlers, which cannot stop the document', () => {
    browserHost()
    const order: string[] = []

    const { container } = render(
      <div
        onLoadCapture={event => {
          order.push('capture')
          event.stopPropagation()
        }}
      >
        <SandboxedFrame onLoad={() => order.push('sdk')} src={guest} title="Widget" />
      </div>
    )

    const frame = container.querySelector('iframe')!
    const send = vi.spyOn(frame.contentWindow!, 'postMessage')

    fireEvent.load(frame)
    expect(order).toEqual([])
    expect(send).toHaveBeenCalledOnce()
    guestLoaded(frame)
    expect(order).toEqual(['capture'])
  })

  it('serves data: HTML through the guest shell with the caller tokens, and fires onLoad for the guest', () => {
    browserHost()
    const onLoad = vi.fn()
    const ref = createRef<HTMLIFrameElement>()

    const { container } = render(
      <SandboxedFrame
        onLoad={onLoad}
        ref={ref}
        sandbox="allow-forms ALLOW-SAME-ORIGIN allow-popups"
        src={guest}
        title="Widget"
      />
    )

    const frame = container.querySelector('iframe')!

    expect(ref.current).toBe(frame)
    expect(frame.getAttribute('srcdoc')).toBeNull()
    expect(frame.src).toBe(`${window.location.origin}/hermes/assets/webapp-guest.html`)
    expect(frame.getAttribute('sandbox')).toBe('allow-forms allow-scripts')
    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer')

    const send = vi.spyOn(frame.contentWindow!, 'postMessage')
    fireEvent.load(frame)
    expect(send).toHaveBeenLastCalledWith(
      {
        type: 'hermes-preview-document',
        html: '<script>parent.postMessage("ready", "*")</script>',
        sandbox: 'allow-forms'
      },
      '*'
    )
    // The shell loading is not the guest loading.
    expect(onLoad).not.toHaveBeenCalled()

    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { type: 'hermes-guest-loaded' }, source: window }))
    })
    expect(onLoad).not.toHaveBeenCalled()

    act(() => {
      window.dispatchEvent(
        new MessageEvent('message', { data: { type: 'hermes-guest-loaded' }, source: frame.contentWindow })
      )
    })
    expect(onLoad).toHaveBeenCalledTimes(1)
  })

  it('keeps http(s) and non-HTML data: embeds on the direct frame', () => {
    browserHost()

    for (const src of ['https://example.com/feed', 'data:image/svg+xml,%3Csvg%2F%3E', 'data:,plain']) {
      const { container, unmount } = render(<SandboxedFrame src={src} title="Feed" />)

      expect(container.querySelector('iframe')!.getAttribute('src'), src).toBe(src)
      unmount()
    }
  })

  it('leaves native Desktop on the direct data: frame', () => {
    const { container } = render(<SandboxedFrame src={guest} title="Widget" />)
    const frame = container.querySelector('iframe')!

    expect(frame.getAttribute('src')).toBe(guest)
    expect(frame.getAttribute('sandbox')).toBe(SANDBOXED_FRAME_DEFAULT_SANDBOX)
  })
})
