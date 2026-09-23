// #77311: the renderer heap ceiling knob (`desktop.renderer_max_old_space_mb`)
// and `desktop.electron_flags` must reach app.commandLine on packaged launches
// that never go through the `hermes desktop` launcher.
import { describe, expect, it, vi } from 'vitest'

import { planLaunchSwitches, readDesktopLaunchConfig } from './renderer-heap-flags'

describe('renderer heap flags', () => {
  it('reads both launch keys from config.yaml and plans a well-formed js-flags switch', () => {
    const cfg = readDesktopLaunchConfig(
      [
        'model:',
        '  default: x',
        'desktop:',
        '  font_family: ""',
        '  electron_flags:',
        '    - --ozone-platform=x11',
        '    - "--js-flags=--expose-gc"',
        '  renderer_max_old_space_mb: 2048  # ceiling',
        'terminal:',
        '  electron_flags: [--not-desktop]'
      ].join('\n')
    )

    expect(cfg).toEqual({ electronFlags: ['--ozone-platform=x11', '--js-flags=--expose-gc'], rendererMaxOldSpaceMb: 2048 })
    expect(planLaunchSwitches(cfg)).toEqual([
      { name: 'ozone-platform', value: 'x11' },
      { name: 'js-flags', value: '--expose-gc --max-old-space-size=2048' }
    ])
    // Default 0 = Chromium's own heap limit, nothing applied.
    expect(planLaunchSwitches(readDesktopLaunchConfig('desktop:\n  renderer_max_old_space_mb: 0\n'))).toEqual([])
  })

  it('merges with a js-flags switch already on argv instead of overwriting it', () => {
    const cfg = readDesktopLaunchConfig('desktop:\n  electron_flags: [--disable-gpu]\n  renderer_max_old_space_mb: 1536\n')
    const planned = planLaunchSwitches(cfg, ['/app/hermes', '--js-flags=--expose-gc', '--disable-gpu'])

    // --disable-gpu is already on the launcher's argv: not re-applied.
    expect(planned).toEqual([{ name: 'js-flags', value: '--expose-gc --max-old-space-size=1536' }])

    // Space-separated is the same switch: the argv value must reach the merge
    // (and win the ceiling) rather than being scanned as a switch of its own.
    expect(planLaunchSwitches(cfg, ['/app/hermes', '--js-flags', '--max-old-space-size=4096'])).toEqual([
      { name: 'disable-gpu' },
      { name: 'js-flags', value: '--max-old-space-size=4096' }
    ])
  })

  it('warns once when a desktop block names the keys in an indentation it cannot read', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    // Valid YAML, unsupported subset (four-space keys / two-space list items).
    expect(readDesktopLaunchConfig('desktop:\n    renderer_max_old_space_mb: 2048\n')).toEqual({
      electronFlags: [],
      rendererMaxOldSpaceMb: 0
    })
    expect(warn).toHaveBeenCalledTimes(1)

    // A desktop block that never mentions them is not a misconfiguration.
    readDesktopLaunchConfig('desktop:\n  font_family: ""\n')
    expect(warn).toHaveBeenCalledTimes(1)

    warn.mockRestore()
  })
})
