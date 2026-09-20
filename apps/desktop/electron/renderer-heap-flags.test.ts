// #77311: the renderer heap ceiling knob (`desktop.renderer_max_old_space_mb`)
// and `desktop.electron_flags` must reach app.commandLine on packaged launches
// that never go through the `hermes desktop` launcher.
import { describe, expect, it } from 'vitest'

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
  })
})
