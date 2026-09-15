/**
 * Hermes desktop plugin template. Save as:
 *   <hermes home>/desktop-plugins/<id>/plugin.js   (folder name == id)
 * where <hermes home> is ~/.hermes by default, or ~/.hermes/profiles/<name>
 * when running a named profile (`hermes -p <name>`). Run `hermes doctor` (or
 * check the app's Settings → Plugins folder path) if unsure which is active.
 * Then run "Reload desktop plugins" from ⌘K in the desktop app.
 *
 * Plain ESM, loaded uncompiled — UI is jsx() calls, not JSX syntax.
 * Only these imports resolve: @hermes/plugin-sdk, react, react/jsx-runtime.
 */

import { cn, haptic, host, Tip, usePluginI18n, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

// Ship your OWN strings (never edit core en.ts). `usePluginI18n` resolves them
// against the app's active locale, falling back to `en`, then the raw key.
const ID = 'my-plugin'

function MyPane() {
  const t = usePluginI18n(ID)
  const gateway = useValue(host.state.gateway)

  return jsxs('div', {
    className: 'flex h-full flex-col gap-2 p-3 text-sm',
    children: [
      jsx('div', { className: 'font-medium', children: t('paneTitle') }),
      jsx('div', {
        className: 'text-(--ui-text-tertiary)',
        children: t('gateway', gateway)
      })
    ]
  })
}

function MyChip() {
  const t = usePluginI18n(ID)

  return jsx(Tip, {
    label: t('chipTip'),
    children: jsx('button', {
      className: cn(
        'inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] transition-colors',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      ),
      type: 'button',
      onClick: () => {
        haptic('tap')
        host.notify({ kind: 'info', message: t('hello') })
      },
      children: 'my-plugin'
    })
  })
}

export default {
  id: ID, // must match the folder name
  name: 'My Plugin',
  register(ctx) {
    // Register locale bundles — scoped to this plugin, torn down on reload.
    // Values are literals or interpolators (`arg => `…${arg}…``).
    ctx.i18n.register({
      en: {
        paneTitle: 'My Plugin Pane',
        gateway: state => `gateway: ${state}`,
        chipTip: 'My plugin — click me',
        hello: 'Hello from my plugin!'
      },
      ja: {
        paneTitle: 'マイプラグイン',
        gateway: state => `ゲートウェイ: ${state}`,
        chipTip: 'マイプラグイン — クリック',
        hello: 'マイプラグインからこんにちは！'
      }
    })

    // A layout pane — auto-placed by the placement hint; user can drag it.
    // To land on a specific edge instead of stacking, add a dock gesture,
    // e.g. below the conversation:
    //   data: { placement: 'bottom', dock: { pane: 'workspace', pos: 'bottom' }, height: '200px' }
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'my plugin',
      data: { placement: 'right', width: '237px' },
      render: () => jsx(MyPane, {})
    })

    // A statusbar chip.
    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 130,
      render: () => jsx(MyChip, {})
    })
  }
}
