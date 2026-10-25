# Mission control

Bundled Hermes Desktop plugin, **off by default**. Enable **Mission control** in
Capabilities → Plugins to add its pane to the workspace dock (right edge by
default; drag, tab, or minimize it like any pane afterwards).

## What it shows

- **Sessions** — every recent stored session, most-recent first, with the ones
  mid-turn ("working") separated on top behind a pulse dot. Rows carry the
  model, message count and a live age; clicking one opens it
  (`host.openSession`, `intent: 'stack'`), including gateway / cron sessions
  the sidebar doesn't front.
- **Cron** — the job roster with the next three runs (local clock time), status
  dots for error and paused jobs, and the gateway line (`gateway_running` plus
  the renderer's socket state).

Fresh data arrives on a 10–20 s poll, refreshed instantly on gateway events
(`sessions.changed`, `message.complete`, `session.title`, `cron.changed`,
`gateway.ready`), and the ↻ chip refetches everything on demand.

## Plugin boundary

`plugin.js` imports only `@hermes/plugin-sdk`, `react`, and
`react/jsx-runtime`. Existing bundled discovery finds it automatically;
`defaultEnabled: false` uses the ordinary live enable toggle. No shell,
SDK, backend, dependency, or registry changes are needed. All reads go through
three existing RPCs — `session.list`, `session.active_list`, and `cron.manage`
(list) — and nothing leaves the machine.

The same plain-ESM file can be loaded through the runtime plugin door at
`$HERMES_HOME/desktop-plugins/mission-control/plugin.js` for development. Do not
install that duplicate alongside the bundled version. Layout styles use Hermes
theme tokens, are scoped to `[data-mission-control]`, and are removed on
disable. Copy ships in `en`, `ja`, `zh`, and `zh-hant` via `ctx.i18n.register`;
the pane's tab follows the locale through `data.tabTitle`.

## Verification

`src/contrib/mission-control-plugin.test.tsx` (the Radio test's home — plugin
trees import only the SDK) drives the real bundled-discovery path: it
inventories off by
default, enables/disables through the ordinary toggle (pane contribution and
styles appear and disappear), then renders the registered pane against mocked
`session.list` / `session.active_list` / `cron.manage` responses and asserts the
roster, the working/recent split, the cron footer, and click-to-open.

```bash
cd apps/desktop && npx vitest run --project ui src/contrib/mission-control-plugin.test.tsx
```
