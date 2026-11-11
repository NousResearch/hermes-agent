import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'

import electron from 'electron'
import { build } from 'esbuild'

// Uses Electron's native API, not CDP, and never attaches to the user's window.
// Run from apps/desktop with the renderer already serving on port 5174.
test('shared tab close slots never overlap in RTL or LTR', { timeout: 90_000 }, async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'pane-tab-layout-'))
  const origin = process.env.HERMES_DESKTOP_DEV_SERVER ?? 'http://127.0.0.1:5174'
  const response = await fetch(`${origin}/src/styles.css?direct`)
  assert.equal(response.ok, true, 'running renderer must serve the actual stylesheet')
  const css = (await response.text()).replace(/url\((['"]?)\//g, `url($1${origin}/`)
  const bundle = await build({
    stdin: {
      contents: `import React from 'react';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import { PaneTab, PaneTabLabel, PaneTabStrip, PaneStripGlyph } from './src/components/ui/pane-tab';
import { ContextMenu, ContextMenuTrigger } from './src/components/ui/context-menu';
const root = createRoot(document.getElementById('root'));
window.renderTabs = ({ direction, width, count, tabWidth, decorated }) => {
  document.documentElement.dir = direction;
  document.getElementById('root').style.width = width + 'px';
  const tool = (label) => <PaneStripGlyph icon={<span>+</span>} label={label} onSelect={() => {}} />;
  flushSync(() => root.render(<PaneTabStrip trailing={<>{tool('Split')}{tool('Menu')}{tool('Scroll')}{tool('Overflow')}</>}>
    {Array.from({length: count}, (_, index) => <ContextMenu key={index}><ContextMenuTrigger asChild>
      <PaneTab role="tab" tabIndex={0} active={index === 0} dirty={decorated} onClose={() => {}} style={{width: tabWidth}}>
        {decorated && <span data-neighbour="icon" style={{flexShrink: 0}}>◆</span>}
        <PaneTabLabel>{index % 2 ? 'عنوان محادثة طويل جداً للاختبار' : 'Long English tab title that must ellipsize'}</PaneTabLabel>
        {decorated && <span data-neighbour="badge" style={{flexShrink: 0}}>9</span>}
      </PaneTab>
    </ContextMenuTrigger></ContextMenu>)}
    <span style={{display: 'flex', flexShrink: 0}}>{tool('New tab')}</span>
  </PaneTabStrip>));
};
window.measureTabs = () => {
  const failures = [];
  const rect = el => el.getBoundingClientRect();
  const gap = (a, b) => Math.max(b.left - a.right, a.left - b.right);
  const tabs = [...document.querySelectorAll('[data-slot="pane-tab"]')];
  const boxes = tabs.map(tab => {
    const close = tab.querySelector('button[aria-label="common.close"]');
    const c = rect(close), t = rect(tab);
    if (c.width !== 24 || c.height !== 24) failures.push('close target is not 24px');
    const inset = document.documentElement.dir === 'rtl' ? t.right - c.right : c.left - t.left;
    if (inset < 4 || inset > 5) failures.push('close is not at inline-start');
    for (const other of tab.querySelectorAll('.pane-tab-content, [data-neighbour], [data-slot="pane-tab-dirty"]')) {
      if (gap(c, rect(other)) < 4 - 0.1) failures.push('close crowds ' + other.outerHTML);
    }
    return [c.x, c.y, c.width, c.height, t.width];
  });
  const list = document.querySelector('[role="tablist"]');
  const l = rect(list);
  const siblings = [...list.children];
  for (let i = 1; i < siblings.length; i++) {
    if (gap(rect(siblings[i - 1]), rect(siblings[i])) < 3.9) failures.push('tab/control gap lost');
  }
  for (const control of document.querySelectorAll('button[aria-label]:not([aria-label="common.close"])')) {
    for (const tab of tabs) {
      const c = rect(tab.querySelector('button[aria-label="common.close"]'));
      // Offscreen tab pixels are clipped by the real scroller, not interactive.
      const visible = {left: Math.max(c.left, l.left), right: Math.min(c.right, l.right)};
      if (visible.right > visible.left && gap(visible, rect(control)) < 3.9) failures.push('close crowds strip control');
    }
  }
  return { failures, boxes, overflowing: list.scrollWidth > list.clientWidth };
};`,
      loader: 'tsx',
      resolveDir: process.cwd()
    },
    alias: { '@': path.resolve('src') },
    bundle: true,
    format: 'iife',
    loader: { '.css': 'empty' },
    define: { 'process.env.NODE_ENV': '"development"' },
    // Translation and tooltip stores are outside the layout contract.
    plugins: [
      {
        name: 'layout-only-services',
        setup(builder) {
          builder.onResolve({ filter: /^@\/(i18n|components\/ui\/tooltip)$/ }, args => ({
            path: args.path,
            namespace: 'fixture'
          }))
          builder.onLoad({ filter: /.*/, namespace: 'fixture' }, args => ({
            contents:
              args.path === '@/i18n'
                ? 'export const translateNow = key => key; export const useI18n = () => ({t: {}})'
                : 'export const Tip = ({children}) => children; export const TipHintLabel = () => null',
            loader: 'js'
          }))
        }
      }
    ],
    write: false
  })
  await writeFile(path.join(temporary, 'style.css'), css)
  await writeFile(path.join(temporary, 'fixture.js'), bundle.outputFiles[0].text)
  await writeFile(
    path.join(temporary, 'index.html'),
    '<html><head><link rel="stylesheet" href="style.css"></head><body><div id="root"></div><script src="fixture.js"></script></body></html>'
  )
  await writeFile(
    path.join(temporary, 'main.cjs'),
    `
const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(async () => {
  const window = new BrowserWindow({show: false, width: 1400, height: 500, webPreferences: {sandbox: true, contextIsolation: true, backgroundThrottling: false}});
  try {
    await window.loadFile(${JSON.stringify(path.join(temporary, 'index.html'))});
    console.log('Tab fixture loaded');
    let scenarios = 0, overflowScenarios = 0;
    for (const direction of ['rtl', 'ltr']) for (const width of [160, 320, 1280])
      for (const count of [1, 16]) for (const tabWidth of [52, 104, 192]) {
        const options = {direction, width, count, tabWidth, decorated: tabWidth !== 52};
        await window.webContents.executeJavaScript('window.renderTabs(' + JSON.stringify(options) + ')');
        const before = await window.webContents.executeJavaScript('window.measureTabs()');
        if (before.failures.length) throw Error(JSON.stringify({options, ...before}));
        const [x, y] = before.boxes[0];
        window.webContents.sendInputEvent({type: 'mouseMove', x: Math.round(x + 12), y: Math.round(y + 12)});
        await window.webContents.executeJavaScript(${JSON.stringify('document.querySelector(\'[role="tab"]\').focus()')});
        const after = await window.webContents.executeJavaScript('window.measureTabs()');
        if (after.failures.length || JSON.stringify(before.boxes) !== JSON.stringify(after.boxes)) throw Error('hover/focus changed layout: ' + JSON.stringify(options));
        for (const scroll of [37, 100000]) {
          await window.webContents.executeJavaScript(${JSON.stringify('document.querySelector(\'[role="tablist"]\').scrollLeft = ')} + (direction === 'rtl' ? -scroll : scroll));
          const scrolled = await window.webContents.executeJavaScript('window.measureTabs()');
          if (scrolled.failures.length) throw Error(JSON.stringify({options, scroll, ...scrolled}));
        }
        scenarios++; if (before.overflowing) overflowScenarios++;
      }
    const screenshots = [];
    for (const direction of ['rtl', 'ltr']) {
      await window.webContents.executeJavaScript('window.renderTabs(' + JSON.stringify({direction, width: 1280, count: 4, tabWidth: 192, decorated: true}) + ')');
      await window.webContents.executeJavaScript(${JSON.stringify('document.querySelector(\'[role="tab"]\').focus()')});
      await window.webContents.executeJavaScript('document.fonts.ready');
      await new Promise(resolve => setTimeout(resolve, 200));
      const screenshot = require('path').join(${JSON.stringify(temporary)}, direction + '.png');
      const image = await window.capturePage(undefined, {stayHidden: true});
      require('fs').writeFileSync(screenshot, image.toPNG());
      screenshots.push(screenshot);
    }
    process.stdout.write('TAB_LAYOUT_RESULT=' + JSON.stringify({scenarios, overflowScenarios, screenshots}) + '\\n');
    app.exit(0);
  } catch (error) { console.error(error); app.exit(1); }
});`
  )
  const result = await new Promise((resolve, reject) => {
    const child = spawn(electron, [path.join(temporary, 'main.cjs')], {
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '' }
    })
    let output = ''
    const timer = setTimeout(() => child.kill('SIGTERM'), 60_000)
    child.stdout.on('data', chunk => {
      output += chunk
      process.stdout.write(chunk)
    })
    child.stderr.on('data', chunk => {
      output += chunk
      process.stderr.write(chunk)
    })
    child.on('error', reject)
    child.on('close', code => {
      clearTimeout(timer)
      resolve({ code, output })
    })
  })
  assert.equal(result.code, 0, result.output)
  const match = result.output.match(/TAB_LAYOUT_RESULT=(.+)/)
  assert.ok(match, result.output)
  const summary = JSON.parse(match[1])
  assert.equal(summary.scenarios, 36)
  assert.ok(summary.overflowScenarios > 0)
  console.log(summary)
})
