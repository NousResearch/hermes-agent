"""Browser contracts for the real Kanban dashboard plugin application."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


LANES = ["Triage", "Todo", "Ready", "In Progress", "Blocked", "Review", "Done", "scheduled"]
MOBILE_ACCEPTANCE_WIDTHS = (320, 375, 390, 430)


def test_mobile_acceptance_widths_match_issue_contract():
    assert MOBILE_ACCEPTANCE_WIDTHS == (320, 375, 390, 430)


def _task(task_id: str, status: str, *, assignee: str = "alpha", tenant: str = "acme") -> dict[str, Any]:
    return {
        "id": task_id,
        "title": f"{status} task {task_id}",
        "body": "Deterministic privacy-safe fixture",
        "status": status,
        "assignee": assignee,
        "tenant": tenant,
        "priority": 1,
        "parents": [],
        "children": [],
        "created_at": 1,
        "updated_at": 1,
    }


@pytest.fixture()
def app_fixture(tmp_path: Path) -> Path:
    """Bundle React into a tiny host, then execute the shipped plugin unchanged."""
    repo = Path(__file__).resolve().parents[2]
    node_modules = repo / "node_modules"
    playwright = node_modules / "playwright"
    esbuild = node_modules / "esbuild" / "bin" / "esbuild"
    if not playwright.exists() or not esbuild.exists():
        pytest.skip("Playwright and esbuild are required for the Kanban browser contract")

    statuses = ["triage", "todo", "ready", "running", "blocked", "review", "done", "scheduled"]
    columns = []
    for status in statuses:
        tasks = [] if status == "todo" else [_task(f"{status}-1", status)]
        if status == "triage":
            tasks = [_task(f"triage-{i}", status) for i in range(20)]
        if status == "running":
            tasks = [
                _task("running-alpha", status, assignee="alpha"),
                _task("running-beta", status, assignee="beta"),
            ]
        if status == "review":
            tasks = [_task("review-beta-other", status, assignee="beta", tenant="other")]
        columns.append({"name": status, "tasks": tasks})
    board = {
        "columns": columns,
        "tenants": ["acme", "other"],
        "assignees": ["alpha", "beta"],
        "latest_event_id": 0,
    }

    entry = tmp_path / "entry.jsx"
    entry.write_text(
        """import React from 'react';
import { createRoot } from 'react-dom/client';
const pass = (tag) => React.forwardRef(({children, ...props}, ref) => React.createElement(tag, {...props, ref}, children));
const Select = React.forwardRef(({children, onValueChange, ...props}, ref) =>
  React.createElement('select', {...props, ref, onChange: e => onValueChange ? onValueChange(e.target.value) : props.onChange?.(e)}, children));
const Checkbox = React.forwardRef(({onCheckedChange, ...props}, ref) =>
  React.createElement('input', {...props, ref, type: 'checkbox', onChange: e => onCheckedChange?.(e.target.checked)}));
window.__HERMES_PLUGIN_SDK__ = {
  React,
  hooks: React,
  components: {Card: pass('div'), CardContent: pass('div'), Badge: pass('span'), Button: pass('button'),
    Input: pass('input'), Label: pass('label'), Select, SelectOption: pass('option'), Checkbox},
  utils: {cn: (...xs) => xs.filter(Boolean).join(' '), timeAgo: () => 'now'},
  fetchJSON: async url => {
    if (url.includes('/config')) return {lane_by_profile: true, render_markdown: true};
    if (url.includes('/boards')) return {boards: [{slug: 'default', name: 'Default'}], current: 'default'};
    if (url.includes('/board')) return window.__BOARD_FIXTURE__;
    if (url.includes('/tasks/')) {
      const id = decodeURIComponent(url.split('/tasks/')[1].split('?')[0]);
      const task = window.__BOARD_FIXTURE__.columns.flatMap(column => column.tasks).find(item => item.id === id);
      return {task, comments: [], events: [], attachments: []};
    }
    return {};
  },
  // Keep the event stream pending: deterministic fixtures do not need a socket,
  // and this avoids reconnect timers keeping Chromium alive after assertions.
  buildWsUrl: () => new Promise(() => {}),
  useI18n: () => ({t: {kanban: null}, locale: 'en'}),
};
window.__HERMES_PLUGINS__ = {register(_name, Component) { createRoot(document.querySelector('#root')).render(React.createElement(Component)); }};
""",
        encoding="utf-8",
    )
    bundle = tmp_path / "harness.js"
    subprocess.run(
        [str(esbuild), str(entry), "--bundle", "--format=iife", f"--outfile={bundle}"],
        check=True,
        cwd=repo,
        env={**os.environ, "NODE_PATH": str(node_modules)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    page = tmp_path / "index.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<link rel='stylesheet' href='{(repo / 'plugins/kanban/dashboard/dist/style.css').as_uri()}'>"
        "<style>*{box-sizing:border-box}html,body,#root{margin:0;width:100%;background:#111;color:#eee}"
        ".hermes-kanban{padding:16px}.hermes-kanban-toolbar{display:flex}.hermes-kanban-columns{margin-top:12px}</style>"
        f"<div id='root'></div><script>window.__BOARD_FIXTURE__={json.dumps(board)}</script>"
        f"<script src='{bundle.as_uri()}'></script>"
        f"<script src='{(repo / 'plugins/kanban/dashboard/dist/index.js').as_uri()}'></script>",
        encoding="utf-8",
    )
    return page


def _run_browser(
    page: Path,
    width: int,
    screenshot: Path | None = None,
    *,
    reduced_motion: bool = False,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    playwright = repo / "node_modules" / "playwright"
    browser_executable = next(
        (path for name in ("chromium", "chromium-browser", "google-chrome", "msedge") if (path := shutil.which(name))),
        "",
    )
    probe = page.parent / f"probe-{width}.cjs"
    probe.write_text(
        """const { chromium } = require(process.argv[2]);
(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless:true});
  } catch (error) {
    if (!String(error).includes("Executable doesn't exist")) throw error;
    if (!process.argv[7]) throw error;
    browser = await chromium.launch({headless:true, executablePath:process.argv[7]});
  }
  const context = await browser.newContext({
    viewport:{width:Number(process.argv[4]),height:900},
    hasTouch:true,
    reducedMotion:process.argv[6] === 'reduce' ? 'reduce' : 'no-preference'
  });
  await context.addInitScript(() => {
    window.__KANBAN_SCROLL_BEHAVIORS__ = [];
    const original = Element.prototype.scrollTo;
    Element.prototype.scrollTo = function (...args) {
      if (this.classList?.contains('hermes-kanban-columns')) {
        window.__KANBAN_SCROLL_BEHAVIORS__.push(args[0]?.behavior || null);
      }
      return original.apply(this, args);
    };
  });
  const page = await context.newPage();
  await page.goto(process.argv[3]);
  page.on('console', msg => console.error('browser:', msg.text()));
  page.on('pageerror', error => console.error('pageerror:', error.message));
  await page.locator('.hermes-kanban-column-nav-item').first().waitFor({state:'attached',timeout:5000});
  console.error('step: mounted');
  const labels = await page.locator('.hermes-kanban-column-nav-item').allTextContents();
  const visits = [];
  if (Number(process.argv[4]) <= 767) {
    for (const button of await page.locator('.hermes-kanban-column-nav-item').all()) {
      await button.tap(); await page.waitForTimeout(50);
      const controls = await button.getAttribute('aria-controls');
      visits.push({controls, current: await button.getAttribute('aria-current'),
        visibleCount: await page.locator('.hermes-kanban-column:visible').count(),
        visible: await page.locator('#' + controls).evaluate((el) => {
          const lane = el.getBoundingClientRect(); const rail = el.parentElement.getBoundingClientRect();
          return lane.left >= rail.left && lane.right <= rail.right + 1;
        })});
    }
    await page.locator('.hermes-kanban-column-nav-item').first().tap();
    await page.waitForTimeout(50);
  }
  console.error('step: lanes');
  const empty = await page.evaluate(() => document.querySelector('[data-kanban-column="todo"] .hermes-kanban-empty')?.textContent || null);
  console.error('step: empty');
  await page.evaluate(() => {
    const input = document.querySelector('.hermes-kanban-toolbar-search input');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    Reflect.apply(setter, input, ['running-alpha']); input.dispatchEvent(new Event('input', {bubbles:true}));
  });
  await page.waitForTimeout(50);
  const filteredCounts = await page.locator('.hermes-kanban-column-nav-count').allTextContents();
  await page.evaluate(() => {
    const input = document.querySelector('.hermes-kanban-toolbar-search input');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    Reflect.apply(setter, input, ['']); input.dispatchEvent(new Event('input', {bubbles:true}));
  });
  await page.waitForTimeout(50);
  const tenant = page.locator('.hermes-kanban-toolbar-filter').filter({hasText:'Tenant'}).locator('select');
  await tenant.selectOption('other');
  await page.waitForTimeout(50);
  const tenantFilteredCounts = await page.locator('.hermes-kanban-column-nav-count').allTextContents();
  const assignee = page.locator('.hermes-kanban-toolbar-filter').filter({hasText:'Assignee'}).locator('select');
  await assignee.selectOption('beta');
  await page.waitForTimeout(50);
  const tenantAssigneeFilteredCounts = await page.locator('.hermes-kanban-column-nav-count').allTextContents();
  await page.getByRole('button', {name:'Clear filters'}).click();
  await page.waitForTimeout(50);
  const clearedCounts = await page.locator('.hermes-kanban-column-nav-count').allTextContents();
  const clearedFilterValues = {tenant: await tenant.inputValue(), assignee: await assignee.inputValue()};
  console.error('step: filters');
  const groupedBefore = await page.locator('[data-kanban-column="running"] .hermes-kanban-lane').count();
  await page.evaluate(() => [...document.querySelectorAll('.hermes-kanban-toolbar-toggle')]
    .find(el => el.textContent.includes('Lanes by profile')).querySelector('input').click());
  console.error('step: grouping');
  const groupedAfter = await page.locator('[data-kanban-column="running"] .hermes-kanban-lane').count();
  const rects = xs => xs.map(x => {
    const r = x.getBoundingClientRect(); return {width:r.width,height:r.height};
  });
  const navTargets = await page.locator('.hermes-kanban-column-nav-item').evaluateAll(rects);
  const toolbarTargets = await page.locator('.hermes-kanban-toolbar input:not([type="checkbox"]), .hermes-kanban-toolbar select, .hermes-kanban-toolbar button, .hermes-kanban-toolbar-toggle').evaluateAll(rects);
  const addTargets = await page.locator('.hermes-kanban-column-add:visible').evaluateAll(rects);
  const triageBody = page.locator('[data-kanban-column="triage"] .hermes-kanban-column-body');
  const verticalScroll = await triageBody.evaluate(el => {
    const before = el.scrollTop; el.scrollTop = el.scrollHeight;
    return {before,after:el.scrollTop,clientHeight:el.clientHeight,scrollHeight:el.scrollHeight};
  });
  const opener = page.locator('[data-task-id="triage-0"]');
  await opener.focus(); await page.keyboard.press('Enter');
  const drawer = page.locator('.hermes-kanban-drawer');
  await drawer.waitFor({state:'visible'});
  await page.waitForTimeout(250);
  const drawerRect = await drawer.evaluate(el => {
    const r = el.getBoundingClientRect(); return {left:r.left,right:r.right,top:r.top,bottom:r.bottom};
  });
  const drawerCloseTarget = await page.locator('.hermes-kanban-drawer-close').evaluate(el => {
    const r = el.getBoundingClientRect(); return {width:r.width,height:r.height};
  });
  await page.locator('.hermes-kanban-drawer-close').click();
  await drawer.waitFor({state:'detached'});
  await page.waitForTimeout(50);
  const focusReturned = await opener.evaluate(el => document.activeElement === el);
  const laneWidth = await page.locator('.hermes-kanban-column').first().evaluate(el => el.getBoundingClientRect().width);
  const navHeight = await page.locator('.hermes-kanban-column-nav').evaluate(el => el.getBoundingClientRect().height);
  if (process.argv[5]) await page.screenshot({path:process.argv[5], fullPage:true});
  process.stdout.write(JSON.stringify({labels,visits,empty,filteredCounts,tenantFilteredCounts,tenantAssigneeFilteredCounts,
    clearedCounts,clearedFilterValues,groupedBefore,groupedAfter,navTargets,toolbarTargets,addTargets,
    verticalScroll,drawerRect,drawerCloseTarget,focusReturned,
    scrollBehaviors:await page.evaluate(() => window.__KANBAN_SCROLL_BEHAVIORS__),laneWidth,navHeight,
    bodyWidth:await page.evaluate(() => document.body.scrollWidth), viewport:await page.evaluate(() => innerWidth)}));
  await browser.close();
})().catch(e => {console.error(e);process.exit(1)});""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "node", str(probe), str(playwright), page.as_uri(), str(width),
            str(screenshot or ""), "reduce" if reduced_motion else "no-preference", browser_executable,
        ],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        pytest.fail(result.stderr)
    return json.loads(result.stdout)


@pytest.mark.parametrize("width", MOBILE_ACCEPTANCE_WIDTHS)
def test_real_app_exposes_and_activates_every_mobile_lane(app_fixture: Path, width: int):
    screenshot_dir = os.environ.get("KANBAN_SCREENSHOT_DIR")
    screenshot = Path(screenshot_dir) / f"kanban-mobile-{width}.png" if screenshot_dir else None
    measured = _run_browser(app_fixture, width, screenshot)
    assert measured["labels"] == ["Triage20", "Todo0", "Ready1", "In Progress2", "Blocked1", "Review1", "Done1", "scheduled1"]
    assert all(
        visit["current"] == "true" and visit["visible"] and visit["visibleCount"] == 1
        for visit in measured["visits"]
    ), measured["visits"]
    assert measured["empty"] == "— no tasks —"
    assert measured["filteredCounts"] == ["0", "0", "0", "1", "0", "0", "0", "0"]
    assert measured["tenantFilteredCounts"] == ["0", "0", "0", "0", "0", "1", "0", "0"]
    assert measured["tenantAssigneeFilteredCounts"] == ["0", "0", "0", "0", "0", "1", "0", "0"]
    assert measured["clearedCounts"] == ["20", "0", "1", "2", "1", "1", "1", "1"]
    assert measured["clearedFilterValues"] == {"tenant": "", "assignee": ""}
    assert measured["groupedBefore"] == 2
    assert measured["groupedAfter"] == 0
    required_targets = measured["navTargets"] + measured["toolbarTargets"] + measured["addTargets"]
    assert required_targets
    assert all(target["width"] >= 44 and target["height"] >= 44 for target in required_targets)
    assert measured["drawerCloseTarget"]["width"] >= 44
    assert measured["drawerCloseTarget"]["height"] >= 44
    assert measured["verticalScroll"]["scrollHeight"] > measured["verticalScroll"]["clientHeight"]
    assert measured["verticalScroll"]["after"] > measured["verticalScroll"]["before"]
    assert measured["drawerRect"]["left"] >= 0
    assert measured["drawerRect"]["right"] <= width
    assert measured["drawerRect"]["top"] >= 0
    assert measured["drawerRect"]["bottom"] <= 900
    assert measured["focusReturned"] is True
    assert measured["bodyWidth"] == width
    assert measured["laneWidth"] == pytest.approx(width - 32, abs=1)


def test_real_app_preserves_desktop_lane_geometry(app_fixture: Path):
    measured = _run_browser(app_fixture, 1280)
    assert measured["bodyWidth"] == 1280
    assert measured["laneWidth"] == pytest.approx(280, abs=1)
    assert measured["navHeight"] == 0


def test_real_app_honors_reduced_motion_for_lane_activation(app_fixture: Path):
    normal = _run_browser(app_fixture, 390)
    reduced = _run_browser(app_fixture, 390, reduced_motion=True)
    assert normal["scrollBehaviors"] == []
    assert reduced["scrollBehaviors"] == []
