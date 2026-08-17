"""Browser-level responsive geometry checks for the Kanban dashboard plugin."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def layout_fixture(tmp_path: Path) -> Path:
    stylesheet = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "dist"
        / "style.css"
    ).as_uri()
    page = tmp_path / "layout.html"
    page.write_text(
        f"""<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<link rel="stylesheet" href="{stylesheet}">
<style>
* {{ box-sizing: border-box }}
html, body {{ margin: 0; width: 100%; background: #111; color: #eee }}
.hermes-kanban {{ padding: 16px }}
.hermes-kanban-toolbar {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end }}
.hermes-kanban-toolbar-search input {{ width: 224px }}
.hermes-kanban-columns {{ margin-top: 12px }}
.hermes-kanban-column {{ box-sizing: border-box }}
.hermes-kanban-drawer-shade {{ display: flex }}
</style>
<div class="hermes-kanban">
  <div class="hermes-kanban-toolbar">
    <div class="hermes-kanban-toolbar-search"><input></div>
    <div class="hermes-kanban-toolbar-filter"><select><option>all</option></select></div>
    <button>Refresh</button><button>Clear filters</button>
  </div>
  <nav class="hermes-kanban-column-nav" aria-label="Kanban lanes">
    <button class="hermes-kanban-column-nav-item hermes-kanban-column-nav-item--active">Triage<span class="hermes-kanban-column-nav-count">20</span></button>
    <button class="hermes-kanban-column-nav-item">Todo<span class="hermes-kanban-column-nav-count">0</span></button>
    <button class="hermes-kanban-column-nav-item">Scheduled<span class="hermes-kanban-column-nav-count">1</span></button>
    <button class="hermes-kanban-column-nav-item">Ready<span class="hermes-kanban-column-nav-count">2</span></button>
    <button class="hermes-kanban-column-nav-item">Running<span class="hermes-kanban-column-nav-count">3</span></button>
    <button class="hermes-kanban-column-nav-item">Blocked<span class="hermes-kanban-column-nav-count">1</span></button>
    <button class="hermes-kanban-column-nav-item">Review<span class="hermes-kanban-column-nav-count">4</span></button>
    <button class="hermes-kanban-column-nav-item">Done<span class="hermes-kanban-column-nav-count">9</span></button>
  </nav>
  <div class="hermes-kanban-columns">
    <section class="hermes-kanban-column"><button class="hermes-kanban-column-add">+</button></section>
    <section class="hermes-kanban-column"></section>
  </div>
</div>
<div class="hermes-kanban-drawer-shade"><aside class="hermes-kanban-drawer"><button class="hermes-kanban-drawer-close">x</button></aside></div>
<pre id="result"></pre>
<script>
addEventListener('load', () => {{
  const rect = s => document.querySelector(s).getBoundingClientRect();
  const rail = document.querySelector('.hermes-kanban-columns');
  const lane = rect('.hermes-kanban-column');
  const drawer = rect('.hermes-kanban-drawer');
  const input = rect('.hermes-kanban-toolbar-search input');
  const add = rect('.hermes-kanban-column-add');
  const close = rect('.hermes-kanban-drawer-close');
  const nav = rect('.hermes-kanban-column-nav');
  const navItems = [...document.querySelectorAll('.hermes-kanban-column-nav-item')];
  document.querySelector('#result').textContent = JSON.stringify({{
    viewport: innerWidth,
    bodyScrollWidth: document.body.scrollWidth,
    laneWidth: lane.width,
    drawerWidth: drawer.width,
    inputWidth: input.width,
    addWidth: add.width,
    addHeight: add.height,
    closeWidth: close.width,
    closeHeight: close.height,
    navHeight: nav.height,
    navLabels: navItems.map(item => item.firstChild.textContent),
    navTargets: navItems.map(item => item.getBoundingClientRect().height),
    railOverflow: rail.scrollWidth > rail.clientWidth,
    snap: getComputedStyle(rail).scrollSnapType,
  }});
}});
</script>""",
        encoding="utf-8",
    )
    return page


def _measure(page: Path, width: int) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    playwright_candidates = [
        repo_root / "node_modules" / "playwright",
        repo_root.parent.parent / "node_modules" / "playwright",
    ]
    playwright = next((path for path in playwright_candidates if path.exists()), None)
    if playwright is None:
        pytest.skip("Playwright dependencies are required for browser geometry coverage")
    probe = page.parent / "probe.cjs"
    probe.write_text(
        """const { chromium } = require(process.argv[2]);
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: '/usr/bin/chromium', args: ['--no-sandbox']});
  const context = await browser.newContext({viewport: {width: Number(process.argv[4]), height: 900}});
  const page = await context.newPage();
  await page.goto(process.argv[3]);
  const result = await page.locator('#result').textContent();
  process.stdout.write(result);
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(probe), str(playwright), page.as_uri(), str(width)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("width", [320, 360, 390, 430])
def test_mobile_board_uses_viewport_lanes_and_touch_targets(layout_fixture: Path, width: int):
    measured = _measure(layout_fixture, width)

    assert measured["viewport"] == width
    assert measured["bodyScrollWidth"] == width
    assert measured["laneWidth"] == pytest.approx(width - 32, abs=1)
    assert measured["drawerWidth"] == pytest.approx(width, abs=1)
    assert measured["inputWidth"] == pytest.approx(width - 32, abs=1)
    assert measured["addWidth"] >= 44
    assert measured["addHeight"] >= 44
    assert measured["closeWidth"] >= 44
    assert measured["closeHeight"] >= 44
    assert measured["navLabels"] == ["Triage", "Todo", "Scheduled", "Ready", "Running", "Blocked", "Review", "Done"]
    assert all(height >= 44 for height in measured["navTargets"])
    assert measured["navHeight"] >= 88
    assert measured["railOverflow"] is True
    assert measured["snap"] == "x mandatory"


def test_desktop_keeps_compact_columns_and_disables_snap(layout_fixture: Path):
    measured = _measure(layout_fixture, 1280)

    assert measured["bodyScrollWidth"] == 1280
    assert measured["laneWidth"] == pytest.approx(280, abs=1)
    assert measured["drawerWidth"] == pytest.approx(640, abs=1)
    assert measured["navHeight"] == 0
    assert measured["snap"] == "none"
