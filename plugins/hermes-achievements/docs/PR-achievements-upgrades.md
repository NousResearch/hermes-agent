# Achievements Upgrades

> PR for `nousresearch/hermes-agent` — enhancements to the `hermes-achievements` plugin

## Summary

The achievements plugin currently lives entirely inside the web dashboard. There is no way to query, filter, sort, or export achievements from the CLI or from within a Hermes agent session. Completed achievements vanish from the default view, evidence is recorded but never surfaced meaningfully, and `hermes curator` has no dry-run mode for safely previewing plugin operations before they mutate state. This PR addresses four capability gaps.

---

## 1. Filtering by Completion State & Evidence Ordering

### Problem

The `/achievements` endpoint returns every achievement in a flat list. The dashboard JS renders all states (unlocked, discovered, secret) in one stream with no server-side filtering or ordering. The `state.json` file records per-achievement `unlocked_at` timestamps and `evidence` blobs, but:

- There is no way to request **only completed** achievements (e.g. `?state=unlocked`)
- There is no way to **sort by evidence depth** — how much session history contributed to an achievement
- The `/recent-unlocks` endpoint is the only filtering path, and it only does reverse-chronological top-20

### Proposed Changes

**Backend (`plugin_api.py`)**

Add query parameters to `GET /achievements`:

```
GET /api/plugins/hermes-achievements/achievements?state=unlocked&sort_by=evidence&order=desc
```

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `state` | `unlocked`, `discovered`, `secret`, `all` | `all` | Filter by achievement state |
| `category` | any category string, e.g. `Debugging Chaos` | all | Filter by category |
| `sort_by` | `name`, `tier`, `progress`, `evidence`, `unlocked_at` | `name` | Sort key |
| `order` | `asc`, `desc` | `asc` (desc for `evidence` and `unlocked_at`) | Sort direction |
| `limit` | integer | no limit | Cap results |

**Evidence depth metric:**

For tiered achievements, evidence depth = the raw `progress` value (the underlying counter). For multi-condition achievements, evidence depth = the sum of per-requirement fulfillment percentages. This gives a meaningful "how much of this achievement's lifecycle have you traversed" number that makes sorting by evidence useful — you see your deepest commitments first.

**Dashboard frontend (`dist/index.js`)**

Add a filter bar above the achievement grid:
- Dropdown: All / Unlocked / Discovered / Secret
- Dropdown: All Categories / Agent Autonomy / Debugging Chaos / ... (populated from `achievements` response)
- Sort selector: Name / Tier / Evidence / Progress / Recently Unlocked

### Implementation Spec

**Step 1: Refactor `/achievements` handler to accept query params**

File: `plugins/hermes-achievements/dashboard/plugin_api.py`

Current signature (line ~991):
```python
@router.get("/achievements")
async def achievements():
```

New signature:
```python
@router.get("/achievements")
async def achievements(
    state: Optional[str] = Query(None, regex=r"^(unlocked|discovered|secret|all)$"),
    category: Optional[str] = None,
    sort_by: Optional[str] = Query(None, regex=r"^(name|tier|progress|evidence|unlocked_at)$"),
    order: Optional[str] = Query(None, regex=r"^(asc|desc)$"),
    limit: Optional[int] = Query(None, ge=1),
):
```

Note: FastAPI `Query()` with regex validation keeps the param surface safe. If the dashboard is running without FastAPI (the `APIRouter` stub at lines 16-23), the stub's `.get()` decorator just swallows the params — no breakage.

**Step 2: Extract filtering logic into a pure function**

Add to `plugin_api.py` after `display_achievement()`:

```python
def filter_and_sort_achievements(
    items: List[Dict[str, Any]],
    state: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: Optional[str] = None,
    order: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pure function: filter and sort evaluated achievements.
    
    Extracted so the CLI handler and the API endpoint share the same logic
    without duplicating filter/sort code.
    """
    result = list(items)
    
    # Filter by state
    if state and state != "all":
        result = [a for a in result if a.get("state") == state]
    
    # Filter by category
    if category:
        result = [a for a in result if a.get("category") == category]
    
    # Evidence depth: raw progress for tiered, pct for multi-condition
    def evidence_depth(a: Dict[str, Any]) -> int:
        if a.get("tier") or a.get("kind") == "best_session" or a.get("kind") == "lifetime":
            return int(a.get("progress", 0))
        return int(a.get("progress_pct", 0))
    
    # Sort
    key = sort_by or "name"
    # Default direction: desc for evidence and unlocked_at, asc otherwise
    if order:
        reverse = order == "desc"
    elif key in ("evidence", "unlocked_at"):
        reverse = True
    else:
        reverse = False
    
    if key == "evidence":
        result.sort(key=evidence_depth, reverse=reverse)
    elif key == "unlocked_at":
        result.sort(key=lambda a: a.get("unlocked_at") or 0, reverse=reverse)
    elif key == "tier":
        tier_order = {"Olympian": 5, "Diamond": 4, "Gold": 3, "Silver": 2, "Copper": 1}
        result.sort(key=lambda a: tier_order.get(a.get("tier"), 0), reverse=reverse)
    elif key == "progress":
        result.sort(key=lambda a: a.get("progress_pct", 0), reverse=reverse)
    else:
        result.sort(key=lambda a: a.get("name", ""), reverse=reverse)
    
    if limit:
        result = result[:limit]
    
    return result
```

**Step 3: Wire into the endpoint**

Replace the current `/achievements` handler body:

```python
@router.get("/achievements")
async def achievements(
    state: Optional[str] = Query(None, regex=r"^(unlocked|discovered|secret|all)$"),
    category: Optional[str] = None,
    sort_by: Optional[str] = Query(None, regex=r"^(name|tier|progress|evidence|unlocked_at)$"),
    order: Optional[str] = Query(None, regex=r"^(asc|desc)$"),
    limit: Optional[int] = Query(None, ge=1),
):
    data = evaluate_all()
    items = filter_and_sort_achievements(
        data.get("achievements", []),
        state=state, category=category,
        sort_by=sort_by, order=order, limit=limit,
    )
    payload = {
        "achievements": items,
        "unlocked_count": data.get("unlocked_count", 0),
        "discovered_count": data.get("discovered_count", 0),
        "secret_count": data.get("secret_count", 0),
        "total_count": data.get("total_count", 0),
        "filtered_count": len(items),
        "error": data.get("error"),
        "generated_at": data.get("generated_at"),
        "is_stale": _is_snapshot_stale(data),
        "scan_meta": {**(data.get("scan_meta") or {}), "status": _scan_status_payload()},
    }
    return payload
```

**Step 4: Add filter bar to dashboard frontend**

File: `plugins/hermes-achievements/dashboard/dist/index.js`

The current `AchievementsPage` component fetches from `/api/plugins/hermes-achievements/achievements` with no params and renders the full list. Patch:

1. Add state at the top of the component:
```javascript
const [filterState, setFilterState] = React.useState("all");
const [filterCategory, setFilterCategory] = React.useState(null);
const [sortBy, setSortBy] = React.useState("name");
```

2. Build query string from state and pass to `api()`:
```javascript
const params = new URLSearchParams();
if (filterState !== "all") params.set("state", filterState);
if (filterCategory) params.set("category", filterCategory);
if (sortBy !== "name") params.set("sort_by", sortBy);
const url = "/achievements" + (params.toString() ? "?" + params.toString() : "");
```

3. Render a filter bar above the grid using SDK components:
```javascript
React.createElement(C.Card, { className: "ha-filter-bar" },
  React.createElement(C.Select, { value: filterState, onValueChange: setFilterState, options: [
    { value: "all", label: "All States" },
    { value: "unlocked", label: "Unlocked" },
    { value: "discovered", label: "Discovered" },
    { value: "secret", label: "Secret" },
  ]}),
  React.createElement(C.Select, { value: sortBy, onValueChange: setSortBy, options: [
    { value: "name", label: "Name" },
    { value: "evidence", label: "Evidence" },
    { value: "tier", label: "Tier" },
    { value: "progress", label: "Progress" },
    { value: "unlocked_at", label: "Recently Unlocked" },
  ]})
)
```

**Step 5: Add tests**

File: `plugins/hermes-achievements/tests/test_achievement_engine.py`

```python
def test_filter_by_state_returns_only_matching_achievements(self):
    data = plugin_api.compute_all()
    all_items = data["achievements"]
    
    unlocked = plugin_api.filter_and_sort_achievements(all_items, state="unlocked")
    self.assertTrue(all(a["state"] == "unlocked" for a in unlocked))
    self.assertLessEqual(len(unlocked), len(all_items))
    
    discovered = plugin_api.filter_and_sort_achievements(all_items, state="discovered")
    self.assertTrue(all(a["state"] == "discovered" for a in discovered))

def test_sort_by_evidence_orders_by_progress_depth(self):
    data = plugin_api.compute_all()
    items = plugin_api.filter_and_sort_achievements(
        data["achievements"], sort_by="evidence", order="desc"
    )
    if len(items) >= 2:
        # Evidence depth should be non-increasing
        for i in range(len(items) - 1):
            self.assertGreaterEqual(
                (items[i].get("progress") or items[i].get("progress_pct", 0)),
                (items[i+1].get("progress") or items[i+1].get("progress_pct", 0)),
            )

def test_sort_by_tier_orders_highest_first(self):
    data = plugin_api.compute_all()
    items = plugin_api.filter_and_sort_achievements(
        data["achievements"], state="unlocked", sort_by="tier", order="desc"
    )
    tier_order = {"Olympian": 5, "Diamond": 4, "Gold": 3, "Silver": 2, "Copper": 1}
    if len(items) >= 2:
        for i in range(len(items) - 1):
            self.assertGreaterEqual(
                tier_order.get(items[i].get("tier"), 0),
                tier_order.get(items[i+1].get("tier"), 0),
            )

def test_limit_caps_results(self):
    data = plugin_api.compute_all()
    items = plugin_api.filter_and_sort_achievements(data["achievements"], limit=3)
    self.assertLessEqual(len(items), 3)

def test_filter_by_category(self):
    data = plugin_api.compute_all()
    items = plugin_api.filter_and_sort_achievements(
        data["achievements"], category="Debugging Chaos"
    )
    self.assertTrue(all(a.get("category") == "Debugging Chaos" for a in items))
```

---

## 2. TUI & Command-Line Interface

### Problem

The achievements plugin is currently web-dashboard-only. There is no CLI access. Users who live in the terminal (the core Hermes demographic) cannot check their badges, see progress toward an achievement, or review what sessions contributed evidence — without opening a browser. The plugin does not register any slash commands and has no `hermes achievements` subcommand.

### Proposed Changes

**New CLI subcommand: `hermes achievements`**

```
hermes achievements                          # summary: X/60 unlocked, top 5 by tier
hermes achievements list                     # all achievements, tabular
hermes achievements list --state unlocked    # only completed
hermes achievements list --category "Debugging Chaos"
hermes achievements list --sort evidence --limit 10
hermes achievements show <id>                # full detail: tiers, evidence, contributing session
hermes achievements rescan                   # force a rescan
hermes achievements export [--format json|markdown|svg]  # see section 3
```

**New slash command: `/achievements`** with aliases `/ach`, `/badges`

**TUI panel** in the Ink TUI with category-grouped card view, keyboard navigation, and evidence drill-down.

### Implementation Spec

**Step 1: Create `hermes_cli/achievements_cmd.py`**

This is the command handler for both `hermes achievements` (CLI) and `/achievements` (slash command). It imports the engine directly — no HTTP needed.

```python
"""Hermes achievements CLI and slash-command handler.

Reuses the achievement engine from the bundled hermes-achievements plugin
without requiring the dashboard to be running.
"""
from __future__ import annotations

import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress as RichProgress
    from rich.text import Text
except ImportError:
    Console = None  # type: ignore

# Import the engine. The plugin is bundled, so this should always resolve.
try:
    # Add plugin path if needed
    import importlib.util
    from pathlib import Path
    
    _PLUGIN_DIR = Path.home() / ".hermes" / "hermes-agent" / "plugins" / "hermes-achievements" / "dashboard"
    _spec = importlib.util.spec_from_file_location("plugin_api", _PLUGIN_DIR / "plugin_api.py")
    _api = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_api)
except Exception:
    _api = None


TIER_COLORS = {
    "Copper": "#B87333",
    "Silver": "#C0C0C0",
    "Gold": "#FFD700",
    "Diamond": "#B9F2FF",
    "Olympian": "#FF00FF",
}

TIER_ORDER = {"Olympian": 5, "Diamond": 4, "Gold": 3, "Silver": 2, "Copper": 1}


def _tier_bar(tier: Optional[str], pct: int, width: int = 10) -> str:
    """Build a unicode progress bar with tier coloring hint."""
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled) + f" {pct}%"


def achievements_summary():
    """Show a compact summary panel."""
    if not _api:
        print("Achievements plugin not found. Install or enable hermes-achievements first.")
        return
    
    data = _api.evaluate_all()
    unlocked = data.get("unlocked_count", 0)
    total = data.get("total_count", 0)
    discovered = data.get("discovered_count", 0)
    secret = data.get("secret_count", 0)
    
    if Console:
        console = Console()
        console.print(Panel(
            f"[bold]{unlocked}/{total}[/] unlocked  |  "
            f"{discovered} discovered  |  "
            f"{secret} secret",
            title="Hermes Achievements",
            border_style="bright_blue",
        ))
        
        # Top 5 unlocked by tier
        unlocked_items = [a for a in data.get("achievements", []) if a.get("unlocked")]
        unlocked_items.sort(key=lambda a: TIER_ORDER.get(a.get("tier"), 0), reverse=True)
        for item in unlocked_items[:5]:
            tier = item.get("tier", "?")
            name = item.get("name", "?")
            pct = item.get("progress_pct", 0)
            color = TIER_COLORS.get(tier, "white")
            console.print(f"  [{color}]{tier}[/{color}]  {name}  {_tier_bar(tier, pct)}")
    else:
        print(f"Achievements: {unlocked}/{total} unlocked | {discovered} discovered | {secret} secret")


def achievements_list(
    state: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: Optional[str] = None,
    order: Optional[str] = None,
    limit: Optional[int] = None,
):
    """List achievements in a table."""
    if not _api:
        print("Achievements plugin not found.")
        return
    
    data = _api.evaluate_all()
    items = _api.filter_and_sort_achievements(
        data.get("achievements", []),
        state=state, category=category,
        sort_by=sort_by, order=order, limit=limit,
    )
    unlocked_count = data.get("unlocked_count", 0)
    total = data.get("total_count", 0)
    filtered = len(items)
    
    if Console:
        console = Console()
        table = Table(title=f"Achievements ({unlocked_count}/{total} unlocked, {filtered} shown)")
        table.add_column("Name", style="bold")
        table.add_column("Tier")
        table.add_column("Progress")
        table.add_column("Category")
        
        for item in items:
            name = item.get("name", "???")
            tier = item.get("tier", "-")
            pct = item.get("progress_pct", 0)
            cat = item.get("category", "")
            color = TIER_COLORS.get(tier, "dim")
            table.add_row(name, f"[{color}]{tier}[/{color}]", _tier_bar(tier, pct), cat)
        
        console.print(table)
    else:
        for item in items:
            name = item.get("name", "???")
            tier = item.get("tier", "-")
            pct = item.get("progress_pct", 0)
            cat = item.get("category", "")
            print(f"{name:30s} {tier:10s} {pct:3d}%  {cat}")


def achievements_show(achievement_id: str):
    """Show full detail for one achievement."""
    if not _api:
        print("Achievements plugin not found.")
        return
    
    data = _api.evaluate_all()
    item = next((a for a in data.get("achievements", []) if a.get("id") == achievement_id), None)
    if not item:
        print(f"Achievement '{achievement_id}' not found.")
        return
    
    if Console:
        console = Console()
        tier = item.get("tier", "-")
        color = TIER_COLORS.get(tier, "white")
        
        console.print(Panel(
            f"[bold]{item.get('name', '???')}[/]\n\n"
            f"{item.get('description', '')}\n\n"
            f"[{color}]Tier: {tier}[/{color}]  |  "
            f"Progress: {item.get('progress_pct', 0)}%  |  "
            f"State: {item.get('state', '')}  |  "
            f"Category: {item.get('category', '')}",
            subtitle=item.get("id", ""),
        ))
        
        # Evidence
        evidence = item.get("evidence")
        if evidence:
            console.print(f"\n  Evidence: session [cyan]{evidence.get('session_id', '')}[/] "
                         f"({evidence.get('title', '')}) — value: {evidence.get('value', '')}")
        
        # Criteria
        criteria = item.get("criteria", "")
        if criteria:
            console.print(f"\n  {criteria}")
        
        # Tier ladder
        tiers = item.get("tiers", [])
        if tiers:
            console.print("\n  Tier ladder:")
            for t in tiers:
                marker = " ✓" if TIER_ORDER.get(t["name"], 0) <= TIER_ORDER.get(tier, 0) and tier else ""
                console.print(f"    {t['name']}: {t['threshold']}{marker}")
    else:
        print(f"{item.get('name', '???')}")
        print(f"  {item.get('description', '')}")
        print(f"  Tier: {tier} | Progress: {item.get('progress_pct', 0)}% | State: {item.get('state', '')}")


def achievements_rescan():
    """Force a rescan."""
    if not _api:
        print("Achievements plugin not found.")
        return
    data = _api.evaluate_all(force=True)
    print(f"Rescan complete: {data.get('unlocked_count', 0)}/{data.get('total_count', 0)} unlocked")


def handle_achievements_command(args_str: str):
    """Entry point for /achievements slash command and hermes achievements subcommand."""
    parts = (args_str or "").strip().split()
    
    if not parts or parts[0] in ("summary", ""):
        achievements_summary()
        return
    
    sub = parts[0]
    
    if sub == "list":
        # Parse flags: --state X --category X --sort X --order X --limit N
        kwargs = {}
        i = 1
        while i < len(parts):
            if parts[i] == "--state" and i + 1 < len(parts):
                kwargs["state"] = parts[i + 1]; i += 2
            elif parts[i] == "--category" and i + 1 < len(parts):
                kwargs["category"] = parts[i + 1]; i += 2
            elif parts[i] in ("--sort", "--sort-by") and i + 1 < len(parts):
                kwargs["sort_by"] = parts[i + 1]; i += 2
            elif parts[i] == "--order" and i + 1 < len(parts):
                kwargs["order"] = parts[i + 1]; i += 2
            elif parts[i] == "--limit" and i + 1 < len(parts):
                kwargs["limit"] = int(parts[i + 1]); i += 2
            else:
                i += 1
        achievements_list(**kwargs)
    
    elif sub == "show" and len(parts) > 1:
        achievements_show(parts[1])
    
    elif sub == "rescan":
        achievements_rescan()
    
    elif sub == "export":
        # Delegate to export handler (section 3)
        from hermes_cli.achievements_export import handle_export
        handle_export(parts[1:])
    
    else:
        print(f"Unknown subcommand: {sub}")
        print("Usage: achievements [summary|list|show <id>|rescan|export]")
```

**Step 2: Register the slash command**

File: `hermes_cli/commands.py`

Add to `COMMAND_REGISTRY`:

```python
CommandDef(
    name="achievements",
    aliases=["ach", "badges"],
    category="session",
    description="Show achievement progress, list badges, or rescan history",
    handler="hermes_cli.achievements_cmd.handle_achievements_command",
)
```

**Step 3: Register the `hermes achievements` subcommand**

File: `cli.py` (or wherever `hermes` top-level subcommands are dispatched — check `hermes_cli/` for the argparse/click/fire wiring)

Add a subcommand entry:

```python
# In the CLI subcommand dispatcher:
if command == "achievements":
    from hermes_cli.achievements_cmd import handle_achievements_command
    handle_achievements_command(" ".join(sys.argv[2:]))
```

**Step 4: TUI panel**

File: `ui-tui/src/components/AchievementsPanel.tsx` (new)

This is the biggest piece and the one maps expressed interest in helping build. Scope TBD — the Ink TUI already has a tab system and a JSON-RPC client to the dashboard backend. The panel can:

1. Fetch from `/api/plugins/hermes-achievements/achievements` on mount
2. Render a category-grouped list with tier-colored icons
3. Accept keyboard input to expand/collapse individual achievements
4. Show evidence detail on selection

This is a separate commit from the CLI work. The CLI and slash command ship first; the TUI panel follows.

---

## 3. Achievement Export & Agent Communication

### Problem

Achievements are trapped in `state.json` and the dashboard. There is no way to:
- Export achievements as a shareable format (JSON, markdown, SVG badge sheet)
- Communicate achievement state to Hermes agents during sessions
- Use achievement progress as input to other systems (skill recommendations, model routing, profile customization)

### Proposed Changes

**Export endpoint and CLI command:**

```
GET /api/plugins/hermes-achievements/export?format=json&state=unlocked
GET /api/plugins/hermes-achievements/export?format=markdown
GET /api/plugins/hermes-achievements/export?format=svg

hermes achievements export --format json
hermes achievements export --format markdown
hermes achievements export --format svg --output ~/badges.svg
```

**Formats:**

| Format | Description |
|--------|-------------|
| `json` | Full achievement data with evidence, tiers, timestamps. Machine-readable. |
| `markdown` | Human-readable summary: category tables, tier progress bars, evidence links. Pasteable into READMEs or docs. |
| `svg` | Badge sheet: one row per unlocked achievement, tier-colored shields with icons. Renderable in GitHub READMEs, personal sites, etc. |

**Agent communication:** Compact `/achievements/summary` endpoint + `agent_summary.json` context file.

### Implementation Spec

**Step 1: Add export formatters to `plugin_api.py`**

```python
def export_json(data: Dict[str, Any], state: Optional[str] = None) -> str:
    """Export achievements as structured JSON."""
    items = filter_and_sort_achievements(
        data.get("achievements", []),
        state=state,
    )
    export = {
        "generated_at": data.get("generated_at"),
        "unlocked_count": data.get("unlocked_count", 0),
        "total_count": data.get("total_count", 0),
        "achievements": items,
    }
    return json.dumps(export, indent=2, default=str)


def export_markdown(data: Dict[str, Any], state: Optional[str] = None) -> str:
    """Export achievements as markdown with progress bars and shields.io badges."""
    items = filter_and_sort_achievements(
        data.get("achievements", []),
        state=state or "unlocked",
    )
    unlocked = data.get("unlocked_count", 0)
    total = data.get("total_count", 0)
    from datetime import datetime
    scan_date = datetime.fromtimestamp(data.get("generated_at", 0)).strftime("%Y-%m-%d")
    
    lines = [
        f"# Hermes Achievements",
        f"",
        f"**{unlocked}/{total} unlocked** | Last scanned: {scan_date}",
        "",
    ]
    
    # Group by category
    categories: Dict[str, list] = {}
    for item in items:
        cat = item.get("category", "Other")
        categories.setdefault(cat, []).append(item)
    
    tier_colors = {
        "Copper": "CD7F32", "Silver": "C0C0C0", "Gold": "FFD700",
        "Diamond": "B9F2FF", "Olympian": "FF00FF",
    }
    
    for cat, cat_items in categories.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Achievement | Tier | Progress |")
        lines.append("|---|---|---|")
        for item in cat_items:
            name = item.get("name", "???")
            tier = item.get("tier", "-")
            pct = item.get("progress_pct", 0)
            color = tier_colors.get(tier, "gray")
            badge = f"![{tier}](https://img.shields.io/badge/{tier}-{pct}%25-{color})"
            bar_filled = int(pct / 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled) + f" {pct}%"
            lines.append(f"| {name} | {badge} | {bar} |")
        lines.append("")
    
    return "\n".join(lines)


def export_svg(data: Dict[str, Any], state: Optional[str] = None) -> str:
    """Export unlocked achievements as an SVG badge sheet."""
    items = filter_and_sort_achievements(
        data.get("achievements", []),
        state=state or "unlocked",
    )
    
    tier_colors = {
        "Copper": "#B87333", "Silver": "#C0C0C0", "Gold": "#FFD700",
        "Diamond": "#B9F2FF", "Olympian": "#FF00FF",
    }
    
    badge_w, badge_h, pad = 280, 28, 8
    rows = len(items)
    height = rows * (badge_h + pad) + pad
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{badge_w}" height="{height}" '
        f'viewBox="0 0 {badge_w} {height}">',
        f'<style>.badge{{rx:4;fill:#1a1a2e;stroke:#333;stroke-width:1}}'
        f'.name{{fill:#e0e0e0;font-family:monospace;font-size:11px}}'
        f'.tier{{font-family:monospace;font-size:10px;font-weight:bold}}</style>',
    ]
    
    for i, item in enumerate(items):
        y = i * (badge_h + pad) + pad
        name = item.get("name", "???")[:24]
        tier = item.get("tier", "-")
        color = tier_colors.get(tier, "#666")
        svg_parts.append(
            f'<rect class="badge" x="0" y="{y}" width="{badge_w}" height="{badge_h}"/>'
            f'<circle cx="14" cy="{y + badge_h // 2}" r="5" fill="{color}"/>'
            f'<text class="name" x="24" y="{y + 18}">{name}</text>'
            f'<text class="tier" x="{badge_w - 60}" y="{y + 18}" fill="{color}">{tier}</text>'
        )
    
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
```

**Step 2: Add `/export` endpoint**

```python
@router.get("/export")
async def export_achievements(
    format: str = Query("json", regex=r"^(json|markdown|svg)$"),
    state: Optional[str] = None,
):
    data = evaluate_all()
    if format == "markdown":
        content = export_markdown(data, state=state)
        return PlainTextResponse(content, media_type="text/markdown")
    elif format == "svg":
        content = export_svg(data, state=state)
        return PlainTextResponse(content, media_type="image/svg+xml")
    else:
        content = export_json(data, state=state)
        return JSONResponse(json.loads(content))
```

Note: need `from fastapi.responses import PlainTextResponse, JSONResponse` at the top of the file (or stub them in the no-FastAPI fallback).

**Step 3: Add `/achievements/summary` endpoint for agent communication**

```python
@router.get("/achievements/summary")
async def achievements_summary_for_agents():
    """Compact achievement profile for agent context injection.
    
    Designed to be small enough to fit in context_files without eating
    token budget. Generated once per rescan and cached to disk.
    """
    data = evaluate_all()
    items = data.get("achievements", [])
    aggregate = data.get("aggregate", {})
    
    # Determine strengths: categories where user has unlocked achievements
    cat_unlocks: Dict[str, int] = {}
    for item in items:
        if item.get("unlocked"):
            cat = item.get("category", "Other")
            cat_unlocks[cat] = cat_unlocks.get(cat, 0) + 1
    
    strengths = sorted(cat_unlocks, key=cat_unlocks.get, reverse=True)[:5]
    
    # Determine gaps: categories with zero unlocks but high discovery progress
    cat_progress: Dict[str, float] = {}
    for item in items:
        if not item.get("unlocked"):
            cat = item.get("category", "Other")
            cat_progress[cat] = max(cat_progress.get(cat, 0), item.get("progress_pct", 0))
    gaps = sorted(cat_progress, key=cat_progress.get, reverse=True)[:3]
    
    # Top tier across all unlocked
    top_tier = None
    for item in items:
        if item.get("unlocked") and item.get("tier"):
            if not top_tier or TIER_ORDER.get(item["tier"], 0) > TIER_ORDER.get(top_tier, 0):
                top_tier = item["tier"]
    
    summary = {
        "total_sessions": aggregate.get("session_count", 0),
        "total_tool_calls": aggregate.get("total_tool_calls", 0),
        "unlocked_count": data.get("unlocked_count", 0),
        "total_count": data.get("total_count", 0),
        "top_categories": strengths,
        "top_tier": top_tier,
        "strengths": strengths,
        "gaps": gaps,
        "unlocked_ids": [a["id"] for a in items if a.get("unlocked")],
    }
    return summary
```

**Step 4: Write `agent_summary.json` on rescan**

Add to `_run_scan_and_update_cache()`, after `_SNAPSHOT_CACHE` is set:

```python
# Also write agent_summary.json for context_files injection
try:
    summary = _build_agent_summary(computed)
    summary_path = Path.home() / ".hermes" / "plugins" / "hermes-achievements" / "agent_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
except Exception:
    pass  # Non-critical: summary is best-effort
```

And the helper:

```python
def _build_agent_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build compact agent-consumable summary from evaluated data."""
    items = data.get("achievements", [])
    aggregate = data.get("aggregate", {})
    
    cat_unlocks: Dict[str, int] = {}
    for item in items:
        if item.get("unlocked"):
            cat = item.get("category", "Other")
            cat_unlocks[cat] = cat_unlocks.get(cat, 0) + 1
    
    strengths = sorted(cat_unlocks, key=cat_unlocks.get, reverse=True)[:5]
    
    cat_progress: Dict[str, float] = {}
    for item in items:
        if not item.get("unlocked"):
            cat = item.get("category", "Other")
            cat_progress[cat] = max(cat_progress.get(cat, 0), item.get("progress_pct", 0))
    gaps = sorted(cat_progress, key=cat_progress.get, reverse=True)[:3]
    
    top_tier = None
    for item in items:
        if item.get("unlocked") and item.get("tier"):
            if not top_tier or TIER_ORDER.get(item["tier"], 0) > TIER_ORDER.get(top_tier, 0):
                top_tier = item["tier"]
    
    return {
        "total_sessions": aggregate.get("session_count", 0),
        "total_tool_calls": aggregate.get("total_tool_calls", 0),
        "unlocked_count": data.get("unlocked_count", 0),
        "total_count": data.get("total_count", 0),
        "top_categories": strengths,
        "top_tier": top_tier,
        "strengths": strengths,
        "gaps": gaps,
        "unlocked_ids": [a["id"] for a in items if a.get("unlocked")],
    }
```

**Step 5: Create `hermes_cli/achievements_export.py`**

CLI handler for the `export` subcommand:

```python
"""Export handler for hermes achievements export."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def handle_export(args: list[str]):
    """Handle: hermes achievements export [--format json|markdown|svg] [--output PATH] [--state X]"""
    fmt = "json"
    output = None
    state = None
    
    i = 0
    while i < len(args):
        if args[i] in ("--format", "-f") and i + 1 < len(args):
            fmt = args[i + 1]; i += 2
        elif args[i] in ("--output", "-o") and i + 1 < len(args):
            output = args[i + 1]; i += 2
        elif args[i] == "--state" and i + 1 < len(args):
            state = args[i + 1]; i += 2
        else:
            i += 1
    
    # Import engine
    try:
        import importlib.util
        _PLUGIN_DIR = Path.home() / ".hermes" / "hermes-agent" / "plugins" / "hermes-achievements" / "dashboard"
        _spec = importlib.util.spec_from_file_location("plugin_api", _PLUGIN_DIR / "plugin_api.py")
        _api = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_api)
    except Exception as exc:
        print(f"Failed to load achievements engine: {exc}")
        sys.exit(1)
    
    data = _api.evaluate_all()
    
    if fmt == "markdown":
        content = _api.export_markdown(data, state=state)
    elif fmt == "svg":
        content = _api.export_svg(data, state=state)
    else:
        content = _api.export_json(data, state=state)
    
    if output:
        Path(output).write_text(content)
        print(f"Exported {fmt} to {output}")
    else:
        print(content)
```

**Step 6: Add export tests**

```python
def test_export_json_produces_valid_json(self):
    data = plugin_api.compute_all()
    result = plugin_api.export_json(data)
    parsed = json.loads(result)
    self.assertIn("achievements", parsed)
    self.assertIn("unlocked_count", parsed)

def test_export_markdown_produces_category_headers(self):
    data = plugin_api.compute_all()
    result = plugin_api.export_markdown(data, state="unlocked")
    self.assertIn("## ", result)

def test_export_svg_produces_valid_svg(self):
    data = plugin_api.compute_all()
    result = plugin_api.export_svg(data, state="unlocked")
    self.assertIn("<svg", result)
    self.assertIn("</svg>", result)

def test_agent_summary_has_strengths_and_gaps(self):
    data = plugin_api.compute_all()
    summary = plugin_api._build_agent_summary(data)
    self.assertIn("strengths", summary)
    self.assertIn("gaps", summary)
    self.assertIn("unlocked_ids", summary)
```

---

## 4. Dry-Run Mode for `hermes curator`

### Problem

`hermes curator` manages plugin lifecycle — install, update, enable, disable, remove. It currently has no dry-run mode. Any curator operation that modifies plugin state (updating an achievement catalog, resetting unlock state, toggling a plugin) takes effect immediately. For the achievements plugin specifically, a `curator update` could change achievement definitions (new badges, renamed IDs, shifted thresholds), which silently mutates unlock state. A bad update could reset progress or orphan unlocks.

More broadly, every Hermes plugin that `curator` manages has the same risk: you can't preview what a curator operation will do before it does it. There is no `--dry-run` flag, no diff output, no "here's what would change" step.

### Proposed Changes

**Add `--dry-run` flag to `hermes curator`:**

```
hermes curator update hermes-achievements --dry-run
hermes curator enable songseed-harvester --dry-run
hermes curator remove hermes-achievements --dry-run
hermes curator rescan --dry-run
```

When `--dry-run` is passed, the curator precomputes the operation without executing, prints a structured preview, and exits without modifying state.

### Implementation Spec

**Step 1: Add `--dry-run` argument to curator parser**

File: `hermes_cli/plugins_cmd.py`

Find the argparse/click argument definitions for the curator subcommands. Add a `--dry-run` flag to each mutating subcommand (install, update, enable, disable, remove):

```python
# For each mutating subcommand's argument parser:
parser.add_argument("--dry-run", action="store_true",
                    help="Preview changes without applying them")
```

**Step 2: Implement dry-run path for file operations**

The core idea: every curator operation boils down to a sequence of file writes, config edits, and state transitions. A dry-run computes the same sequence but intercepts each mutation, records it, and skips the actual write.

Add a `DryRunRecorder` class:

```python
class DryRunRecorder:
    """Records planned mutations without executing them."""
    
    def __init__(self):
        self.file_ops: list[dict] = []      # {op: "write"/"delete"/"rename", path, size, ...}
        self.config_ops: list[dict] = []    # {op: "set"/"remove", key, old, new}
        self.state_ops: list[dict] = []     # {op: "enable"/"disable"/"install"/"remove", plugin, ...}
        self.warnings: list[str] = []       # things that look risky
    
    def record_file_write(self, path: str, content_size: int, exists: bool):
        self.file_ops.append({"op": "write", "path": path, "size": content_size, "exists": exists})
    
    def record_file_delete(self, path: str, exists: bool):
        self.file_ops.append({"op": "delete", "path": path, "exists": exists})
    
    def record_config_change(self, key: str, old_value, new_value):
        self.config_ops.append({"op": "set", "key": key, "old": old_value, "new": new_value})
    
    def record_config_remove(self, key: str, old_value):
        self.config_ops.append({"op": "remove", "key": key, "old": old_value})
    
    def record_state_transition(self, plugin: str, old_state: str, new_state: str):
        self.state_ops.append({"op": new_state, "plugin": plugin, "from": old_state})
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def has_breaking_changes(self) -> bool:
        """True if any operation looks destructive (deleting files, removing config, etc)."""
        return any(op["op"] == "delete" and op.get("exists") for op in self.file_ops) or \
               any(op["op"] == "remove" for op in self.config_ops) or \
               any(op["op"] == "remove" for op in self.state_ops)
    
    def format_report(self) -> str:
        """Human-readable summary of planned changes."""
        lines = []
        if self.file_ops:
            lines.append("File changes:")
            for op in self.file_ops:
                if op["op"] == "write":
                    marker = "overwrite" if op.get("exists") else "create"
                    lines.append(f"  [{marker}] {op['path']} ({op['size']} bytes)")
                elif op["op"] == "delete":
                    lines.append(f"  [delete] {op['path']}")
        
        if self.config_ops:
            lines.append("Config changes:")
            for op in self.config_ops:
                if op["op"] == "set":
                    lines.append(f"  {op['key']}: {op['old']} → {op['new']}")
                elif op["op"] == "remove":
                    lines.append(f"  {op['key']}: {op['old']} → (removed)")
        
        if self.state_ops:
            lines.append("Plugin state:")
            for op in self.state_ops:
                lines.append(f"  {op['plugin']}: {op['from']} → {op['op']}")
        
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Machine-readable diff for --dry-run --format json."""
        return {
            "file_ops": self.file_ops,
            "config_ops": self.config_ops,
            "state_ops": self.state_ops,
            "warnings": self.warnings,
            "has_breaking_changes": self.has_breaking_changes(),
        }
```

**Step 3: Wire dry-run into each curator subcommand**

The pattern is the same for each: if `--dry-run` is set, create a `DryRunRecorder`, pass it through the operation logic instead of actually writing, then print the report and exit.

Sketch for `curator update`:

```python
def handle_curator_update(args):
    plugin_name = args.plugin
    dry_run = getattr(args, "dry_run", False)
    
    if dry_run:
        recorder = DryRunRecorder()
        # Resolve target version, download/simulate extraction
        # For each file that would be written:
        #   recorder.record_file_write(path, size, exists=path.exists())
        # For each config key that would change:
        #   recorder.record_config_change(key, old_value, new_value)
        # Check for plugin-specific impact via dry_run_preview() hook
        _check_plugin_dry_run_preview(plugin_name, recorder)
        
        if getattr(args, "format", None) == "json":
            print(json.dumps(recorder.to_dict(), indent=2))
        else:
            print(recorder.format_report())
        
        sys.exit(1 if recorder.has_breaking_changes() else 0)
    else:
        # Existing update logic
        ...
```

Same pattern for install, enable, disable, remove.

**Step 4: Plugin `dry_run_preview()` hook**

Plugins can optionally define a `dry_run_preview(proposed_version, current_state)` function in their `plugin_api.py`:

```python
def _check_plugin_dry_run_preview(plugin_name: str, recorder: DryRunRecorder):
    """Call the plugin's dry_run_preview hook if it exists."""
    try:
        # Load the plugin's API module
        api = _load_plugin_api(plugin_name)
        if api and hasattr(api, "dry_run_preview"):
            preview = api.dry_run_preview(
                proposed_version=_resolve_target_version(plugin_name),
                current_state=_load_plugin_state(plugin_name),
            )
            # Add any warnings from the plugin
            for warning in preview.get("warnings", []):
                recorder.add_warning(warning)
    except Exception:
        pass  # Optional hook — failure is non-fatal
```

For the achievements plugin, implement `dry_run_preview()`:

```python
def dry_run_preview(proposed_version: str, current_state: dict) -> dict:
    """Preview what a version update would do to achievement state.
    
    Called by hermes curator --dry-run. Compares current achievement
    definitions against the proposed version and reports unlock impacts.
    """
    current_ids = {a["id"] for a in ACHIEVEMENTS}
    current_unlocks = current_state.get("unlocks", {})
    
    # We can't load the proposed version's definitions without actually
    # downloading them, so we report what we *can* know: current state
    # that would be at risk.
    warnings = []
    
    # Check for unlocked achievements whose IDs don't appear in current catalog
    orphaned = [uid for uid in current_unlocks if uid not in current_ids]
    if orphaned:
        warnings.append(f"{len(orphaned)} unlock(s) reference achievement IDs no longer in catalog: {orphaned[:5]}")
    
    return {
        "current_achievement_count": len(ACHIEVEMENTS),
        "current_unlock_count": len(current_unlocks),
        "orphaned_unlock_ids": orphaned,
        "warnings": warnings,
    }
```

Note: A full implementation would also diff the proposed `ACHIEVEMENTS` list against the current one (new IDs, removed IDs, threshold changes). That requires downloading the proposed version's `plugin_api.py` and parsing its `ACHIEVEMENTS` list. This can be done as a follow-up — the initial implementation reports what it can from current state.

**Step 5: Exit codes**

| Code | Meaning |
|------|---------|
| 0 | Dry-run completed, no breaking changes detected |
| 1 | Dry-run detected a breaking change (would lose data, remove config, delete files) |
| 2 | Dry-run failed to compute preview |

**Step 6: Add dry-run tests**

```python
# In hermes-cli test suite (not the plugin test suite — this tests curator)

def test_curator_update_dry_run_does_not_modify_files(self):
    """Verify --dry-run produces a report without writing anything."""
    # Setup: snapshot the plugin directory mtime/size
    # Run: hermes curator update hermes-achievements --dry-run
    # Assert: no files changed, exit code 0 or 1, report printed

def test_curator_remove_dry_run_reports_breaking(self):
    """Verify --dry-run on 'remove' detects breaking change."""
    # Run: hermes curator remove hermes-achievements --dry-run
    # Assert: exit code 1, report mentions file deletions

def test_achievements_dry_run_preview_reports_orphans(self):
    """Verify the plugin hook detects orphaned unlock IDs."""
    import plugin_api
    state = {"unlocks": {"fake_id_1": {"unlocked_at": 1}, "let_him_cook": {"unlocked_at": 2}}}
    result = plugin_api.dry_run_preview("0.4.0", state)
    self.assertIn("fake_id_1", result["orphaned_unlock_ids"])
    self.assertTrue(any("orphan" in w.lower() for w in result["warnings"]))
```

---

## Implementation Order

1. **Filtering & sorting** (backend only, no frontend dependency) — smallest scope, highest value for the API
2. **Export** (endpoint + CLI) — builds on filtering, enables agent communication
3. **CLI & TUI** (slash command + `hermes achievements` subcommand + TUI panel) — largest surface area, depends on 1 and 2
4. **Curator dry-run** (cross-cutting, not achievements-specific) — independent of 1-3 but benefits from the export format for diff output

Each phase is a separate commit with its own tests. The TUI panel is a separate PR that depends on phases 1-3 landing first.

---

## Files Changed

| File | Change |
|------|--------|
| `plugins/hermes-achievements/dashboard/plugin_api.py` | Add `filter_and_sort_achievements()`, query params on `/achievements`, add `/export` endpoint, add `/achievements/summary` endpoint, add `_build_agent_summary()`, add `export_json/markdown/svg()`, add `dry_run_preview()` hook |
| `plugins/hermes-achievements/dashboard/manifest.json` | Bump version to `0.4.0` |
| `plugins/hermes-achievements/tests/test_achievement_engine.py` | Add tests for filtering, sorting, export formats, evidence depth, agent summary, dry-run preview |
| `hermes_cli/commands.py` | Register `/achievements` slash command |
| `hermes_cli/achievements_cmd.py` | New file: `hermes achievements` subcommand handler + Rich output |
| `hermes_cli/achievements_export.py` | New file: export subcommand handler |
| `hermes_cli/plugins_cmd.py` | Add `--dry-run` flag + `DryRunRecorder` to curator subcommands |
| `ui-tui/src/components/AchievementsPanel.tsx` | New file: TUI panel (separate PR after 1-3 land) |

---

## Non-Goals

- **Achievement syncing across machines** — out of scope, would require a backend service
- **Social/sharing features** — the export formats enable this, but no built-in social layer
- **Custom achievement authoring** — users can already add to `ACHIEVEMENTS` list in `plugin_api.py`; a proper authoring UI is a separate PR
- **Real-time unlock notifications** — the dashboard already surfaces recent unlocks; push notifications are a separate concern
