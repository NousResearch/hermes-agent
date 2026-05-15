# Aviation Threat MCP-UI Server — Design Spec

**Status:** Approved for implementation planning
**Date:** 2026-05-07
**Author:** brainstormed with josh

## Summary

A standalone MCP server that exposes four tools for surfacing aviation security and threat data. Each tool returns an MCP-UI `rawHtml` resource — a fully self-contained HTML+CSS+JS fragment that renders inside the host's sandboxed iframe with a custom dark/cyan brand identity modeled after josh's existing internal portal.

The deliverable is a stakeholder-demo-grade prototype: deterministic, fixture-backed, visually polished. No live APIs, auth, or persistence.

## Goals

- Replicate the look-and-feel of the existing internal aviation portal (dark teal panels, cyan accents, letter-spaced uppercase labels, conic-gradient rating dials, status pins on dark maps) inside Claude chat.
- Demonstrate four hero scenarios that map cleanly to natural-language prompts:
  1. Route risk overlay — "Show me security risks along my flight from PHX to JFK"
  2. Airport deep-dive — "Tell me about LSZH"
  3. FIR detail — "What's happening in the Baghdad FIR?"
  4. Fleet status — "Are any planes in my fleet experiencing issues?"
- Run anywhere MCP-UI is consumed (Claude Desktop, web/browser chat, any MCP-UI-compliant host).
- Be runnable via a single `npx` command after install.

## Non-goals

- Live data integration — all data is bundled JSON fixtures.
- Auth, multi-tenancy, persistence, user accounts.
- User-editable state — fleet is a server-side fixture; flight routes come in via tool args only.
- Alert search/filter as its own tool (out for v1; can be added later).
- Mobile-responsive polish — desktop iframe only.
- Integration tests against a live MCP host.

## Architecture

A single Node.js / TypeScript process running over stdio MCP transport. Tool handlers read fixtures, do lightweight geo math, and return a UI resource composed from a shared template + theme.

```
┌──────────────────────┐   stdio    ┌───────────────────────────┐
│ Claude (Desktop/Web) │ ─────────► │ aviation-mcp-ui (Node/TS) │
│                      │            │  ├─ tools/ (4 handlers)   │
│   renders ui:// in   │ ◄───────── │  ├─ fixtures/ (JSON)      │
│   sandboxed iframe   │  ui://     │  ├─ templates/ (HTML)     │
└──────────────────────┘            │  └─ render.ts (compose)   │
                                    └───────────────────────────┘
```

Each tool returns:
1. A short text summary (so the LLM can narrate the result).
2. A single MCP-UI `EmbeddedResource` of the form:
   ```json
   {
     "type": "resource",
     "resource": {
       "uri": "ui://aviation/<view>/<id>",
       "mimeType": "text/html",
       "text": "<!DOCTYPE html>..."
     }
   }
   ```

## Tool Surface

| Tool | Args | Behavior | UI returned |
|---|---|---|---|
| `get_route_risks` | `origin: string` (ICAO), `dest: string` (ICAO) | Look up both airports, compute great-circle waypoints, find alerts within a corridor band, render a Leaflet map with route polyline + alert pins. | `route-map.html` |
| `get_airport` | `icao: string` | Look up airport, find any active alerts within 50 km, render the branded card. | `airport-card.html` |
| `get_fir` | `fir_id: string` | Look up FIR, find linked alerts, render the threat-detail panel with weaponry-range bar and flight-level risk band. | `fir-detail.html` |
| `get_fleet_status` | *(none)* | Render the full fixture fleet as a status grid; aircraft with `status: "issue"` show prominently. | `fleet-grid.html` |

**Error handling:** Unknown ICAO / FIR id returns a tool error with a helpful message ("LSZQ not found — try LSZH for Zurich"). No partial UI on missing data.

**HTML escaping:** All fixture-derived strings are HTML-escaped before substitution. Templates use `{{field}}` placeholders only — no logic, no eval.

## Data Model & Fixtures

Five JSON files in `fixtures/`. All IDs are real-world (ICAOs, FIR codes); only threat/incident content is fabricated.

### `airports.json` (~30 entries)

```ts
type Airport = {
  icao: string;          // "LSZH"
  iata: string;          // "ZRH"
  name: string;          // "Zurich"
  country: string;       // "Switzerland"
  lat: number;
  lng: number;
  rating: 1 | 2 | 3 | 4 | 5;       // 1 = unrestricted, 5 = severe
  rating_label: string;            // "Unrestricted Environment"
  covid_inbound: 1 | 2 | 3 | 4 | 5;
  covid_domestic: 1 | 2 | 3 | 4 | 5;
  city_risk: 1 | 2 | 3 | 4 | 5;
  city_risk_label: string;
  medical_risk: 1 | 2 | 3 | 4 | 5;
  medical_risk_label: string;
  runways: { id: string; length_ft: number; width_ft: number; elevation_ft: number; }[];
  ground_handling: { slots_required: boolean; handling_required: boolean; };
  date_of_publish: string;         // "2026-04-10"
};
```

Required coverage: KPHX, KJFK, LSZH, OEJN, EGLL, EDDF, LFPG, LIRF, LTBA, OIIE, ORBI, HECA, OEDF, KORD, KLAX, KDEN, KATL, KSFO, KMIA, KSEA, RJTT, VHHH, WSSS, OMDB, ZBAA, EHAM, LEMD, LSGG, EBBR, EPWA. (30 total — picked to support route demos and global coverage.)

### `firs.json` (~10 entries)

```ts
type FIR = {
  id: string;             // "ORBB" (Baghdad)
  name: string;           // "Baghdad FIR"
  country: string;
  polygon: [number, number][];     // GeoJSON-style ring [lng, lat][]
  weaponry_range_floor: number;    // FL260
  flight_level_floor: number;      // FL180
  flight_level_ceiling: number;    // FL600
  weapons: ("small_arms" | "aaa_light" | "aaa" | "manpads" | "manpads_advanced"
          | "rpg" | "atgm" | "sam" | "sam_mobile" | "sam_advanced")[];
  hostile_intercepts: boolean;
  cz_warnings: string[];
  issued_by: string;               // "Canada"
};
```

Required coverage: ORBB (Baghdad), OIIX (Tehran), LSAS (Switzerland), HECC (Cairo), EGTT (London), URRV (Rostov), LTBB (Ankara), OYSC (Sanaa), HKNA (Nairobi), UTAK (Almaty).

### `alerts.json` (~25 entries)

```ts
type Alert = {
  id: string;
  severity: "advisory" | "notice" | "warning" | "critical";
  category: "terrorism" | "security" | "medical" | "weather" | "police_operation";
  region: string;                  // "EUROPE & CIS"
  lat: number;
  lng: number;
  headline: string;                // "EXERCISE VIGILANCE DUE TO INCREASED RISK..."
  body: string;                    // multi-paragraph, can include simple HTML
  active_from: string;             // ISO timestamp
  active_to: string;
  linked_airport_icao?: string;    // optional FK
  linked_fir_id?: string;          // optional FK
  tags: string[];                  // ["TERRORISM", "POLICE/SECURITY OPERATION"]
};
```

Distribution: 3-4 critical, 6-8 warning, rest advisory/notice. Geographic spread to support route-risk demos (cluster along PHX→JFK and Europe→Middle East corridors).

### `fleet.json` (~8 entries)

```ts
type Aircraft = {
  tail_number: string;             // "N737AZ"
  type: string;                    // "B737-800"
  callsign: string;                // "OSPREY101"
  status: "in_flight" | "on_ground" | "maintenance" | "issue";
  location: { icao?: string; lat?: number; lng?: number; };
  origin?: string;                 // ICAO if in-flight
  dest?: string;                   // ICAO if in-flight
  issue?: { severity: "advisory" | "warning" | "critical"; headline: string; detail: string; };
};
```

Composition: 4 in-flight, 2 on-ground, 1 maintenance, 1 with active issue (e.g. "diverting due to medical emergency, currently 80nm SE of OEJN"). The issue aircraft is positioned near a fixture alert so the LLM can correlate.

### `routes.json` (precomputed helper)

```ts
type Route = {
  origin: string;       // ICAO
  dest: string;
  waypoints: [number, number][];   // [lng, lat][]
};
```

Pre-baked great-circle waypoints (10-15 points each) for ~6 demo pairs: PHX-JFK, JFK-LHR, LHR-DXB, DXB-SIN, FRA-DEL, LAX-NRT. Tool falls back to live great-circle calc for any other pair.

## UI Rendering

A shared `render.ts` module:

```ts
function renderTemplate(name: string, data: Record<string, unknown>): string;
function escapeHtml(value: string): string;
function buildResource(view: string, id: string, html: string): EmbeddedResource;
```

Substitution is mustache-style: `{{field}}` and `{{#section}}...{{/section}}` for arrays. Implemented inline (~40 lines) — no runtime template-engine dependency, keeps the iframe payload small.

### Templates

All templates inline `theme.css` and any view-specific JS.

| Template | Notes |
|---|---|
| `theme.css` | Design tokens. Single source of truth for color/typography. |
| `airport-card.html` | The LSZH-style card we mocked (header, alert strip, four rating dials with conic-gradient, side-tab nav, runway/handling rows). |
| `fir-detail.html` | The Baghdad-style panel — flight-level risk bar, weaponry-range bar, weapons grid (active vs greyed), CZ warnings, issued-by row. |
| `fleet-grid.html` | Responsive card grid. Each card: tail number, type, status dot (green/amber/red), location, optional issue blurb. The aircraft with `status: "issue"` gets a red glow + pulsing animation. |
| `route-map.html` | Full-iframe Leaflet map. CDN-loaded Leaflet 1.9, CartoDB dark tiles. Route polyline (cyan, 3px), origin/dest markers, alert pins styled by severity. Side panel (absolute-positioned overlay) lists alerts in order along the route. |

### Design tokens (`theme.css`)

```css
:root {
  --bg: #0b1419;
  --panel: #11202a;
  --border: #1a2a32;
  --accent: #3ddbd9;       /* cyan */
  --success: #3dd97a;
  --warning: #d9a83d;
  --danger: #d94d3d;
  --text: #d8e6ec;
  --text-muted: #7a99a8;
  --text-dim: #5b7a87;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
```

Common patterns: 11px letter-spaced (1-2px) uppercase labels in `--text-muted`; 13px body in `--text`; cyan left-border (3px) for active alerts; 1px borders in `--border` for panel divisions.

## Project Layout

```
aviation-mcp-ui/
├── package.json          # type: "module", bin: { "aviation-mcp-ui": "./dist/server.js" }
├── tsconfig.json
├── README.md             # install + Claude Desktop config + 4 demo prompts
├── src/
│   ├── server.ts         # MCP server bootstrap (StdioServerTransport, register 4 tools)
│   ├── tools/
│   │   ├── getAirport.ts
│   │   ├── getFir.ts
│   │   ├── getFleetStatus.ts
│   │   └── getRouteRisks.ts
│   ├── render.ts         # template loader + mustache-lite + escapeHtml + buildResource
│   ├── data.ts           # fixture loader, lookups (by ICAO/FIR/tail), corridor query
│   ├── geo.ts            # haversine, great-circle waypoints, point-near-polyline
│   └── types.ts
├── templates/            # bundled into dist/ at build
│   ├── theme.css
│   ├── airport-card.html
│   ├── fir-detail.html
│   ├── fleet-grid.html
│   └── route-map.html
├── fixtures/             # bundled into dist/ at build
│   ├── airports.json
│   ├── firs.json
│   ├── alerts.json
│   ├── fleet.json
│   └── routes.json
└── tests/
    ├── tools.test.ts
    ├── geo.test.ts
    └── render.test.ts
```

### Dependencies

Runtime:
- `@modelcontextprotocol/sdk` — MCP server primitives
- `@mcp-ui/server` — `createUIResource` helper for emitting `ui://` resources

Dev:
- `typescript`
- `vitest`
- `@types/node`

No runtime dependencies beyond the two MCP SDKs. Leaflet is CDN-loaded inside the route-map iframe (not a Node dep).

## Geo math (`geo.ts`)

- `haversine(a, b)` — great-circle distance in km
- `greatCircleWaypoints(a, b, n=12)` — interpolate `n` lat/lng points along the great-circle arc using slerp on unit vectors
- `pointNearPolyline(point, polyline, thresholdKm)` — true if `point` is within `thresholdKm` of any segment; used by `get_route_risks` to filter alerts onto the corridor (default 200 km)

## Testing

`vitest` for unit tests:

- **`tools.test.ts`** — for each tool: valid input returns a well-formed MCP-UI resource (correct uri scheme, mimeType, non-empty html); invalid input returns a tool error with a sensible message; one snapshot per tool of the rendered HTML for a fixed input (PHX/JFK route, LSZH airport, ORBB FIR, full fleet) to catch template regressions.
- **`geo.test.ts`** — haversine against known city-pair distances (±1%); great-circle waypoint count + endpoints; point-near-polyline correctness on a known PHX→JFK corridor.
- **`render.test.ts`** — escape-html on `<script>alert(1)</script>`-style inputs, missing-field handling, array section rendering.

`README.md` includes manual demo script: 4 prompts to paste into Claude Desktop after registering the server, each producing the screenshot-worthy output for one tool.

## Open questions / future work

These are explicitly out of v1 but worth noting:

- **Interactive flight-route input** via `postMessage` intent (iframe → host → tool roundtrip).
- **Live data adapters** — swap fixture loader for a real API client; structure of `data.ts` already isolates this.
- **Per-user fleet** — would need auth + persistence layer.
- **Alert search/filter tool** — `list_alerts(region?, severity?)`.
- **Mobile-responsive templates.**
