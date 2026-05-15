# Aviation Threat MCP-UI Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TypeScript MCP server that exposes 4 tools returning rawHtml MCP-UI resources rendering branded aviation threat dashboards (airport, FIR, fleet, route risk) for stakeholder demo use.

**Architecture:** Single Node process, stdio MCP transport. Each tool reads JSON fixtures, runs minimal logic (geo math for the route tool), and returns a self-contained HTML payload composed from a template + shared theme CSS. No runtime deps beyond `@modelcontextprotocol/sdk` and `@mcp-ui/server`. Pure-logic modules (geo, render, data) get TDD; templates and wiring get snapshot tests.

**Tech Stack:** TypeScript (ESM), Node 20+, `@modelcontextprotocol/sdk`, `@mcp-ui/server`, Leaflet (CDN, in-iframe only), vitest.

**Spec:** [docs/superpowers/specs/2026-05-07-aviation-mcp-ui-server-design.md](../specs/2026-05-07-aviation-mcp-ui-server-design.md)

---

## Project Layout

```
aviation-mcp-ui/
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── README.md
├── src/
│   ├── server.ts            # MCP bootstrap (entrypoint)
│   ├── tools/
│   │   ├── getAirport.ts
│   │   ├── getFir.ts
│   │   ├── getFleetStatus.ts
│   │   └── getRouteRisks.ts
│   ├── render.ts            # template substitution + escape + buildResource
│   ├── data.ts              # fixture loader + lookups + corridor query
│   ├── geo.ts               # haversine, greatCircleWaypoints, pointNearPolyline
│   └── types.ts
├── templates/
│   ├── theme.css
│   ├── airport-card.html
│   ├── fir-detail.html
│   ├── fleet-grid.html
│   └── route-map.html
├── fixtures/
│   ├── airports.json
│   ├── firs.json
│   ├── alerts.json
│   ├── fleet.json
│   └── routes.json
└── tests/
    ├── geo.test.ts
    ├── render.test.ts
    ├── data.test.ts
    └── tools.test.ts
```

The repo will be created **inside the current worktree** at `aviation-mcp-ui/` (subdirectory). All paths below are relative to the worktree root unless otherwise noted.

---

## Task 1: Scaffold project

**Files:**
- Create: `aviation-mcp-ui/package.json`
- Create: `aviation-mcp-ui/tsconfig.json`
- Create: `aviation-mcp-ui/vitest.config.ts`
- Create: `aviation-mcp-ui/.gitignore`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "aviation-mcp-ui",
  "version": "0.1.0",
  "description": "Aviation threat MCP-UI server (stakeholder demo)",
  "type": "module",
  "bin": {
    "aviation-mcp-ui": "./dist/server.js"
  },
  "main": "./dist/server.js",
  "files": ["dist", "templates", "fixtures", "README.md"],
  "scripts": {
    "build": "tsc",
    "start": "node dist/server.js",
    "dev": "tsc -w",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.4",
    "@mcp-ui/server": "^5.0.0"
  },
  "devDependencies": {
    "@types/node": "^22.10.0",
    "typescript": "^5.7.2",
    "vitest": "^2.1.8"
  },
  "engines": {
    "node": ">=20"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "declaration": false,
    "sourceMap": true
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 3: Create vitest.config.ts**

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
```

- [ ] **Step 4: Create .gitignore**

```
node_modules/
dist/
*.log
.DS_Store
```

- [ ] **Step 5: Install dependencies**

Run: `cd aviation-mcp-ui && npm install`
Expected: lockfile created, no errors. If `@mcp-ui/server` resolution fails, run `npm view @mcp-ui/server version` to confirm a version exists; pin to that version.

- [ ] **Step 6: Commit**

```bash
git add aviation-mcp-ui/package.json aviation-mcp-ui/tsconfig.json aviation-mcp-ui/vitest.config.ts aviation-mcp-ui/.gitignore aviation-mcp-ui/package-lock.json
git commit -m "scaffold aviation-mcp-ui project"
```

---

## Task 2: Types

**Files:**
- Create: `aviation-mcp-ui/src/types.ts`

- [ ] **Step 1: Write types.ts**

```ts
export type RatingLevel = 1 | 2 | 3 | 4 | 5;

export interface Runway {
  id: string;
  length_ft: number;
  width_ft: number;
  elevation_ft: number;
}

export interface Airport {
  icao: string;
  iata: string;
  name: string;
  country: string;
  lat: number;
  lng: number;
  rating: RatingLevel;
  rating_label: string;
  covid_inbound: RatingLevel;
  covid_domestic: RatingLevel;
  city_risk: RatingLevel;
  city_risk_label: string;
  medical_risk: RatingLevel;
  medical_risk_label: string;
  runways: Runway[];
  ground_handling: { slots_required: boolean; handling_required: boolean };
  date_of_publish: string;
}

export type Weapon =
  | "small_arms"
  | "aaa_light"
  | "aaa"
  | "manpads"
  | "manpads_advanced"
  | "rpg"
  | "atgm"
  | "sam"
  | "sam_mobile"
  | "sam_advanced";

export interface FIR {
  id: string;
  name: string;
  country: string;
  polygon: [number, number][];
  weaponry_range_floor: number;
  flight_level_floor: number;
  flight_level_ceiling: number;
  weapons: Weapon[];
  hostile_intercepts: boolean;
  cz_warnings: string[];
  issued_by: string;
}

export type AlertSeverity = "advisory" | "notice" | "warning" | "critical";
export type AlertCategory =
  | "terrorism"
  | "security"
  | "medical"
  | "weather"
  | "police_operation";

export interface Alert {
  id: string;
  severity: AlertSeverity;
  category: AlertCategory;
  region: string;
  lat: number;
  lng: number;
  headline: string;
  body: string;
  active_from: string;
  active_to: string;
  linked_airport_icao?: string;
  linked_fir_id?: string;
  tags: string[];
}

export type AircraftStatus = "in_flight" | "on_ground" | "maintenance" | "issue";

export interface Aircraft {
  tail_number: string;
  type: string;
  callsign: string;
  status: AircraftStatus;
  location: { icao?: string; lat?: number; lng?: number };
  origin?: string;
  dest?: string;
  issue?: { severity: "advisory" | "warning" | "critical"; headline: string; detail: string };
}

export interface Route {
  origin: string;
  dest: string;
  waypoints: [number, number][]; // [lng, lat][]
}
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/src/types.ts
git commit -m "add core type definitions"
```

---

## Task 3: Geo — haversine

**Files:**
- Create: `aviation-mcp-ui/src/geo.ts`
- Create: `aviation-mcp-ui/tests/geo.test.ts`

- [ ] **Step 1: Write failing test**

`tests/geo.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { haversineKm } from "../src/geo.js";

describe("haversineKm", () => {
  it("returns 0 for identical points", () => {
    expect(haversineKm([0, 0], [0, 0])).toBeCloseTo(0, 5);
  });

  it("computes PHX -> JFK ~3447 km within 1%", () => {
    const phx: [number, number] = [-112.0078, 33.4343];
    const jfk: [number, number] = [-73.7781, 40.6413];
    const d = haversineKm(phx, jfk);
    expect(d).toBeGreaterThan(3413);
    expect(d).toBeLessThan(3481);
  });

  it("computes LHR -> JFK ~5550 km within 1%", () => {
    const lhr: [number, number] = [-0.4543, 51.4700];
    const jfk: [number, number] = [-73.7781, 40.6413];
    const d = haversineKm(lhr, jfk);
    expect(d).toBeGreaterThan(5494);
    expect(d).toBeLessThan(5605);
  });
});
```

- [ ] **Step 2: Run test, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: FAIL — "Cannot find module '../src/geo.js'"

- [ ] **Step 3: Implement haversineKm**

`src/geo.ts`:

```ts
const R_KM = 6371.0088;
const toRad = (deg: number): number => (deg * Math.PI) / 180;

export function haversineKm(a: [number, number], b: [number, number]): number {
  const [lng1, lat1] = a;
  const [lng2, lat2] = b;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * R_KM * Math.asin(Math.min(1, Math.sqrt(s)));
}
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/geo.ts aviation-mcp-ui/tests/geo.test.ts
git commit -m "add haversineKm with tests"
```

---

## Task 4: Geo — great-circle waypoints

**Files:**
- Modify: `aviation-mcp-ui/src/geo.ts`
- Modify: `aviation-mcp-ui/tests/geo.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/geo.test.ts`:

```ts
import { greatCircleWaypoints } from "../src/geo.js";

describe("greatCircleWaypoints", () => {
  it("returns n points", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 12);
    expect(pts).toHaveLength(12);
  });

  it("first and last points match endpoints (within 0.01 deg)", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 8);
    expect(pts[0][0]).toBeCloseTo(-112, 1);
    expect(pts[0][1]).toBeCloseTo(33, 1);
    expect(pts[pts.length - 1][0]).toBeCloseTo(-74, 1);
    expect(pts[pts.length - 1][1]).toBeCloseTo(41, 1);
  });

  it("interior points lie roughly on the arc (latitude bows slightly north for east-west)", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 11);
    const mid = pts[5];
    const directMidLat = (33 + 41) / 2;
    expect(mid[1]).toBeGreaterThan(directMidLat - 1);
    expect(mid[1]).toBeLessThan(directMidLat + 5);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: FAIL — "greatCircleWaypoints is not a function"

- [ ] **Step 3: Implement greatCircleWaypoints**

Append to `src/geo.ts`:

```ts
function toCartesian(lng: number, lat: number): [number, number, number] {
  const phi = toRad(lat);
  const lam = toRad(lng);
  return [Math.cos(phi) * Math.cos(lam), Math.cos(phi) * Math.sin(lam), Math.sin(phi)];
}

function fromCartesian(v: [number, number, number]): [number, number] {
  const [x, y, z] = v;
  const lat = Math.atan2(z, Math.sqrt(x * x + y * y));
  const lng = Math.atan2(y, x);
  return [(lng * 180) / Math.PI, (lat * 180) / Math.PI];
}

export function greatCircleWaypoints(
  a: [number, number],
  b: [number, number],
  n = 12
): [number, number][] {
  if (n < 2) throw new Error("n must be >= 2");
  const pa = toCartesian(a[0], a[1]);
  const pb = toCartesian(b[0], b[1]);
  const dot = Math.max(-1, Math.min(1, pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]));
  const omega = Math.acos(dot);
  if (omega === 0) return Array.from({ length: n }, () => [...a] as [number, number]);
  const sinO = Math.sin(omega);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c1 = Math.sin((1 - t) * omega) / sinO;
    const c2 = Math.sin(t * omega) / sinO;
    const v: [number, number, number] = [
      c1 * pa[0] + c2 * pb[0],
      c1 * pa[1] + c2 * pb[1],
      c1 * pa[2] + c2 * pb[2],
    ];
    out.push(fromCartesian(v));
  }
  return out;
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/geo.ts aviation-mcp-ui/tests/geo.test.ts
git commit -m "add greatCircleWaypoints with tests"
```

---

## Task 5: Geo — point near polyline

**Files:**
- Modify: `aviation-mcp-ui/src/geo.ts`
- Modify: `aviation-mcp-ui/tests/geo.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/geo.test.ts`:

```ts
import { pointNearPolyline } from "../src/geo.js";

describe("pointNearPolyline", () => {
  const phx: [number, number] = [-112.0078, 33.4343];
  const jfk: [number, number] = [-73.7781, 40.6413];
  const route: [number, number][] = [phx, [-90, 38], jfk];

  it("returns true for a point exactly on a vertex", () => {
    expect(pointNearPolyline([-90, 38], route, 50)).toBe(true);
  });

  it("returns true for a point within threshold", () => {
    // ~80km north of midpoint
    expect(pointNearPolyline([-90, 38.7], route, 200)).toBe(true);
  });

  it("returns false for a far-away point", () => {
    // somewhere over Mexico
    expect(pointNearPolyline([-100, 20], route, 200)).toBe(false);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: FAIL — "pointNearPolyline is not a function"

- [ ] **Step 3: Implement pointNearPolyline**

Append to `src/geo.ts`:

```ts
function distancePointToSegmentKm(
  p: [number, number],
  a: [number, number],
  b: [number, number]
): number {
  // Approximate: project to local equirectangular plane near `a`, then point-to-segment distance.
  const latRef = toRad((a[1] + b[1]) / 2);
  const kx = Math.cos(latRef) * 111.32; // km per deg lng
  const ky = 110.574; // km per deg lat
  const ax = a[0] * kx;
  const ay = a[1] * ky;
  const bx = b[0] * kx;
  const by = b[1] * ky;
  const px = p[0] * kx;
  const py = p[1] * ky;
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

export function pointNearPolyline(
  point: [number, number],
  polyline: [number, number][],
  thresholdKm: number
): boolean {
  for (let i = 0; i < polyline.length - 1; i++) {
    const d = distancePointToSegmentKm(point, polyline[i], polyline[i + 1]);
    if (d <= thresholdKm) return true;
  }
  return false;
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/geo.test.ts`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/geo.ts aviation-mcp-ui/tests/geo.test.ts
git commit -m "add pointNearPolyline with tests"
```

---

## Task 6: Render — escapeHtml

**Files:**
- Create: `aviation-mcp-ui/src/render.ts`
- Create: `aviation-mcp-ui/tests/render.test.ts`

- [ ] **Step 1: Write failing test**

`tests/render.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { escapeHtml } from "../src/render.js";

describe("escapeHtml", () => {
  it("escapes ampersand, lt, gt, quotes", () => {
    expect(escapeHtml(`<script>alert("xss")</script>&'`)).toBe(
      "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;&amp;&#39;"
    );
  });

  it("returns empty string for null/undefined", () => {
    expect(escapeHtml(null as unknown as string)).toBe("");
    expect(escapeHtml(undefined as unknown as string)).toBe("");
  });

  it("stringifies non-strings", () => {
    expect(escapeHtml(42 as unknown as string)).toBe("42");
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement escapeHtml**

`src/render.ts`:

```ts
export function escapeHtml(value: unknown): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/render.ts aviation-mcp-ui/tests/render.test.ts
git commit -m "add escapeHtml with tests"
```

---

## Task 7: Render — template substitution

**Files:**
- Modify: `aviation-mcp-ui/src/render.ts`
- Modify: `aviation-mcp-ui/tests/render.test.ts`

Substitution syntax used in templates:
- `{{field}}` — escaped value
- `{{{field}}}` — raw (unescaped) — use sparingly, only for trusted HTML/CSS blocks
- `{{#array}}...{{/array}}` — repeat block per array item; inside the block, `{{.}}` is the item (string), or `{{prop}}` is a property of the item (object)
- `{{?bool}}...{{/bool}}` — render block iff truthy

- [ ] **Step 1: Add failing test**

Append to `tests/render.test.ts`:

```ts
import { renderTemplate } from "../src/render.js";

describe("renderTemplate", () => {
  it("substitutes a single field with escaping", () => {
    const out = renderTemplate("hello {{name}}", { name: "<b>" });
    expect(out).toBe("hello &lt;b&gt;");
  });

  it("renders raw with triple braces", () => {
    const out = renderTemplate("style: {{{css}}}", { css: ".x{color:red}" });
    expect(out).toBe("style: .x{color:red}");
  });

  it("repeats a section with item objects", () => {
    const out = renderTemplate(
      "<ul>{{#items}}<li>{{label}}</li>{{/items}}</ul>",
      { items: [{ label: "A" }, { label: "B" }] }
    );
    expect(out).toBe("<ul><li>A</li><li>B</li></ul>");
  });

  it("repeats a section with primitive items via {{.}}", () => {
    const out = renderTemplate(
      "{{#tags}}[{{.}}]{{/tags}}",
      { tags: ["one", "two"] }
    );
    expect(out).toBe("[one][two]");
  });

  it("renders conditional iff truthy", () => {
    expect(renderTemplate("{{?on}}YES{{/on}}", { on: true })).toBe("YES");
    expect(renderTemplate("{{?on}}YES{{/on}}", { on: false })).toBe("");
    expect(renderTemplate("{{?on}}YES{{/on}}", {})).toBe("");
  });

  it("treats missing fields as empty", () => {
    expect(renderTemplate("a={{a}} b={{b}}", { a: "x" })).toBe("a=x b=");
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: FAIL — "renderTemplate is not a function"

- [ ] **Step 3: Implement renderTemplate**

Append to `src/render.ts`:

```ts
type Ctx = Record<string, unknown>;

function lookup(ctx: Ctx, key: string): unknown {
  if (key === ".") return (ctx as { _self?: unknown })._self;
  return ctx[key];
}

export function renderTemplate(template: string, data: Ctx): string {
  // Sections (#) and conditionals (?) first.
  let out = template;
  const sectionRe = /\{\{([#?])(\w+)\}\}([\s\S]*?)\{\{\/\2\}\}/;
  while (true) {
    const m = out.match(sectionRe);
    if (!m) break;
    const [full, kind, name, body] = m;
    const value = data[name];
    let replacement = "";
    if (kind === "?") {
      if (value) replacement = renderTemplate(body, data);
    } else {
      if (Array.isArray(value)) {
        replacement = value
          .map((item) => {
            if (item && typeof item === "object") {
              return renderTemplate(body, item as Ctx);
            }
            return renderTemplate(body, { _self: item } as Ctx);
          })
          .join("");
      }
    }
    out = out.slice(0, m.index!) + replacement + out.slice(m.index! + full.length);
  }
  // Triple-brace raw.
  out = out.replace(/\{\{\{(\.|\w+)\}\}\}/g, (_, k) => {
    const v = lookup(data, k);
    return v === undefined || v === null ? "" : String(v);
  });
  // Double-brace escaped.
  out = out.replace(/\{\{(\.|\w+)\}\}/g, (_, k) => escapeHtml(lookup(data, k)));
  return out;
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/render.ts aviation-mcp-ui/tests/render.test.ts
git commit -m "add renderTemplate with tests"
```

---

## Task 8: Render — buildResource + loadTemplate

**Files:**
- Modify: `aviation-mcp-ui/src/render.ts`
- Modify: `aviation-mcp-ui/tests/render.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/render.test.ts`:

```ts
import { buildResource } from "../src/render.js";

describe("buildResource", () => {
  it("returns an MCP-UI EmbeddedResource shape", () => {
    const r = buildResource("airport", "LSZH", "<!DOCTYPE html><html></html>");
    expect(r.type).toBe("resource");
    expect(r.resource.uri).toBe("ui://aviation/airport/LSZH");
    expect(r.resource.mimeType).toBe("text/html");
    expect(r.resource.text).toContain("<!DOCTYPE html>");
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement buildResource and loadTemplate**

Append to `src/render.ts`:

```ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

export interface UIResource {
  type: "resource";
  resource: {
    uri: string;
    mimeType: "text/html";
    text: string;
  };
}

export function buildResource(view: string, id: string, html: string): UIResource {
  return {
    type: "resource",
    resource: {
      uri: `ui://aviation/${view}/${id}`,
      mimeType: "text/html",
      text: html,
    },
  };
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMPLATE_DIR = join(__dirname, "..", "templates");

const cache = new Map<string, string>();

export function loadTemplate(name: string): string {
  if (!cache.has(name)) {
    cache.set(name, readFileSync(join(TEMPLATE_DIR, name), "utf8"));
  }
  return cache.get(name)!;
}
```

Note: `templates/` lives at the package root (alongside `dist/`), and `package.json` `files` ships both. From compiled `dist/render.js`, `__dirname` is `<root>/dist`, so `join(__dirname, "..", "templates")` resolves to `<root>/templates` — correct. From `src/render.ts` under vitest, `__dirname` is `<root>/src`, and the same `..` resolves to `<root>/templates` — also correct.

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/render.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/render.ts aviation-mcp-ui/tests/render.test.ts
git commit -m "add buildResource + loadTemplate"
```

---

## Task 9: Theme CSS

**Files:**
- Create: `aviation-mcp-ui/templates/theme.css`

- [ ] **Step 1: Write theme.css**

`templates/theme.css`:

```css
:root {
  --bg: #0b1419;
  --panel: #11202a;
  --panel-2: #0e1a22;
  --border: #1a2a32;
  --accent: #3ddbd9;
  --accent-dim: #1a3a3a;
  --success: #3dd97a;
  --warning: #d9a83d;
  --danger: #d94d3d;
  --text: #d8e6ec;
  --text-muted: #7a99a8;
  --text-dim: #5b7a87;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
}
.frame {
  padding: 24px;
  max-width: 760px;
  margin: 0 auto;
}
.label {
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--text-muted);
  text-transform: uppercase;
}
.muted { color: var(--text-muted); }
.dim { color: var(--text-dim); }
.divider { height: 1px; background: var(--border); margin: 16px 0; }
.tag {
  display: inline-block;
  font-size: 10px;
  letter-spacing: 1px;
  padding: 2px 6px;
  border-radius: 3px;
  background: var(--accent-dim);
  color: var(--accent);
  text-transform: uppercase;
  margin-right: 4px;
}
.sev-critical { color: var(--danger); }
.sev-warning  { color: var(--warning); }
.sev-notice   { color: var(--accent); }
.sev-advisory { color: var(--text-muted); }
.dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}
.dot-green   { background: var(--success); box-shadow: 0 0 6px var(--success); }
.dot-amber   { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
.dot-red     { background: var(--danger);  box-shadow: 0 0 8px var(--danger);  }
.dot-grey    { background: var(--text-dim); }
.alert-strip {
  background: var(--panel);
  border-left: 3px solid var(--accent);
  padding: 10px 14px;
  margin-bottom: 18px;
  display: flex;
  gap: 10px;
  align-items: center;
  font-size: 13px;
}
.alert-strip .alert-time {
  color: var(--text-muted);
  font-size: 11px;
  letter-spacing: 1px;
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 rgba(217,77,61,0.6); }
  50%      { box-shadow: 0 0 12px rgba(217,77,61,0.9); }
}
.pulse-red { animation: pulse-red 1.6s ease-in-out infinite; }
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/templates/theme.css
git commit -m "add theme.css design tokens"
```

---

## Task 10: Fixtures — airports.json

**Files:**
- Create: `aviation-mcp-ui/fixtures/airports.json`

- [ ] **Step 1: Write fixture**

`fixtures/airports.json`:

```json
[
  {"icao":"KPHX","iata":"PHX","name":"Phoenix Sky Harbor","country":"United States","lat":33.4343,"lng":-112.0078,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"08/26","length_ft":11489,"width_ft":150,"elevation_ft":1135}],"ground_handling":{"slots_required":false,"handling_required":false},"date_of_publish":"2026-04-12"},
  {"icao":"KJFK","iata":"JFK","name":"John F. Kennedy Intl","country":"United States","lat":40.6413,"lng":-73.7781,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"04L/22R","length_ft":12079,"width_ft":200,"elevation_ft":13}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-09"},
  {"icao":"LSZH","iata":"ZRH","name":"Zurich","country":"Switzerland","lat":47.4647,"lng":8.5492,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"16/34","length_ft":12139,"width_ft":197,"elevation_ft":1417}],"ground_handling":{"slots_required":true,"handling_required":false},"date_of_publish":"2026-04-10"},
  {"icao":"OEJN","iata":"JED","name":"Jeddah King Abdulaziz","country":"Saudi Arabia","lat":21.6796,"lng":39.1565,"rating":3,"rating_label":"Caution Advised","covid_inbound":2,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":2,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"16C/34C","length_ft":13123,"width_ft":197,"elevation_ft":48}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-01"},
  {"icao":"EGLL","iata":"LHR","name":"London Heathrow","country":"United Kingdom","lat":51.4700,"lng":-0.4543,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"09L/27R","length_ft":12799,"width_ft":164,"elevation_ft":83}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-04"},
  {"icao":"EDDF","iata":"FRA","name":"Frankfurt","country":"Germany","lat":50.0379,"lng":8.5622,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"07R/25L","length_ft":13123,"width_ft":197,"elevation_ft":364}],"ground_handling":{"slots_required":true,"handling_required":false},"date_of_publish":"2026-04-25"},
  {"icao":"LFPG","iata":"CDG","name":"Paris Charles de Gaulle","country":"France","lat":49.0097,"lng":2.5479,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"08L/26R","length_ft":13829,"width_ft":197,"elevation_ft":392}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-30"},
  {"icao":"LIRF","iata":"FCO","name":"Rome Fiumicino","country":"Italy","lat":41.8003,"lng":12.2389,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":2,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"16R/34L","length_ft":12795,"width_ft":197,"elevation_ft":15}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-28"},
  {"icao":"LTBA","iata":"ISL","name":"Istanbul Atatürk","country":"Turkey","lat":40.9769,"lng":28.8146,"rating":3,"rating_label":"Caution Advised","covid_inbound":2,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":2,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"05/23","length_ft":9842,"width_ft":197,"elevation_ft":160}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-22"},
  {"icao":"OIIE","iata":"IKA","name":"Tehran Imam Khomeini","country":"Iran","lat":35.4161,"lng":51.1522,"rating":4,"rating_label":"Restricted Environment","covid_inbound":3,"covid_domestic":2,"city_risk":4,"city_risk_label":"High Security Risk","medical_risk":3,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"11R/29L","length_ft":13780,"width_ft":197,"elevation_ft":3305}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-03"},
  {"icao":"ORBI","iata":"BGW","name":"Baghdad Intl","country":"Iraq","lat":33.2625,"lng":44.2346,"rating":5,"rating_label":"Severe Environment","covid_inbound":2,"covid_domestic":2,"city_risk":5,"city_risk_label":"Severe Security Risk","medical_risk":4,"medical_risk_label":"High Medical Risk","runways":[{"id":"15L/33R","length_ft":13123,"width_ft":197,"elevation_ft":114}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-05"},
  {"icao":"HECA","iata":"CAI","name":"Cairo Intl","country":"Egypt","lat":30.1219,"lng":31.4056,"rating":3,"rating_label":"Caution Advised","covid_inbound":2,"covid_domestic":2,"city_risk":4,"city_risk_label":"High Security Risk","medical_risk":3,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"05L/23R","length_ft":13124,"width_ft":197,"elevation_ft":382}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-29"},
  {"icao":"OEDF","iata":"DMM","name":"Dammam King Fahd","country":"Saudi Arabia","lat":26.4712,"lng":49.7979,"rating":3,"rating_label":"Caution Advised","covid_inbound":2,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":2,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"16L/34R","length_ft":13123,"width_ft":197,"elevation_ft":72}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-02"},
  {"icao":"KORD","iata":"ORD","name":"Chicago O'Hare","country":"United States","lat":41.9742,"lng":-87.9073,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"10L/28R","length_ft":7500,"width_ft":150,"elevation_ft":672}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-15"},
  {"icao":"KLAX","iata":"LAX","name":"Los Angeles Intl","country":"United States","lat":33.9425,"lng":-118.4081,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"06R/24L","length_ft":10285,"width_ft":150,"elevation_ft":126}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-19"},
  {"icao":"KDEN","iata":"DEN","name":"Denver Intl","country":"United States","lat":39.8617,"lng":-104.6731,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":1,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"16R/34L","length_ft":16000,"width_ft":200,"elevation_ft":5431}],"ground_handling":{"slots_required":false,"handling_required":false},"date_of_publish":"2026-04-18"},
  {"icao":"KATL","iata":"ATL","name":"Atlanta Hartsfield-Jackson","country":"United States","lat":33.6407,"lng":-84.4277,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"08L/26R","length_ft":9000,"width_ft":150,"elevation_ft":1026}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-20"},
  {"icao":"KSFO","iata":"SFO","name":"San Francisco Intl","country":"United States","lat":37.6213,"lng":-122.3790,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"01L/19R","length_ft":7650,"width_ft":200,"elevation_ft":13}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-21"},
  {"icao":"KMIA","iata":"MIA","name":"Miami Intl","country":"United States","lat":25.7959,"lng":-80.2870,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"08L/26R","length_ft":8600,"width_ft":150,"elevation_ft":8}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-23"},
  {"icao":"KSEA","iata":"SEA","name":"Seattle-Tacoma","country":"United States","lat":47.4502,"lng":-122.3088,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":1,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"16L/34R","length_ft":11901,"width_ft":150,"elevation_ft":433}],"ground_handling":{"slots_required":false,"handling_required":false},"date_of_publish":"2026-04-26"},
  {"icao":"RJTT","iata":"HND","name":"Tokyo Haneda","country":"Japan","lat":35.5494,"lng":139.7798,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":1,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"16L/34R","length_ft":9843,"width_ft":197,"elevation_ft":21}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-02"},
  {"icao":"VHHH","iata":"HKG","name":"Hong Kong Intl","country":"Hong Kong SAR","lat":22.3080,"lng":113.9185,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"07L/25R","length_ft":12467,"width_ft":197,"elevation_ft":28}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-01"},
  {"icao":"WSSS","iata":"SIN","name":"Singapore Changi","country":"Singapore","lat":1.3644,"lng":103.9915,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":1,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"02L/20R","length_ft":13123,"width_ft":197,"elevation_ft":22}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-30"},
  {"icao":"OMDB","iata":"DXB","name":"Dubai Intl","country":"United Arab Emirates","lat":25.2532,"lng":55.3657,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"12L/30R","length_ft":13124,"width_ft":197,"elevation_ft":62}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-05-04"},
  {"icao":"ZBAA","iata":"PEK","name":"Beijing Capital","country":"China","lat":40.0801,"lng":116.5846,"rating":2,"rating_label":"Standard Environment","covid_inbound":2,"covid_domestic":2,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":2,"medical_risk_label":"Moderate Medical Risk","runways":[{"id":"01/19","length_ft":12468,"width_ft":197,"elevation_ft":116}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-27"},
  {"icao":"EHAM","iata":"AMS","name":"Amsterdam Schiphol","country":"Netherlands","lat":52.3105,"lng":4.7683,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"06/24","length_ft":11329,"width_ft":148,"elevation_ft":-11}],"ground_handling":{"slots_required":true,"handling_required":false},"date_of_publish":"2026-04-24"},
  {"icao":"LEMD","iata":"MAD","name":"Madrid Barajas","country":"Spain","lat":40.4983,"lng":-3.5676,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"14L/32R","length_ft":11483,"width_ft":197,"elevation_ft":2000}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-26"},
  {"icao":"LSGG","iata":"GVA","name":"Geneva","country":"Switzerland","lat":46.2381,"lng":6.1090,"rating":1,"rating_label":"Unrestricted Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"04/22","length_ft":12795,"width_ft":164,"elevation_ft":1411}],"ground_handling":{"slots_required":true,"handling_required":false},"date_of_publish":"2026-04-21"},
  {"icao":"EBBR","iata":"BRU","name":"Brussels","country":"Belgium","lat":50.9014,"lng":4.4844,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":3,"city_risk_label":"Moderate Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"07L/25R","length_ft":11936,"width_ft":164,"elevation_ft":184}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-22"},
  {"icao":"EPWA","iata":"WAW","name":"Warsaw Chopin","country":"Poland","lat":52.1657,"lng":20.9671,"rating":2,"rating_label":"Standard Environment","covid_inbound":1,"covid_domestic":1,"city_risk":2,"city_risk_label":"Low Security Risk","medical_risk":1,"medical_risk_label":"Low Medical Risk","runways":[{"id":"11/29","length_ft":9213,"width_ft":197,"elevation_ft":362}],"ground_handling":{"slots_required":true,"handling_required":true},"date_of_publish":"2026-04-23"}
]
```

- [ ] **Step 2: Validate JSON parses**

Run: `cd aviation-mcp-ui && node -e "JSON.parse(require('fs').readFileSync('fixtures/airports.json','utf8'))"`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add aviation-mcp-ui/fixtures/airports.json
git commit -m "add airports fixture (30 ICAOs)"
```

---

## Task 11: Fixtures — firs.json

**Files:**
- Create: `aviation-mcp-ui/fixtures/firs.json`

- [ ] **Step 1: Write fixture**

`fixtures/firs.json`:

```json
[
  {"id":"ORBB","name":"Baghdad FIR","country":"Iraq","polygon":[[39,29],[48,29],[48,38],[39,38],[39,29]],"weaponry_range_floor":260,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa_light","aaa","manpads","manpads_advanced","rpg","atgm"],"hostile_intercepts":true,"cz_warnings":["CZ Warning - Conflict zone","Risk of misidentification"],"issued_by":"Canada"},
  {"id":"OIIX","name":"Tehran FIR","country":"Iran","polygon":[[44,25],[63,25],[63,40],[44,40],[44,25]],"weaponry_range_floor":300,"flight_level_floor":200,"flight_level_ceiling":600,"weapons":["small_arms","aaa","manpads_advanced","sam","sam_mobile","sam_advanced"],"hostile_intercepts":true,"cz_warnings":["CZ Warning - Active SAM threat"],"issued_by":"Canada"},
  {"id":"LSAS","name":"Switzerland FIR","country":"Switzerland","polygon":[[5.9,45.8],[10.5,45.8],[10.5,47.8],[5.9,47.8],[5.9,45.8]],"weaponry_range_floor":600,"flight_level_floor":0,"flight_level_ceiling":660,"weapons":[],"hostile_intercepts":false,"cz_warnings":[],"issued_by":"EUROCONTROL"},
  {"id":"HECC","name":"Cairo FIR","country":"Egypt","polygon":[[24,21],[37,21],[37,32],[24,32],[24,21]],"weaponry_range_floor":260,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa_light","manpads"],"hostile_intercepts":false,"cz_warnings":["CZ Warning - Sinai region"],"issued_by":"Canada"},
  {"id":"EGTT","name":"London FIR","country":"United Kingdom","polygon":[[-7,49],[2,49],[2,55],[-7,55],[-7,49]],"weaponry_range_floor":600,"flight_level_floor":0,"flight_level_ceiling":660,"weapons":[],"hostile_intercepts":false,"cz_warnings":[],"issued_by":"EUROCONTROL"},
  {"id":"URRV","name":"Rostov FIR","country":"Russia","polygon":[[36,44],[48,44],[48,52],[36,52],[36,44]],"weaponry_range_floor":260,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa","manpads_advanced","sam","sam_mobile"],"hostile_intercepts":true,"cz_warnings":["CZ Warning - Active conflict zone","Risk of misidentification"],"issued_by":"Canada"},
  {"id":"LTBB","name":"Ankara FIR","country":"Turkey","polygon":[[26,36],[44,36],[44,42],[26,42],[26,36]],"weaponry_range_floor":300,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa_light","manpads"],"hostile_intercepts":false,"cz_warnings":["CZ Warning - SE border region"],"issued_by":"Canada"},
  {"id":"OYSC","name":"Sanaa FIR","country":"Yemen","polygon":[[42,12],[55,12],[55,19],[42,19],[42,12]],"weaponry_range_floor":260,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa","manpads_advanced","sam","sam_advanced"],"hostile_intercepts":true,"cz_warnings":["CZ Warning - Active SAM threat","Houthi missile activity"],"issued_by":"Canada"},
  {"id":"HKNA","name":"Nairobi FIR","country":"Kenya","polygon":[[33,-5],[42,-5],[42,5],[33,5],[33,-5]],"weaponry_range_floor":300,"flight_level_floor":180,"flight_level_ceiling":600,"weapons":["small_arms","aaa_light"],"hostile_intercepts":false,"cz_warnings":[],"issued_by":"Canada"},
  {"id":"UTAK","name":"Almaty FIR","country":"Kazakhstan","polygon":[[60,40],[88,40],[88,55],[60,55],[60,40]],"weaponry_range_floor":600,"flight_level_floor":0,"flight_level_ceiling":660,"weapons":[],"hostile_intercepts":false,"cz_warnings":[],"issued_by":"Canada"}
]
```

- [ ] **Step 2: Validate**

Run: `cd aviation-mcp-ui && node -e "JSON.parse(require('fs').readFileSync('fixtures/firs.json','utf8'))"`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add aviation-mcp-ui/fixtures/firs.json
git commit -m "add FIRs fixture (10 regions)"
```

---

## Task 12: Fixtures — alerts.json

**Files:**
- Create: `aviation-mcp-ui/fixtures/alerts.json`

- [ ] **Step 1: Write fixture**

`fixtures/alerts.json`:

```json
[
  {"id":"AL-2026-0501","severity":"critical","category":"terrorism","region":"EUROPE & CIS","lat":48.8566,"lng":2.3522,"headline":"EXERCISE VIGILANCE DUE TO INCREASED RISK OF ATTACKS LINKED TO MIDDLE EAST CONFLICTS","body":"The risk of isolated and extremism-linked attacks in Europe increased amid the Israel-US military operation against Iran. Several incidents of targeted attacks have occurred across Europe in recent weeks, prompting some countries to enhance security measures.","active_from":"2026-05-04T17:00:00Z","active_to":"2026-06-04T17:00:00Z","tags":["TERRORISM","POLICE/SECURITY OPERATION"]},
  {"id":"AL-2026-0502","severity":"critical","category":"security","region":"GULF","lat":33.2625,"lng":44.2346,"headline":"AVIATION ALERT (GULF REGION) — ELEVATED THREAT TO COMMERCIAL AIRCRAFT","body":"Hostile intercepts and active SAM activity reported across Iraq and Iran airspace. Operators are advised to avoid overflight of ORBB and OIIX FIRs above FL180.","active_from":"2026-05-05T00:00:00Z","active_to":"2026-06-05T00:00:00Z","linked_fir_id":"ORBB","tags":["AVIATION","CONFLICT ZONE"]},
  {"id":"AL-2026-0503","severity":"warning","category":"security","region":"MIDDLE EAST","lat":35.4161,"lng":51.1522,"headline":"ELEVATED RISK OF MILITARY ACTION OVER IRANIAN AIRSPACE","body":"Continued risk of cross-border strikes. Operators should consider alternate routings via UTAK or HECC.","active_from":"2026-05-01T00:00:00Z","active_to":"2026-06-01T00:00:00Z","linked_fir_id":"OIIX","tags":["AVIATION","CONFLICT ZONE"]},
  {"id":"AL-2026-0504","severity":"warning","category":"security","region":"BLACK SEA","lat":47.2357,"lng":39.7015,"headline":"ROSTOV FIR — ACTIVE CONFLICT ZONE, AVOID OVERFLIGHT","body":"Active SAM activity and risk of misidentification. ICAO has advised carriers to avoid the Rostov FIR.","active_from":"2026-04-15T00:00:00Z","active_to":"2026-07-15T00:00:00Z","linked_fir_id":"URRV","tags":["AVIATION","CONFLICT ZONE"]},
  {"id":"AL-2026-0505","severity":"warning","category":"security","region":"ARABIAN PENINSULA","lat":15.3694,"lng":44.1910,"headline":"YEMEN — HOUTHI MISSILE ACTIVITY","body":"Increased risk of advanced SAM and missile threats. Operators should avoid OYSC FIR.","active_from":"2026-04-20T00:00:00Z","active_to":"2026-07-20T00:00:00Z","linked_fir_id":"OYSC","tags":["AVIATION","MISSILE"]},
  {"id":"AL-2026-0506","severity":"notice","category":"medical","region":"EUROPE","lat":47.3769,"lng":8.5417,"headline":"HANTAVIRUS CLUSTER LINKED TO CRUISE SHIP — MONITOR DEVELOPMENTS","body":"A small cluster of hantavirus cases has been linked to a cruise ship docking in Zurich's vicinity. Monitor for travel advisories.","active_from":"2026-05-03T00:00:00Z","active_to":"2026-06-03T00:00:00Z","linked_airport_icao":"LSZH","tags":["MEDICAL"]},
  {"id":"AL-2026-0507","severity":"advisory","category":"weather","region":"NORTH AMERICA","lat":35,"lng":-100,"headline":"SEVERE THUNDERSTORM ACTIVITY ALONG TRANSCONTINENTAL CORRIDOR","body":"Convective SIGMETs in effect across the central US. Expect rerouting and delays on PHX-JFK and similar routings.","active_from":"2026-05-06T00:00:00Z","active_to":"2026-05-08T00:00:00Z","tags":["WEATHER"]},
  {"id":"AL-2026-0508","severity":"advisory","category":"police_operation","region":"EUROPE","lat":51.5074,"lng":-0.1278,"headline":"INCREASED POLICE PRESENCE — LONDON","body":"UK terrorism threat level raised to 'severe' (second-highest). Additional plain-clothed officers near transport hubs.","active_from":"2026-04-30T00:00:00Z","active_to":"2026-05-30T00:00:00Z","linked_airport_icao":"EGLL","tags":["POLICE/SECURITY OPERATION"]},
  {"id":"AL-2026-0509","severity":"notice","category":"security","region":"NORTH AMERICA","lat":40.6413,"lng":-73.7781,"headline":"JFK — INCREASED TSA STAFFING DUE TO ELEVATED THREAT POSTURE","body":"Expect longer screening times. Allow extra time for departures.","active_from":"2026-05-02T00:00:00Z","active_to":"2026-05-31T00:00:00Z","linked_airport_icao":"KJFK","tags":["AVIATION","SECURITY"]},
  {"id":"AL-2026-0510","severity":"advisory","category":"weather","region":"NORTH AMERICA","lat":33.4343,"lng":-112.0078,"headline":"PHX — DUST STORM ADVISORY","body":"Periodic haboobs reducing visibility. Brief departure delays possible.","active_from":"2026-05-05T00:00:00Z","active_to":"2026-05-15T00:00:00Z","linked_airport_icao":"KPHX","tags":["WEATHER"]},
  {"id":"AL-2026-0511","severity":"warning","category":"medical","region":"AFRICA","lat":-1.3,"lng":36.8,"headline":"CHOLERA OUTBREAK — NAIROBI","body":"WHO confirms ongoing cholera outbreak. Vaccinate before travel.","active_from":"2026-04-15T00:00:00Z","active_to":"2026-07-15T00:00:00Z","tags":["MEDICAL"]},
  {"id":"AL-2026-0512","severity":"notice","category":"security","region":"EUROPE","lat":52.5200,"lng":13.4050,"headline":"BERLIN — DEMONSTRATIONS NEAR EMBASSY ROW","body":"Avoid Mitte district during scheduled demonstrations.","active_from":"2026-05-04T00:00:00Z","active_to":"2026-05-12T00:00:00Z","tags":["SECURITY","POLICE/SECURITY OPERATION"]},
  {"id":"AL-2026-0513","severity":"critical","category":"security","region":"MIDDLE EAST","lat":21.6796,"lng":39.1565,"headline":"OEJN — INCREASED MISSILE THREAT FROM YEMEN","body":"Operators should monitor advisories. Brief stand-downs possible.","active_from":"2026-05-04T00:00:00Z","active_to":"2026-06-04T00:00:00Z","linked_airport_icao":"OEJN","tags":["AVIATION","MISSILE"]},
  {"id":"AL-2026-0514","severity":"advisory","category":"weather","region":"EUROPE","lat":50.0379,"lng":8.5622,"headline":"FRA — LOW VISIBILITY OPS THIS MORNING","body":"Fog reducing arrivals to single-runway operations until 11:00 local.","active_from":"2026-05-07T05:00:00Z","active_to":"2026-05-07T11:00:00Z","linked_airport_icao":"EDDF","tags":["WEATHER"]},
  {"id":"AL-2026-0515","severity":"warning","category":"police_operation","region":"NORTH AMERICA","lat":34.0522,"lng":-118.2437,"headline":"LAX — TEMPORARY ROAD CLOSURES, PROTEST ACTIVITY","body":"Ground transport disruptions expected through the weekend.","active_from":"2026-05-06T00:00:00Z","active_to":"2026-05-10T00:00:00Z","linked_airport_icao":"KLAX","tags":["POLICE/SECURITY OPERATION"]},
  {"id":"AL-2026-0516","severity":"notice","category":"security","region":"NORTH ATLANTIC","lat":42,"lng":-50,"headline":"NAT TRACKS — TEMPORARY OCEANIC RESTRICTIONS","body":"Reduced lateral separation in effect on NAT-A and NAT-B.","active_from":"2026-05-06T00:00:00Z","active_to":"2026-05-09T00:00:00Z","tags":["AVIATION"]},
  {"id":"AL-2026-0517","severity":"advisory","category":"medical","region":"EUROPE","lat":41.9028,"lng":12.4964,"headline":"FCO — SEASONAL FLU UPTICK","body":"Standard precautions advised.","active_from":"2026-05-01T00:00:00Z","active_to":"2026-05-31T00:00:00Z","linked_airport_icao":"LIRF","tags":["MEDICAL"]},
  {"id":"AL-2026-0518","severity":"warning","category":"security","region":"AFRICA","lat":30.1219,"lng":31.4056,"headline":"CAIRO — SINAI REGION ADVISORY","body":"Operators should avoid low-altitude flight near Sinai.","active_from":"2026-04-25T00:00:00Z","active_to":"2026-07-25T00:00:00Z","linked_airport_icao":"HECA","linked_fir_id":"HECC","tags":["AVIATION","CONFLICT ZONE"]},
  {"id":"AL-2026-0519","severity":"notice","category":"weather","region":"NORTH AMERICA","lat":39.8617,"lng":-104.6731,"headline":"DEN — HIGH WIND ADVISORY","body":"Crosswind component nearing limits on RW34L. Expect runway changes.","active_from":"2026-05-07T00:00:00Z","active_to":"2026-05-08T00:00:00Z","linked_airport_icao":"KDEN","tags":["WEATHER"]},
  {"id":"AL-2026-0520","severity":"advisory","category":"security","region":"EUROPE","lat":52.3105,"lng":4.7683,"headline":"AMS — INCREASED PRESENCE OF MARECHAUSSEE","body":"Routine elevated screening for inbound flights from MENA region.","active_from":"2026-05-03T00:00:00Z","active_to":"2026-05-30T00:00:00Z","linked_airport_icao":"EHAM","tags":["SECURITY"]},
  {"id":"AL-2026-0521","severity":"critical","category":"security","region":"GULF","lat":26.4712,"lng":49.7979,"headline":"DMM — IMMEDIATE MISSILE THREAT, GROUND STOP IN EFFECT","body":"Active ground stop. Inbound traffic diverting to OERK and OEDR.","active_from":"2026-05-07T08:00:00Z","active_to":"2026-05-07T20:00:00Z","linked_airport_icao":"OEDF","tags":["AVIATION","MISSILE"]},
  {"id":"AL-2026-0522","severity":"warning","category":"weather","region":"ASIA","lat":35.5494,"lng":139.7798,"headline":"HND — TYPHOON SYSTEM APPROACHING","body":"Significant delays expected over next 48 hours.","active_from":"2026-05-07T00:00:00Z","active_to":"2026-05-09T00:00:00Z","linked_airport_icao":"RJTT","tags":["WEATHER"]},
  {"id":"AL-2026-0523","severity":"notice","category":"security","region":"ASIA","lat":22.3080,"lng":113.9185,"headline":"HKG — INCREASED IMMIGRATION SCRUTINY","body":"Expect longer arrival processing times.","active_from":"2026-05-04T00:00:00Z","active_to":"2026-05-30T00:00:00Z","linked_airport_icao":"VHHH","tags":["SECURITY"]},
  {"id":"AL-2026-0524","severity":"advisory","category":"security","region":"NORTH AMERICA","lat":25.7959,"lng":-80.2870,"headline":"MIA — INCREASED MARITIME PATROLS","body":"Routine. No operational impact expected.","active_from":"2026-05-01T00:00:00Z","active_to":"2026-05-30T00:00:00Z","linked_airport_icao":"KMIA","tags":["SECURITY"]},
  {"id":"AL-2026-0525","severity":"warning","category":"terrorism","region":"EUROPE","lat":50.9014,"lng":4.4844,"headline":"BRUSSELS — ELEVATED THREAT POSTURE NEAR EU INSTITUTIONS","body":"Avoid the European Quarter during scheduled events.","active_from":"2026-05-02T00:00:00Z","active_to":"2026-05-31T00:00:00Z","linked_airport_icao":"EBBR","tags":["TERRORISM","POLICE/SECURITY OPERATION"]}
]
```

- [ ] **Step 2: Validate**

Run: `cd aviation-mcp-ui && node -e "JSON.parse(require('fs').readFileSync('fixtures/alerts.json','utf8'))"`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add aviation-mcp-ui/fixtures/alerts.json
git commit -m "add alerts fixture (25 entries)"
```

---

## Task 13: Fixtures — fleet.json

**Files:**
- Create: `aviation-mcp-ui/fixtures/fleet.json`

- [ ] **Step 1: Write fixture**

`fixtures/fleet.json`:

```json
[
  {"tail_number":"N737AZ","type":"B737-800","callsign":"OSPREY101","status":"in_flight","location":{"lat":36.5,"lng":-95.0},"origin":"KPHX","dest":"KJFK"},
  {"tail_number":"N320EU","type":"A320neo","callsign":"OSPREY204","status":"in_flight","location":{"lat":48.0,"lng":15.0},"origin":"EDDF","dest":"LTBA"},
  {"tail_number":"N787LR","type":"B787-9","callsign":"OSPREY312","status":"in_flight","location":{"lat":22.0,"lng":42.0},"origin":"OMDB","dest":"OEJN","issue":{"severity":"warning","headline":"DIVERTING — MEDICAL EMERGENCY","detail":"Passenger medical event onboard. Currently 80nm SE of OEJN, requesting priority handling. Note proximity to active OYSC threat envelope."}},
  {"tail_number":"N350IF","type":"A350-900","callsign":"OSPREY418","status":"in_flight","location":{"lat":15,"lng":-30},"origin":"KJFK","dest":"EGLL"},
  {"tail_number":"N777BG","type":"B777-300ER","callsign":"OSPREY521","status":"on_ground","location":{"icao":"EHAM"}},
  {"tail_number":"N220RG","type":"CRJ-900","callsign":"OSPREY602","status":"on_ground","location":{"icao":"KSFO"}},
  {"tail_number":"N130MX","type":"A330-300","callsign":"OSPREY733","status":"maintenance","location":{"icao":"KATL"},"issue":{"severity":"advisory","headline":"Scheduled C-check","detail":"Out of service through 14 May."}},
  {"tail_number":"N175RJ","type":"E175","callsign":"OSPREY811","status":"in_flight","location":{"lat":42,"lng":-110},"origin":"KSEA","dest":"KDEN"}
]
```

- [ ] **Step 2: Validate**

Run: `cd aviation-mcp-ui && node -e "JSON.parse(require('fs').readFileSync('fixtures/fleet.json','utf8'))"`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add aviation-mcp-ui/fixtures/fleet.json
git commit -m "add fleet fixture (8 aircraft, 1 with active issue)"
```

---

## Task 14: Fixtures — routes.json

**Files:**
- Create: `aviation-mcp-ui/fixtures/routes.json`

- [ ] **Step 1: Write fixture**

`fixtures/routes.json`:

```json
[
  {"origin":"KPHX","dest":"KJFK","waypoints":[[-112.0078,33.4343],[-108.6,34.7],[-104.7,35.9],[-100.3,37.0],[-95.4,37.8],[-90.1,38.4],[-84.6,38.7],[-79.0,38.8],[-73.7781,40.6413]]},
  {"origin":"KJFK","dest":"EGLL","waypoints":[[-73.7781,40.6413],[-65,44.5],[-55,48.5],[-43,52.0],[-30,54.5],[-15,55.5],[-0.4543,51.4700]]},
  {"origin":"EGLL","dest":"OMDB","waypoints":[[-0.4543,51.4700],[8,48],[18,44],[28,40],[38,36],[48,30],[55.3657,25.2532]]},
  {"origin":"OMDB","dest":"WSSS","waypoints":[[55.3657,25.2532],[64,21],[73,17],[82,13],[91,9],[100,5],[103.9915,1.3644]]},
  {"origin":"EDDF","dest":"OIIE","waypoints":[[8.5622,50.0379],[18,46],[28,42],[38,38],[48,36],[51.1522,35.4161]]},
  {"origin":"KLAX","dest":"RJTT","waypoints":[[-118.4081,33.9425],[-135,40],[-152,45],[-170,47],[170,46],[155,42],[139.7798,35.5494]]}
]
```

- [ ] **Step 2: Validate**

Run: `cd aviation-mcp-ui && node -e "JSON.parse(require('fs').readFileSync('fixtures/routes.json','utf8'))"`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add aviation-mcp-ui/fixtures/routes.json
git commit -m "add precomputed demo routes"
```

---

## Task 15: Data module

**Files:**
- Create: `aviation-mcp-ui/src/data.ts`
- Create: `aviation-mcp-ui/tests/data.test.ts`

- [ ] **Step 1: Write failing tests**

`tests/data.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  findAirport,
  findFir,
  alertsNearAirport,
  alertsLinkedToFir,
  alertsAlongRoute,
  getRoute,
  getFleet,
} from "../src/data.js";

describe("data lookups", () => {
  it("finds an airport by ICAO", () => {
    const a = findAirport("LSZH");
    expect(a?.iata).toBe("ZRH");
  });

  it("returns undefined for unknown ICAO", () => {
    expect(findAirport("XXXX")).toBeUndefined();
  });

  it("finds a FIR by id", () => {
    expect(findFir("ORBB")?.name).toBe("Baghdad FIR");
  });

  it("returns alerts within 50km of an airport", () => {
    const lszh = findAirport("LSZH")!;
    const a = alertsNearAirport(lszh, 50);
    expect(a.some((x) => x.id === "AL-2026-0506")).toBe(true);
  });

  it("returns alerts linked to a FIR", () => {
    const a = alertsLinkedToFir("ORBB");
    expect(a.some((x) => x.id === "AL-2026-0502")).toBe(true);
  });

  it("uses precomputed route when present", () => {
    const r = getRoute("KPHX", "KJFK");
    expect(r.waypoints[0]).toEqual([-112.0078, 33.4343]);
    expect(r.waypoints.at(-1)).toEqual([-73.7781, 40.6413]);
  });

  it("falls back to great-circle for unknown pairs", () => {
    const r = getRoute("KSEA", "KMIA");
    expect(r.waypoints.length).toBeGreaterThan(2);
  });

  it("alertsAlongRoute returns alerts within corridor", () => {
    const r = getRoute("KPHX", "KJFK");
    const a = alertsAlongRoute(r.waypoints, 250);
    expect(a.length).toBeGreaterThan(0);
  });

  it("returns the fleet", () => {
    const f = getFleet();
    expect(f.length).toBeGreaterThanOrEqual(8);
    expect(f.some((a) => a.status === "issue")).toBe(true);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/data.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement data.ts**

`src/data.ts`:

```ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { Airport, FIR, Alert, Aircraft, Route } from "./types.js";
import { haversineKm, greatCircleWaypoints, pointNearPolyline } from "./geo.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FIXTURE_DIR = join(__dirname, "..", "fixtures");

function load<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURE_DIR, name), "utf8")) as T;
}

const airports: Airport[] = load("airports.json");
const firs: FIR[] = load("firs.json");
const alerts: Alert[] = load("alerts.json");
const fleet: Aircraft[] = load("fleet.json");
const routes: Route[] = load("routes.json");

export function findAirport(icao: string): Airport | undefined {
  const u = icao.toUpperCase();
  return airports.find((a) => a.icao === u);
}

export function findFir(id: string): FIR | undefined {
  const u = id.toUpperCase();
  return firs.find((f) => f.id === u);
}

export function getFleet(): Aircraft[] {
  return fleet;
}

export function alertsNearAirport(airport: Airport, radiusKm = 50): Alert[] {
  return alerts.filter((al) => {
    if (al.linked_airport_icao === airport.icao) return true;
    return haversineKm([airport.lng, airport.lat], [al.lng, al.lat]) <= radiusKm;
  });
}

export function alertsLinkedToFir(firId: string): Alert[] {
  const u = firId.toUpperCase();
  return alerts.filter((al) => al.linked_fir_id === u);
}

export function getRoute(origin: string, dest: string): Route {
  const o = origin.toUpperCase();
  const d = dest.toUpperCase();
  const cached = routes.find((r) => r.origin === o && r.dest === d);
  if (cached) return cached;
  const a = findAirport(o);
  const b = findAirport(d);
  if (!a || !b) {
    throw new Error(`Unknown airport: ${a ? d : o}`);
  }
  return {
    origin: o,
    dest: d,
    waypoints: greatCircleWaypoints([a.lng, a.lat], [b.lng, b.lat], 12),
  };
}

export function alertsAlongRoute(
  waypoints: [number, number][],
  corridorKm = 250
): Alert[] {
  return alerts.filter((al) =>
    pointNearPolyline([al.lng, al.lat], waypoints, corridorKm)
  );
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/data.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/data.ts aviation-mcp-ui/tests/data.test.ts
git commit -m "add data module with fixture lookups"
```

---

## Task 16: Airport-card template

**Files:**
- Create: `aviation-mcp-ui/templates/airport-card.html`

The template embeds `theme.css` inline. Substitution: `{{icao}}`, `{{iata}}`, `{{name}}` (used as city header), `{{country}}`, `{{rating}}`, `{{rating_label}}`, `{{rating_deg}}`, `{{covid_inbound}}`, `{{covid_domestic}}`, `{{city_risk_label}}`, `{{city_risk_deg}}`, `{{medical_risk_label}}`, `{{medical_risk_deg}}`, `{{date_of_publish}}`, `{{runway_summary}}`, `{{ground_handling_summary}}`, optional alert section `{{?alert}}...{{/alert}}` with `{{alert_time}}`, `{{alert_text}}`, `{{alert_distance_km}}`.

The `*_deg` fields are precomputed by the tool (degrees of arc on the conic-gradient ring): `rating_deg = rating * 72`, `risk_deg = risk * 72` for a 1-5 scale = 72-360 degrees.

- [ ] **Step 1: Write template**

`templates/airport-card.html`:

```html
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{icao}}</title>
<style>{{{theme_css}}}
.ac-frame { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; max-width: 720px; margin: 24px auto; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.ac-head { display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:20px; }
.ac-icao { font-size: 32px; font-weight: 700; letter-spacing: 1px; color:#fff; }
.ac-city { font-size: 13px; color: var(--text-muted); letter-spacing: 2px; margin-top:4px; text-transform: uppercase; }
.ac-iata { font-size: 11px; color: var(--text-dim); letter-spacing: 3px; margin-top:6px; }
.ac-tabs { display:flex; gap: 0; margin-bottom:20px; border-bottom:1px solid var(--border); }
.ac-tab { padding: 10px 20px; font-size: 11px; letter-spacing: 2px; color: var(--text-muted); border-bottom: 2px solid transparent; }
.ac-tab.active { color: #fff; border-bottom-color: var(--accent); }
.ac-grid { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:20px; }
.ac-stat { display:flex; gap:14px; align-items:center; }
.ac-dial { width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center; position:relative; }
.ac-dial::after { content:""; position:absolute; inset: 6px; border-radius:50%; background: var(--bg); }
.ac-stat-label { font-size: 10px; letter-spacing:2px; color: var(--text-muted); margin-bottom:3px; text-transform: uppercase; }
.ac-stat-value { font-size: 13px; color: var(--text); }
.ac-row { display:flex; gap:14px; padding:12px 0; border-top:1px solid var(--border); align-items:center; }
.ac-row-icon { width:30px; color: var(--accent); font-size:18px; text-align:center; }
.ac-row-label { font-size:10px; letter-spacing:2px; color: var(--text-muted); text-transform: uppercase; }
.ac-row-value { font-size:12px; color: var(--text); }
.ac-side { display:flex; flex-direction:column; gap:4px; font-size:10px; letter-spacing:2px; color: var(--text-dim); padding-right:14px; padding-top: 14px; text-transform: uppercase; }
.ac-side-item.active { color: var(--accent); border-left:2px solid var(--accent); padding-left:8px; margin-left:-10px; }
.ac-rows-wrap { display:flex; }
.ac-rows { flex:1; }
.ac-publish { font-size:10px; color: var(--text-dim); letter-spacing:2px; margin-bottom:18px; }
</style>
</head>
<body>
<div class="ac-frame">
  <div class="ac-head">
    <div>
      <div class="ac-icao">{{icao}}</div>
      <div class="ac-city">{{name}}</div>
      <div class="ac-iata">{{iata}} · {{country}}</div>
    </div>
  </div>
  <div class="ac-tabs">
    <div class="ac-tab active">AIRPORT</div>
    <div class="ac-tab">CITY</div>
    <div class="ac-tab">COUNTRY</div>
    <div class="ac-tab">AIRSPACE</div>
  </div>
  {{?alert}}
  <div class="alert-strip">
    <span style="color: var(--accent)">▲</span>
    <span class="alert-time">{{alert_time}}</span>
    <span>{{alert_text}}</span>
    <span class="tag" style="margin-left:auto">{{alert_distance_km}}KM</span>
  </div>
  {{/alert}}
  <div class="ac-publish">DATE OF PUBLISH {{date_of_publish}}</div>
  <div class="ac-grid">
    <div class="ac-stat">
      <div class="ac-dial" style="background: conic-gradient(var(--accent) 0deg {{rating_deg}}deg, var(--border) {{rating_deg}}deg 360deg);"></div>
      <div><div class="ac-stat-label">AIRPORT RATING</div><div class="ac-stat-value">{{rating}} — {{rating_label}}</div></div>
    </div>
    <div class="ac-stat">
      <div class="ac-dial" style="background: conic-gradient(var(--accent) 0deg {{covid_deg}}deg, var(--border) {{covid_deg}}deg 360deg);"></div>
      <div><div class="ac-stat-label">COVID IMPACT</div><div class="ac-stat-value">Inbound {{covid_inbound}} · Domestic {{covid_domestic}}</div></div>
    </div>
    <div class="ac-stat">
      <div class="ac-dial" style="background: conic-gradient(var(--success) 0deg {{city_risk_deg}}deg, var(--border) {{city_risk_deg}}deg 360deg);"></div>
      <div><div class="ac-stat-label">CITY RISK</div><div class="ac-stat-value">{{city_risk_label}}</div></div>
    </div>
    <div class="ac-stat">
      <div class="ac-dial" style="background: conic-gradient(var(--success) 0deg {{medical_risk_deg}}deg, var(--border) {{medical_risk_deg}}deg 360deg);"></div>
      <div><div class="ac-stat-label">MEDICAL RISK</div><div class="ac-stat-value">{{medical_risk_label}}</div></div>
    </div>
  </div>
  <div class="ac-rows-wrap">
    <div class="ac-side">
      <div class="ac-side-item active">OVERVIEW</div>
      <div class="ac-side-item">OPERATIONS</div>
      <div class="ac-side-item">SECURITY</div>
    </div>
    <div class="ac-rows">
      <div class="ac-row"><div class="ac-row-icon">✈</div><div><div class="ac-row-label">RUNWAY SPECS</div><div class="ac-row-value">{{runway_summary}}</div></div></div>
      <div class="ac-row"><div class="ac-row-icon">⛟</div><div><div class="ac-row-label">GROUND HANDLING</div><div class="ac-row-value">{{ground_handling_summary}}</div></div></div>
    </div>
  </div>
</div>
</body></html>
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/templates/airport-card.html
git commit -m "add airport-card template"
```

---

## Task 17: getAirport tool

**Files:**
- Create: `aviation-mcp-ui/src/tools/getAirport.ts`
- Modify: `aviation-mcp-ui/tests/tools.test.ts` (create if missing)

- [ ] **Step 1: Write failing test**

`tests/tools.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { getAirport } from "../src/tools/getAirport.js";

describe("getAirport", () => {
  it("returns text + UI resource for a known ICAO", async () => {
    const r = await getAirport({ icao: "LSZH" });
    expect(r.content[0].type).toBe("text");
    expect(r.content[0].text).toContain("LSZH");
    expect(r.content[1].type).toBe("resource");
    expect(r.content[1].resource.uri).toBe("ui://aviation/airport/LSZH");
    expect(r.content[1].resource.text).toContain("Zurich");
    expect(r.content[1].resource.text).toContain("12139");
  });

  it("includes a linked alert when present", async () => {
    const r = await getAirport({ icao: "LSZH" });
    expect(r.content[1].resource.text).toContain("Hantavirus");
  });

  it("throws a helpful error for unknown ICAO", async () => {
    await expect(getAirport({ icao: "XXXX" })).rejects.toThrow(/Unknown airport/);
  });

  it("escapes HTML in fixture-derived strings", async () => {
    // sanity: smoke test that rendered output never contains raw `<script>`
    const r = await getAirport({ icao: "LSZH" });
    expect(r.content[1].resource.text).not.toContain("<script>alert");
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement getAirport**

`src/tools/getAirport.ts`:

```ts
import { findAirport, alertsNearAirport } from "../data.js";
import { loadTemplate, renderTemplate, buildResource, type UIResource } from "../render.js";
import { haversineKm } from "../geo.js";
import type { Airport } from "../types.js";

export interface GetAirportArgs {
  icao: string;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | UIResource>;
}

function ratingDeg(level: number): number {
  return Math.max(0, Math.min(5, level)) * 72;
}

function runwaySummary(a: Airport): string {
  const r = a.runways[0];
  if (!r) return "Not published";
  return `RW${r.id} · ${r.length_ft.toLocaleString()} ft × ${r.width_ft} ft @ ${r.elevation_ft.toLocaleString()} ft`;
}

function groundHandlingSummary(a: Airport): string {
  const parts: string[] = [];
  parts.push(a.ground_handling.slots_required ? "Slots Required" : "No Slots Required");
  parts.push(a.ground_handling.handling_required ? "Handling Required" : "Handling Not Required");
  return parts.join(" · ");
}

export async function getAirport(args: GetAirportArgs): Promise<ToolResult> {
  const airport = findAirport(args.icao);
  if (!airport) {
    throw new Error(`Unknown airport: ${args.icao}. Try a 4-letter ICAO like LSZH (Zurich) or KJFK (New York).`);
  }
  const alerts = alertsNearAirport(airport, 50);
  const top = alerts[0];
  const themeCss = loadTemplate("theme.css");
  const tpl = loadTemplate("airport-card.html");

  const data: Record<string, unknown> = {
    theme_css: themeCss,
    icao: airport.icao,
    iata: airport.iata,
    name: airport.name,
    country: airport.country,
    rating: airport.rating,
    rating_label: airport.rating_label,
    rating_deg: ratingDeg(airport.rating),
    covid_inbound: airport.covid_inbound,
    covid_domestic: airport.covid_domestic,
    covid_deg: ratingDeg(Math.max(airport.covid_inbound, airport.covid_domestic)),
    city_risk_label: airport.city_risk_label,
    city_risk_deg: ratingDeg(airport.city_risk),
    medical_risk_label: airport.medical_risk_label,
    medical_risk_deg: ratingDeg(airport.medical_risk),
    date_of_publish: airport.date_of_publish,
    runway_summary: runwaySummary(airport),
    ground_handling_summary: groundHandlingSummary(airport),
    alert: top
      ? {
          alert_time: new Date(top.active_from).toUTCString().slice(5, 22),
          alert_text: top.headline,
          alert_distance_km: Math.round(
            haversineKm([airport.lng, airport.lat], [top.lng, top.lat])
          ),
        }
      : null,
  };

  const html = renderTemplate(tpl, data);
  const summary =
    `${airport.icao} (${airport.iata}) — ${airport.name}, ${airport.country}. ` +
    `Rating ${airport.rating} (${airport.rating_label}). ` +
    (alerts.length > 0 ? `${alerts.length} active alert(s) within 50 km.` : "No active alerts nearby.");

  return {
    content: [
      { type: "text", text: summary },
      buildResource("airport", airport.icao, html),
    ],
  };
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/tools/getAirport.ts aviation-mcp-ui/tests/tools.test.ts
git commit -m "add getAirport tool with tests"
```

---

## Task 18: FIR-detail template

**Files:**
- Create: `aviation-mcp-ui/templates/fir-detail.html`

Substitutions: `{{id}}`, `{{name}}`, `{{country}}`, `{{issued_by}}`, `{{flight_level_floor}}`, `{{flight_level_ceiling}}`, `{{weaponry_range_floor}}`, `{{date_of_publish}}`, sections `{{#weapons_grid}}{{name}}{{active_class}}{{/weapons_grid}}`, `{{#cz_warnings}}{{.}}{{/cz_warnings}}`, optional `{{?hostile_intercepts}}...{{/hostile_intercepts}}`, optional `{{?top_alert}}...{{/top_alert}}` with `{{headline}}`, `{{severity}}`.

- [ ] **Step 1: Write template**

`templates/fir-detail.html`:

```html
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{id}} {{name}}</title>
<style>{{{theme_css}}}
.fir-frame { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; max-width: 720px; margin: 24px auto; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.fir-head { border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 18px; }
.fir-id { font-size: 28px; font-weight: 700; color: #fff; letter-spacing: 1px; }
.fir-name { color: var(--text-muted); letter-spacing: 2px; font-size: 13px; text-transform: uppercase; }
.fir-country-row { display:flex; justify-content:space-between; align-items:center; padding: 8px 0; }
.fir-country { color: var(--text); font-size: 13px; }
.fir-publish { font-size: 10px; color: var(--text-dim); letter-spacing:2px; }
.fir-section-title { font-size: 11px; letter-spacing: 2px; color: var(--text-muted); margin: 18px 0 10px; text-transform: uppercase; }
.fir-bar-row { display:flex; align-items:center; gap: 18px; margin-bottom: 14px; }
.fir-bar-label { font-size: 10px; letter-spacing: 1px; color: var(--text-muted); min-width: 160px; }
.fir-bar-label .dot { background: var(--danger); }
.fir-bar { flex:1; height: 22px; background: repeating-linear-gradient(45deg, rgba(217,77,61,0.25) 0 8px, rgba(217,77,61,0.4) 8px 16px); border: 1px solid rgba(217,77,61,0.5); border-radius: 2px; position: relative; }
.fir-bar.weapon { background: repeating-linear-gradient(45deg, rgba(217,77,61,0.45) 0 8px, rgba(217,77,61,0.65) 8px 16px); border-color: rgba(217,77,61,0.8); }
.fir-bar-text { position:absolute; right: 10px; top: 4px; font-size: 11px; color: var(--text); letter-spacing: 1px; }
.fir-bar-floor { position:absolute; left: 10px; top: 4px; font-size: 11px; color: var(--text); letter-spacing: 1px; }
.fir-weapons { display:grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-bottom: 18px; }
.fir-weapon { padding: 10px 8px; text-align: center; font-size: 11px; border: 1px solid var(--border); color: var(--text-dim); border-radius: 2px; }
.fir-weapon.active { color: var(--text); background: var(--panel); border-color: var(--text-muted); }
.fir-cz { background: var(--panel); padding: 10px 14px; margin-bottom: 10px; font-size: 12px; }
.fir-issued { display:flex; justify-content:space-between; padding: 12px 0; border-top: 1px solid var(--border); font-size: 11px; letter-spacing: 1px; color: var(--text-muted); }
.fir-alert { background: var(--panel); border-left: 3px solid var(--danger); padding: 10px 14px; margin-top: 14px; }
.fir-alert .label { color: var(--danger); font-weight: 600; }
.fir-cta { display:block; text-align:center; padding: 14px; margin-top: 18px; background: var(--accent); color: #062626; font-size: 13px; letter-spacing: 2px; border-radius: 24px; font-weight: 600; }
</style>
</head>
<body>
<div class="fir-frame">
  <div class="fir-head">
    <div class="fir-id">{{id}}</div>
    <div class="fir-name">{{name}}</div>
  </div>
  <div class="fir-country-row">
    <span class="fir-country">{{country}}</span>
    <span class="fir-publish">DATE OF PUBLISH {{date_of_publish}}</span>
  </div>
  {{?top_alert}}
  <div class="fir-alert">
    <div class="label">{{severity}}</div>
    <div>{{headline}}</div>
  </div>
  {{/top_alert}}
  <div class="fir-section-title">FLIGHT LEVEL RISK</div>
  <div class="fir-bar-row">
    <span class="fir-bar-label"><span class="dot"></span>WEAPONRY RANGE</span>
    <div class="fir-bar weapon"><span class="fir-bar-floor">FL-{{weaponry_range_floor}}</span></div>
  </div>
  <div class="fir-bar-row">
    <span class="fir-bar-label"><span class="dot"></span>FLIGHT RESTRICTIONS</span>
    <div class="fir-bar"><span class="fir-bar-floor">FL-{{flight_level_floor}}</span><span class="fir-bar-text">FL-{{flight_level_ceiling}}</span></div>
  </div>
  <div class="fir-section-title">AIRCRAFT WEAPONRY RISK</div>
  <div class="fir-weapons">
    {{#weapons_grid}}<div class="fir-weapon {{active_class}}">{{name}}</div>{{/weapons_grid}}
  </div>
  {{?hostile_intercepts}}<div class="muted" style="font-size:11px; margin-bottom: 14px;">Hostile Aircraft Intercepts</div>{{/hostile_intercepts}}
  {{#cz_warnings}}<div class="fir-cz">{{.}}</div>{{/cz_warnings}}
  <div class="fir-issued">
    <span>CZ Warnings</span>
    <span>ISSUED BY {{issued_by}}</span>
  </div>
  <div class="fir-cta">▤ GENERATE REPORT</div>
</div>
</body></html>
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/templates/fir-detail.html
git commit -m "add fir-detail template"
```

---

## Task 19: getFir tool

**Files:**
- Create: `aviation-mcp-ui/src/tools/getFir.ts`
- Modify: `aviation-mcp-ui/tests/tools.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/tools.test.ts`:

```ts
import { getFir } from "../src/tools/getFir.js";

describe("getFir", () => {
  it("returns text + UI resource for a known FIR", async () => {
    const r = await getFir({ fir_id: "ORBB" });
    expect(r.content[0].text).toContain("Baghdad");
    expect(r.content[1].resource.uri).toBe("ui://aviation/fir/ORBB");
    const html = r.content[1].resource.text;
    expect(html).toContain("Baghdad FIR");
    expect(html).toContain("FL-180");
    expect(html).toContain("FL-600");
    expect(html).toContain("MANPADS");
  });

  it("throws for unknown FIR", async () => {
    await expect(getFir({ fir_id: "XXXX" })).rejects.toThrow(/Unknown FIR/);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement getFir**

`src/tools/getFir.ts`:

```ts
import { findFir, alertsLinkedToFir } from "../data.js";
import { loadTemplate, renderTemplate, buildResource } from "../render.js";
import type { ToolResult } from "./getAirport.js";
import type { Weapon } from "../types.js";

export interface GetFirArgs {
  fir_id: string;
}

const ALL_WEAPONS: { id: Weapon; name: string }[] = [
  { id: "small_arms", name: "Small Arms" },
  { id: "aaa_light", name: "AAA (Light)" },
  { id: "aaa", name: "AAA" },
  { id: "manpads", name: "MANPADS" },
  { id: "manpads_advanced", name: "MANPADS (Advanced)" },
  { id: "rpg", name: "RPG" },
  { id: "atgm", name: "ATGM" },
  { id: "sam", name: "SAM" },
  { id: "sam_mobile", name: "SAM (Mobile)" },
  { id: "sam_advanced", name: "SAM (Advanced)" },
];

export async function getFir(args: GetFirArgs): Promise<ToolResult> {
  const fir = findFir(args.fir_id);
  if (!fir) {
    throw new Error(`Unknown FIR: ${args.fir_id}. Try ORBB (Baghdad), OIIX (Tehran), or LSAS (Switzerland).`);
  }
  const linked = alertsLinkedToFir(fir.id);
  const top = linked[0];
  const themeCss = loadTemplate("theme.css");
  const tpl = loadTemplate("fir-detail.html");

  const weapons_grid = ALL_WEAPONS.map((w) => ({
    name: w.name,
    active_class: fir.weapons.includes(w.id) ? "active" : "",
  }));

  const data = {
    theme_css: themeCss,
    id: fir.id,
    name: fir.name,
    country: fir.country,
    issued_by: fir.issued_by,
    flight_level_floor: fir.flight_level_floor,
    flight_level_ceiling: fir.flight_level_ceiling,
    weaponry_range_floor: fir.weaponry_range_floor,
    date_of_publish: top?.active_from?.slice(0, 10) ?? "",
    weapons_grid,
    cz_warnings: fir.cz_warnings,
    hostile_intercepts: fir.hostile_intercepts,
    top_alert: top
      ? { headline: top.headline, severity: top.severity.toUpperCase() }
      : null,
  };

  const html = renderTemplate(tpl, data);
  const summary =
    `${fir.id} — ${fir.name} (${fir.country}). ` +
    `Risk envelope FL${fir.flight_level_floor}-FL${fir.flight_level_ceiling}, weaponry range floor FL${fir.weaponry_range_floor}. ` +
    `${fir.weapons.length} weapon categor${fir.weapons.length === 1 ? "y" : "ies"} active. ` +
    (linked.length > 0 ? `${linked.length} linked alert(s).` : "No linked alerts.");

  return {
    content: [{ type: "text", text: summary }, buildResource("fir", fir.id, html)],
  };
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/tools/getFir.ts aviation-mcp-ui/tests/tools.test.ts
git commit -m "add getFir tool with tests"
```

---

## Task 20: Fleet-grid template

**Files:**
- Create: `aviation-mcp-ui/templates/fleet-grid.html`

Substitutions: `{{count}}`, `{{issue_count}}`, sections `{{#aircraft}}...{{/aircraft}}` with each item having `{{tail_number}}`, `{{type}}`, `{{callsign}}`, `{{status_text}}`, `{{dot_class}}`, `{{location_text}}`, `{{?has_issue}}...{{/has_issue}}` with `{{issue_headline}}`, `{{issue_detail}}`, `{{issue_severity_class}}`, `{{?pulse}}` true if status is "issue".

- [ ] **Step 1: Write template**

`templates/fleet-grid.html`:

```html
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fleet Status</title>
<style>{{{theme_css}}}
.fl-frame { padding: 24px; max-width: 1100px; margin: 0 auto; }
.fl-head { display:flex; justify-content:space-between; align-items:flex-end; padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px; }
.fl-title { font-size: 22px; color: #fff; letter-spacing: 1px; }
.fl-meta { color: var(--text-muted); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }
.fl-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.fl-card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px; position: relative; }
.fl-card.alert { border-color: var(--danger); }
.fl-card.alert::before { content: ""; position:absolute; inset: 0; border-radius: 6px; pointer-events:none; box-shadow: 0 0 0 rgba(217,77,61,0.0); animation: pulse-red 1.6s ease-in-out infinite; }
.fl-tail { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: 1px; }
.fl-type { font-size: 11px; color: var(--text-muted); letter-spacing: 2px; margin-bottom: 10px; text-transform: uppercase; }
.fl-status { font-size: 11px; letter-spacing: 1px; color: var(--text); }
.fl-loc { font-size: 11px; color: var(--text-muted); margin-top: 4px; letter-spacing: 1px; }
.fl-issue { background: rgba(217,77,61,0.12); border-left: 3px solid var(--danger); padding: 8px 10px; margin-top: 10px; font-size: 12px; }
.fl-issue-title { font-weight: 600; color: var(--danger); font-size: 11px; letter-spacing: 1px; margin-bottom: 4px; }
</style>
</head>
<body>
<div class="fl-frame">
  <div class="fl-head">
    <div>
      <div class="fl-title">FLEET STATUS</div>
      <div class="fl-meta">{{count}} AIRCRAFT · {{issue_count}} WITH ACTIVE ISSUES</div>
    </div>
  </div>
  <div class="fl-grid">
    {{#aircraft}}
    <div class="fl-card{{?pulse}} alert{{/pulse}}">
      <div class="fl-tail">{{tail_number}}</div>
      <div class="fl-type">{{type}} · {{callsign}}</div>
      <div class="fl-status"><span class="dot {{dot_class}}"></span>{{status_text}}</div>
      <div class="fl-loc">{{location_text}}</div>
      {{?has_issue}}
      <div class="fl-issue">
        <div class="fl-issue-title">{{issue_severity_class}}</div>
        <div>{{issue_headline}}</div>
        <div class="muted" style="margin-top:4px;">{{issue_detail}}</div>
      </div>
      {{/has_issue}}
    </div>
    {{/aircraft}}
  </div>
</div>
</body></html>
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/templates/fleet-grid.html
git commit -m "add fleet-grid template"
```

---

## Task 21: getFleetStatus tool

**Files:**
- Create: `aviation-mcp-ui/src/tools/getFleetStatus.ts`
- Modify: `aviation-mcp-ui/tests/tools.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/tools.test.ts`:

```ts
import { getFleetStatus } from "../src/tools/getFleetStatus.js";

describe("getFleetStatus", () => {
  it("returns the full fleet with at least one issue card", async () => {
    const r = await getFleetStatus();
    expect(r.content[0].text).toMatch(/aircraft/i);
    const html = r.content[1].resource.text;
    expect(html).toContain("N737AZ");
    expect(html).toContain("N787LR");
    expect(html).toContain("DIVERTING");
    expect(html).toContain("alert"); // pulsing card class
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement getFleetStatus**

`src/tools/getFleetStatus.ts`:

```ts
import { getFleet, findAirport } from "../data.js";
import { loadTemplate, renderTemplate, buildResource } from "../render.js";
import type { ToolResult } from "./getAirport.js";
import type { Aircraft } from "../types.js";

function statusText(a: Aircraft): string {
  switch (a.status) {
    case "in_flight":   return "In Flight";
    case "on_ground":   return "On Ground";
    case "maintenance": return "Maintenance";
    case "issue":       return "Active Issue";
  }
}

function dotClass(a: Aircraft): string {
  switch (a.status) {
    case "in_flight":   return "dot-green";
    case "on_ground":   return "dot-grey";
    case "maintenance": return "dot-amber";
    case "issue":       return "dot-red";
  }
}

function locationText(a: Aircraft): string {
  if (a.location.icao) {
    const ap = findAirport(a.location.icao);
    return ap ? `${a.location.icao} · ${ap.name}` : a.location.icao;
  }
  if (a.location.lat !== undefined && a.location.lng !== undefined) {
    const route = a.origin && a.dest ? ` · ${a.origin}→${a.dest}` : "";
    return `${a.location.lat.toFixed(2)}, ${a.location.lng.toFixed(2)}${route}`;
  }
  return "Unknown";
}

export async function getFleetStatus(): Promise<ToolResult> {
  const fleet = getFleet();
  const themeCss = loadTemplate("theme.css");
  const tpl = loadTemplate("fleet-grid.html");
  const issueCount = fleet.filter((a) => a.status === "issue").length;

  const aircraft = fleet.map((a) => ({
    tail_number: a.tail_number,
    type: a.type,
    callsign: a.callsign,
    status_text: statusText(a),
    dot_class: dotClass(a),
    location_text: locationText(a),
    pulse: a.status === "issue",
    has_issue: !!a.issue,
    issue_headline: a.issue?.headline ?? "",
    issue_detail: a.issue?.detail ?? "",
    issue_severity_class: a.issue?.severity?.toUpperCase() ?? "",
  }));

  const data = {
    theme_css: themeCss,
    count: fleet.length,
    issue_count: issueCount,
    aircraft,
  };

  const html = renderTemplate(tpl, data);
  const summary =
    `Fleet: ${fleet.length} aircraft. ${issueCount} with active issues, ` +
    `${fleet.filter((a) => a.status === "in_flight").length} in flight, ` +
    `${fleet.filter((a) => a.status === "on_ground").length} on ground, ` +
    `${fleet.filter((a) => a.status === "maintenance").length} in maintenance.`;

  return {
    content: [{ type: "text", text: summary }, buildResource("fleet", "all", html)],
  };
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/tools/getFleetStatus.ts aviation-mcp-ui/tests/tools.test.ts
git commit -m "add getFleetStatus tool with tests"
```

---

## Task 22: Route-map template

**Files:**
- Create: `aviation-mcp-ui/templates/route-map.html`

Substitutions: `{{origin}}`, `{{dest}}`, `{{origin_name}}`, `{{dest_name}}`, `{{distance_km}}`, `{{alert_count}}`, `{{{route_geojson}}}` (raw JSON), `{{{alerts_geojson}}}` (raw JSON), `{{{alerts_list_html}}}` (raw — pre-rendered list items).

The map uses Leaflet from CDN inside the iframe; tiles are CartoDB dark matter (free, no key).

- [ ] **Step 1: Write template**

`templates/route-map.html`:

```html
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{origin}} → {{dest}}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<style>{{{theme_css}}}
html, body { height: 100%; }
.map-wrap { display:flex; height: 600px; max-width: 1200px; margin: 16px auto; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
#map { flex: 1; background: var(--bg); }
.side { width: 320px; background: var(--panel); border-left: 1px solid var(--border); padding: 16px; overflow-y: auto; }
.side h3 { margin: 0 0 4px; font-size: 16px; color: #fff; letter-spacing: 1px; }
.side .meta { font-size: 11px; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 14px; text-transform: uppercase; }
.alert-item { padding: 10px 0; border-top: 1px solid var(--border); }
.alert-item:first-of-type { border-top: 0; }
.alert-headline { font-size: 12px; color: var(--text); margin-bottom: 4px; }
.alert-meta { font-size: 10px; color: var(--text-muted); letter-spacing: 1px; }
.leaflet-container { background: var(--bg); }
</style>
</head>
<body>
<div class="map-wrap">
  <div id="map"></div>
  <div class="side">
    <h3>{{origin}} → {{dest}}</h3>
    <div class="meta">{{origin_name}} TO {{dest_name}} · {{distance_km}} KM · {{alert_count}} ALERT(S) ON CORRIDOR</div>
    {{{alerts_list_html}}}
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
(function(){
  var route = {{{route_geojson}}};
  var alerts = {{{alerts_geojson}}};
  var map = L.map('map', { zoomControl: true, attributionControl: false }).setView([30, 0], 3);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);
  var line = L.polyline(route.map(function(p){ return [p[1], p[0]]; }), { color: '#3ddbd9', weight: 3, opacity: 0.9 }).addTo(map);
  L.circleMarker([route[0][1], route[0][0]], { color: '#3ddbd9', fillColor: '#3ddbd9', fillOpacity: 1, radius: 6 }).addTo(map);
  L.circleMarker([route[route.length-1][1], route[route.length-1][0]], { color: '#3ddbd9', fillColor: '#3ddbd9', fillOpacity: 1, radius: 6 }).addTo(map);
  var sevColor = { critical: '#d94d3d', warning: '#d9a83d', notice: '#3ddbd9', advisory: '#7a99a8' };
  alerts.forEach(function(a){
    L.circleMarker([a.lat, a.lng], {
      color: sevColor[a.severity] || '#7a99a8',
      fillColor: sevColor[a.severity] || '#7a99a8',
      fillOpacity: 0.85,
      radius: 8,
      weight: 2
    }).addTo(map).bindPopup('<div style="color:#0b1419"><b>'+a.headline+'</b><br/><small>'+a.severity.toUpperCase()+'</small></div>');
  });
  try { map.fitBounds(line.getBounds().pad(0.15)); } catch(e) {}
})();
</script>
</body></html>
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/templates/route-map.html
git commit -m "add route-map template (Leaflet + CartoDB dark)"
```

---

## Task 23: getRouteRisks tool

**Files:**
- Create: `aviation-mcp-ui/src/tools/getRouteRisks.ts`
- Modify: `aviation-mcp-ui/tests/tools.test.ts`

- [ ] **Step 1: Add failing test**

Append to `tests/tools.test.ts`:

```ts
import { getRouteRisks } from "../src/tools/getRouteRisks.js";

describe("getRouteRisks", () => {
  it("returns route map with corridor alerts for PHX-JFK", async () => {
    const r = await getRouteRisks({ origin: "KPHX", dest: "KJFK" });
    expect(r.content[0].text).toContain("KPHX");
    expect(r.content[0].text).toContain("KJFK");
    const html = r.content[1].resource.text;
    expect(html).toContain("L.polyline");
    expect(html.toLowerCase()).toContain("cartocdn");
    expect(html).toContain("KPHX");
  });

  it("throws for unknown origin", async () => {
    await expect(getRouteRisks({ origin: "XXXX", dest: "KJFK" })).rejects.toThrow(/Unknown airport/);
  });

  it("throws for unknown dest", async () => {
    await expect(getRouteRisks({ origin: "KJFK", dest: "XXXX" })).rejects.toThrow(/Unknown airport/);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement getRouteRisks**

`src/tools/getRouteRisks.ts`:

```ts
import { findAirport, getRoute, alertsAlongRoute } from "../data.js";
import { haversineKm } from "../geo.js";
import { loadTemplate, renderTemplate, buildResource, escapeHtml } from "../render.js";
import type { ToolResult } from "./getAirport.js";
import type { Alert } from "../types.js";

export interface GetRouteRisksArgs {
  origin: string;
  dest: string;
}

function alertsListHtml(alerts: Alert[]): string {
  if (alerts.length === 0) {
    return '<div class="alert-item"><div class="alert-headline muted">No active alerts on this corridor.</div></div>';
  }
  return alerts
    .map(
      (a) =>
        `<div class="alert-item">` +
        `<div class="alert-headline">${escapeHtml(a.headline)}</div>` +
        `<div class="alert-meta sev-${escapeHtml(a.severity)}">${escapeHtml(a.severity.toUpperCase())} · ${escapeHtml(a.region)}</div>` +
        `</div>`
    )
    .join("");
}

export async function getRouteRisks(args: GetRouteRisksArgs): Promise<ToolResult> {
  const o = findAirport(args.origin);
  const d = findAirport(args.dest);
  if (!o) throw new Error(`Unknown airport: ${args.origin}`);
  if (!d) throw new Error(`Unknown airport: ${args.dest}`);

  const route = getRoute(args.origin, args.dest);
  const alerts = alertsAlongRoute(route.waypoints, 250);
  const distance = Math.round(
    haversineKm([o.lng, o.lat], [d.lng, d.lat])
  );

  const themeCss = loadTemplate("theme.css");
  const tpl = loadTemplate("route-map.html");

  const data = {
    theme_css: themeCss,
    origin: o.icao,
    dest: d.icao,
    origin_name: o.name,
    dest_name: d.name,
    distance_km: distance.toLocaleString(),
    alert_count: alerts.length,
    route_geojson: JSON.stringify(route.waypoints),
    alerts_geojson: JSON.stringify(
      alerts.map((a) => ({
        lat: a.lat,
        lng: a.lng,
        severity: a.severity,
        headline: a.headline.replace(/[<>"']/g, ""),
      }))
    ),
    alerts_list_html: alertsListHtml(alerts),
  };

  const html = renderTemplate(tpl, data);
  const summary =
    `Route ${o.icao} (${o.name}) → ${d.icao} (${d.name}), ${distance.toLocaleString()} km. ` +
    (alerts.length > 0
      ? `${alerts.length} active alert(s) within 250 km of the corridor: ${alerts
          .slice(0, 3)
          .map((a) => `[${a.severity}] ${a.headline.slice(0, 60)}…`)
          .join("; ")}`
      : "No active alerts on the corridor.");

  return {
    content: [
      { type: "text", text: summary },
      buildResource("route", `${o.icao}-${d.icao}`, html),
    ],
  };
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd aviation-mcp-ui && npx vitest run tests/tools.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aviation-mcp-ui/src/tools/getRouteRisks.ts aviation-mcp-ui/tests/tools.test.ts
git commit -m "add getRouteRisks tool with tests"
```

---

## Task 24: MCP server bootstrap

**Files:**
- Create: `aviation-mcp-ui/src/server.ts`

- [ ] **Step 1: Write server.ts**

`src/server.ts`:

```ts
#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { getAirport } from "./tools/getAirport.js";
import { getFir } from "./tools/getFir.js";
import { getFleetStatus } from "./tools/getFleetStatus.js";
import { getRouteRisks } from "./tools/getRouteRisks.js";

const server = new Server(
  { name: "aviation-mcp-ui", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

const TOOLS = [
  {
    name: "get_airport",
    description:
      "Show a branded airport detail card for the given ICAO. Includes ratings (overall, COVID, city risk, medical), runway specs, ground handling, and any nearby active alerts. Use when the user asks about a specific airport.",
    inputSchema: {
      type: "object",
      properties: { icao: { type: "string", description: "4-letter ICAO code, e.g. LSZH" } },
      required: ["icao"],
    },
  },
  {
    name: "get_fir",
    description:
      "Show a Flight Information Region threat panel: weaponry range, flight-level risk band, weapon-category grid, CZ warnings. Use when the user asks about a FIR by code (e.g. ORBB, OIIX).",
    inputSchema: {
      type: "object",
      properties: { fir_id: { type: "string", description: "FIR identifier, e.g. ORBB (Baghdad)" } },
      required: ["fir_id"],
    },
  },
  {
    name: "get_fleet_status",
    description:
      "Show a status grid for the operator's fleet — every aircraft, its status (in_flight / on_ground / maintenance / issue), location, and any active issue. Use when the user asks about the fleet, planes, or whether any aircraft are experiencing issues.",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "get_route_risks",
    description:
      "Show a dark map with the great-circle route between two ICAOs, alert pins along the corridor (within 250 km), and a side panel listing each alert. Use when the user asks about risks/alerts/threats along a flight route.",
    inputSchema: {
      type: "object",
      properties: {
        origin: { type: "string", description: "Origin ICAO" },
        dest: { type: "string", description: "Destination ICAO" },
      },
      required: ["origin", "dest"],
    },
  },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args } = req.params;
  try {
    switch (name) {
      case "get_airport":
        return await getAirport(args as { icao: string });
      case "get_fir":
        return await getFir(args as { fir_id: string });
      case "get_fleet_status":
        return await getFleetStatus();
      case "get_route_risks":
        return await getRouteRisks(args as { origin: string; dest: string });
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      isError: true,
      content: [{ type: "text", text: message }],
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

- [ ] **Step 2: Build**

Run: `cd aviation-mcp-ui && npm run build`
Expected: `dist/` populated; no TypeScript errors. If MCP SDK API surface differs from imports above, adjust import paths from the actual `@modelcontextprotocol/sdk` exports (the SDK exposes `Server` from `server/index.js` and the schemas from `types.js` in current versions).

- [ ] **Step 3: Smoke-run**

Run: `cd aviation-mcp-ui && echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node dist/server.js | head -c 500`
Expected: JSON output containing the 4 tool names.

- [ ] **Step 4: Commit**

```bash
git add aviation-mcp-ui/src/server.ts
git commit -m "add MCP server entrypoint with 4 tools"
```

---

## Task 25: README

**Files:**
- Create: `aviation-mcp-ui/README.md`

- [ ] **Step 1: Write README**

`aviation-mcp-ui/README.md`:

```markdown
# aviation-mcp-ui

Stakeholder-demo MCP-UI server that surfaces aviation threat data with a branded dark/cyan dashboard inside any MCP-UI-compatible host (Claude Desktop, web/browser chat, etc.).

## Tools

- `get_airport(icao)` — branded airport detail card (ratings, runway, alerts).
- `get_fir(fir_id)` — FIR threat panel (weaponry range, flight-level risk, CZ warnings).
- `get_fleet_status()` — operator fleet grid with status indicators.
- `get_route_risks(origin, dest)` — dark map with route polyline and corridor alerts.

All tools return MCP-UI `rawHtml` resources; data is bundled JSON fixtures (no live APIs).

## Install

```bash
git clone <this repo>
cd aviation-mcp-ui
npm install
npm run build
```

## Use with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "aviation": {
      "command": "node",
      "args": ["/absolute/path/to/aviation-mcp-ui/dist/server.js"]
    }
  }
}
```

Restart Claude Desktop.

## Demo prompts

After install, paste any of these into chat:

1. **Airport** — "Tell me about LSZH."
2. **FIR** — "What's happening in the Baghdad FIR (ORBB)?"
3. **Fleet** — "Are any planes in my fleet experiencing issues?"
4. **Route** — "Show me security risks along my flight from KPHX to KJFK."

Each renders a branded panel inside the chat.

## Develop

```bash
npm test          # vitest run
npm run test:watch
npm run dev       # tsc -w
```
```

- [ ] **Step 2: Commit**

```bash
git add aviation-mcp-ui/README.md
git commit -m "add README with install + demo prompts"
```

---

## Task 26: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd aviation-mcp-ui && npm test`
Expected: ALL tests pass (geo, render, data, tools).

- [ ] **Step 2: Build cleanly**

Run: `cd aviation-mcp-ui && rm -rf dist && npm run build && ls dist/templates dist/fixtures`
Expected: `dist/server.js` exists; `dist/templates/` contains 4 HTML + theme.css; `dist/fixtures/` contains 5 JSON files.

- [ ] **Step 3: List tools via stdio**

Run: `cd aviation-mcp-ui && (echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'; echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}') | node dist/server.js`
Expected: JSON-RPC responses containing the 4 tool definitions.

- [ ] **Step 4: Manually inspect one rendered HTML**

Run: `cd aviation-mcp-ui && node -e "import('./dist/tools/getAirport.js').then(m => m.getAirport({icao:'LSZH'})).then(r => console.log(r.content[1].resource.text.slice(0,400)))"`
Expected: HTML beginning `<!DOCTYPE html>` containing `LSZH` and `Zurich`.

- [ ] **Step 5: Commit a "demo-ready" tag**

```bash
cd /Users/bigbrain/Dev/hermes/.claude/worktrees/dazzling-albattani-18b127
git tag aviation-mcp-ui-v0.1.0
```

---

## Summary

26 tasks across 4 module groups:

- **Foundation (1-2):** scaffold + types
- **Pure logic, TDD (3-8):** geo, render
- **Fixtures (10-14):** airports, FIRs, alerts, fleet, routes
- **Data layer (15):** lookups + corridor query
- **Tools + templates, paired (16-23):** airport, fir, fleet, route — each template + tool
- **Wiring (24-25):** server bootstrap + README
- **Verification (26):** full test suite + smoke + build

Pure-logic modules use TDD strictly. Tools verify shape + key content; templates are visual and verified manually via the rendered HTML in tests + the demo-prompt smoke run.
