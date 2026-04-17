# Infrastructure Patterns

Reusable shapes and line styles for infrastructure / systems-integration diagrams (smart cities, IoT networks, industrial systems, multi-domain architectures).

## Layout pattern: hub-spoke

- **Central hub**: Hexagon or circle representing the integration platform
- **Radiating connections**: Data lines from hub to each subsystem with connection dots
- **Subsystem sections**: Each system (power, water, transport) in its own region
- **Dashboard on top**: Optional UI mockup showing a unified view (see `dashboard-patterns.md`)

```xml
<!-- Central hub (hexagon) -->
<polygon class="iot-hex" points="0,-45 39,-22 39,22 0,45 -39,22 -39,-22"/>

<!-- Data lines with connection dots -->
<path class="data-line" d="M 321 248 L 200 248 L 120 380" stroke-dasharray="4 3"/>
<circle cx="321" cy="248" r="4" fill="#7F77DD"/>
```

## Semantic line styles

Use a dedicated CSS class per subsystem so every diagram reads the same way:

```css
.data-line  { stroke: #7F77DD; stroke-width: 2; fill: none; stroke-dasharray: 4 3; }
.power-line { stroke: #EF9F27; stroke-width: 2; fill: none; }
.water-pipe { stroke: #378ADD; stroke-width: 4; stroke-linecap: round; fill: none; }
.road       { stroke: #888780; stroke-width: 8; stroke-linecap: round; fill: none; }
```

## Power systems

**Solar panel (angled):**
```xml
<polygon class="solar-panel" points="0,25 35,8 38,12 3,29"/>
<line class="solar-frame" x1="12" y1="22" x2="24" y2="13"/>
```

**Wind turbine:**
```xml
<polygon class="wind-tower" points="20,70 30,70 28,25 22,25"/>
<circle class="wind-hub" cx="25" cy="18" r="5"/>
<ellipse class="wind-blade" cx="25" cy="5" rx="3" ry="13"/>
<ellipse class="wind-blade" cx="14" cy="26" rx="3" ry="13" transform="rotate(-120, 25, 18)"/>
<ellipse class="wind-blade" cx="36" cy="26" rx="3" ry="13" transform="rotate(120, 25, 18)"/>
```

**Battery with charge level:**
```xml
<rect class="battery" x="0" y="0" width="45" height="65" rx="5"/>
<rect x="10" y="-6" width="10" height="8" rx="2" fill="#27500A"/> <!-- terminal -->
<rect class="battery-level" x="5" y="12" width="35" height="48" rx="3"/> <!-- fill level -->
```

**Power pylon:**
```xml
<polygon class="pylon" points="30,0 35,0 40,60 25,60"/>
<line x1="15" y1="10" x2="45" y2="10" stroke="#5F5E5A" stroke-width="3"/>
<circle cx="18" cy="10" r="3" fill="#FAEEDA" stroke="#854F0B"/> <!-- insulator -->
```

## Water systems

**Reservoir/dam:**
```xml
<polygon class="reservoir-wall" points="0,60 10,0 70,0 80,60"/>
<polygon class="water" points="12,10 68,10 68,55 75,55 75,58 5,58 5,55 12,55"/>
<!-- Wave effect -->
<path d="M 15 25 Q 25 22 35 25 Q 45 28 55 25" fill="none" stroke="#378ADD" opacity="0.5"/>
```

**Treatment tank:**
```xml
<ellipse class="treatment-tank" cx="35" cy="45" rx="30" ry="18"/>
<rect class="treatment-tank" x="5" y="20" width="60" height="25"/>
<!-- Bubbles -->
<circle cx="20" cy="32" r="2" fill="#378ADD" opacity="0.6"/>
```

**Pipe with joint and valve:**
```xml
<path class="pipe" d="M 80 85 L 110 85"/>
<circle class="pipe-joint" cx="110" cy="85" r="8"/>
<circle class="valve" cx="95" cy="85" r="6"/>
```

## Transport systems

**Road with lane markings:**
```xml
<line class="road" x1="0" y1="50" x2="170" y2="50"/>
<line class="road-mark" x1="10" y1="50" x2="160" y2="50"/>
```

**Traffic light:**
```xml
<rect class="traffic-light" x="0" y="0" width="14" height="32" rx="3"/>
<circle class="light-red" cx="7" cy="8" r="4"/>
<circle class="light-off" cx="7" cy="16" r="4"/>
<circle class="light-green" cx="7" cy="24" r="4"/>
```

**Bus:**
```xml
<rect class="bus" x="0" y="0" width="55" height="28" rx="6"/>
<rect class="bus-window" x="5" y="5" width="12" height="12" rx="2"/>
<circle cx="14" cy="30" r="6" fill="#2C2C2A"/> <!-- wheel -->
<circle cx="14" cy="30" r="3" fill="#5F5E5A"/> <!-- hubcap -->
```

## Full CSS block (add to the host page or inline <style>)

```css
/* Power */
.solar-panel   { fill: #3C3489; stroke: #534AB7; stroke-width: 0.5; }
.wind-tower    { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.wind-blade    { fill: #F1EFE8; stroke: #888780; stroke-width: 0.5; }
.battery       { fill: #27500A; stroke: #3B6D11; stroke-width: 1.5; }
.battery-level { fill: #97C459; }
.power-line    { stroke: #EF9F27; stroke-width: 2; fill: none; }

/* Water */
.reservoir-wall { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.water          { fill: #85B7EB; stroke: #378ADD; stroke-width: 0.5; }
.pipe           { fill: none; stroke: #378ADD; stroke-width: 4; stroke-linecap: round; }
.pipe-joint     { fill: #185FA5; stroke: #0C447C; stroke-width: 1; }
.valve          { fill: #0C447C; stroke: #185FA5; stroke-width: 1; }

/* Transport */
.road          { stroke: #888780; stroke-width: 8; fill: none; stroke-linecap: round; }
.road-mark     { stroke: #F1EFE8; stroke-width: 1; stroke-dasharray: 6 4; fill: none; }
.traffic-light { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }
.light-red     { fill: #E24B4A; }
.light-green   { fill: #97C459; }
.light-off     { fill: #2C2C2A; }
.bus           { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1.5; }
```

## Reference examples

- `examples/smart-city-infrastructure.md` — hub-spoke with multiple subsystems
- `examples/electricity-grid-flow.md` — voltage hierarchy, flow markers
- `examples/wind-turbine-structure.md` — cross-section with legend
