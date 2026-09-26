# Out-of-Order CPU Core Microarchitecture

A structural diagram showing the internal pipeline stages of a modern superscalar out-of-order CPU core. Demonstrates multi-stage vertical flow with parallel paths, fan-out patterns for execution ports, and a separate memory hierarchy sidebar.

## Key Patterns Used

- **Multi-stage vertical flow**: Six pipeline stages (Front End → Rename → Schedule → Execute → Retire)
- **Parallel decode paths**: Main decode and µop cache bypass (dashed line for cache hit)
- **Container grouping**: Logical stages grouped in colored containers
- **Fan-out pattern**: Single scheduler dispatching to 6 execution ports
- **Sidebar layout**: Memory hierarchy placed in separate column on right
- **Stage labels**: Left-aligned labels indicating pipeline phase
- **Color-coded semantics**: Different colors for each functional unit category

## Diagram Type

This is a **hybrid structural/flow** diagram:
- **Flow aspect**: Instructions move top-to-bottom through pipeline stages
- **Structural aspect**: Components are grouped by function (rename unit, execution cluster)
- **Sidebar**: Memory hierarchy is architecturally separate but connected via data paths

## Pipeline Stage Breakdown

### Front End (Purple)
```xml
<!-- Fetch Unit -->
<g class="node c-purple">
  <rect x="40" y="70" width="140" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="110" y="90" text-anchor="middle" dominant-baseline="central">Fetch unit</text>
  <text class="ts" x="110" y="110" text-anchor="middle" dominant-baseline="central">6-wide, 32B/cycle</text>
</g>

<!-- Branch Predictor (subordinate) -->
<g class="node c-purple">
  <rect x="40" y="140" width="140" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="110" y="162" text-anchor="middle" dominant-baseline="central">Branch predictor</text>
</g>

<!-- Decode -->
<g class="node c-purple">
  <rect x="230" y="70" width="160" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="310" y="90" text-anchor="middle" dominant-baseline="central">Decode</text>
  <text class="ts" x="310" y="110" text-anchor="middle" dominant-baseline="central">x86 → µops, 6-wide</text>
</g>
```

### µop Cache Bypass Path (Teal)
The µop cache (Decoded Stream Buffer) provides an alternate path that bypasses the complex decoder:

```xml
<!-- µop Cache parallel to decode -->
<g class="node c-teal">
  <rect x="230" y="150" width="160" height="50" rx="8" stroke-width="0.5"/>
  <text class="th" x="310" y="168" text-anchor="middle" dominant-baseline="central">µop cache (DSB)</text>
  <text class="ts" x="310" y="186" text-anchor="middle" dominant-baseline="central">4K entries, 8-wide</text>
</g>

<!-- Dashed bypass path indicating cache hit -->
<path d="M180 110 L205 110 L205 175 L230 175" fill="none" class="arr" 
      stroke-dasharray="4 3" marker-end="url(#arrow)"/>
<text class="tx" x="164" y="148" opacity=".6">hit</text>
```

### Rename/Allocate Container (Coral)
Groups related rename components in a container:

```xml
<!-- Outer container -->
<g class="c-coral">
  <rect x="40" y="250" width="530" height="130" rx="12" stroke-width="0.5"/>
  <text class="th" x="60" y="274">Rename / allocate</text>
  <text class="ts" x="60" y="292">Map architectural → physical registers</text>
</g>

<!-- Inner components -->
<g class="node c-coral">
  <rect x="60" y="310" width="180" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="150" y="330" text-anchor="middle" dominant-baseline="central">Register alias table</text>
  <text class="ts" x="150" y="350" text-anchor="middle" dominant-baseline="central">180 physical regs</text>
</g>
```

### Scheduler Fan-Out Pattern (Amber → Teal)
Single unified scheduler dispatching to multiple execution ports:

```xml
<!-- Unified Scheduler -->
<g class="node c-amber">
  <rect x="140" y="420" width="330" height="50" rx="8" stroke-width="0.5"/>
  <text class="th" x="305" y="438" text-anchor="middle" dominant-baseline="central">Unified scheduler</text>
  <text class="ts" x="305" y="456" text-anchor="middle" dominant-baseline="central">97 entries, out-of-order dispatch</text>
</g>

<!-- Fan-out arrows to 6 ports -->
<line x1="170" y1="470" x2="90" y2="540" class="arr" marker-end="url(#arrow)"/>
<line x1="215" y1="470" x2="170" y2="540" class="arr" marker-end="url(#arrow)"/>
<line x1="265" y1="470" x2="250" y2="540" class="arr" marker-end="url(#arrow)"/>
<line x1="305" y1="470" x2="330" y2="540" class="arr" marker-end="url(#arrow)"/>
<line x1="355" y1="470" x2="410" y2="540" class="arr" marker-end="url(#arrow)"/>
<line x1="420" y1="470" x2="490" y2="540" class="arr" marker-end="url(#arrow)"/>
```

### Execution Port Box Pattern
Compact boxes showing port number and capabilities:

```xml
<!-- Execution port with multi-line capability -->
<g class="node c-teal">
  <rect x="55" y="540" width="70" height="64" rx="6" stroke-width="0.5"/>
  <text class="th" x="90" y="560" text-anchor="middle" dominant-baseline="central">Port 0</text>
  <text class="tx" x="90" y="576" text-anchor="middle" dominant-baseline="central">ALU</text>
  <text class="tx" x="90" y="590" text-anchor="middle" dominant-baseline="central">DIV</text>
</g>
```

### Reorder Buffer (Pink)
Wide horizontal bar at bottom showing retirement:

```xml
<g class="c-pink">
  <rect x="40" y="670" width="530" height="40" rx="10" stroke-width="0.5"/>
  <text class="th" x="305" y="694" text-anchor="middle" dominant-baseline="central">Reorder buffer (ROB) — 512 entries, 8-wide retire</text>
</g>
```

### Memory Hierarchy Sidebar (Blue)
Separate column showing cache levels:

```xml
<!-- Container -->
<g class="c-blue">
  <rect x="600" y="30" width="190" height="360" rx="16" stroke-width="0.5"/>
  <text class="th" x="695" y="54" text-anchor="middle">Memory hierarchy</text>
</g>

<!-- Cache levels stacked vertically -->
<g class="node c-blue">
  <rect x="620" y="70" width="150" height="50" rx="8" stroke-width="0.5"/>
  <text class="th" x="695" y="88" text-anchor="middle" dominant-baseline="central">L1-I cache</text>
  <text class="ts" x="695" y="106" text-anchor="middle" dominant-baseline="central">32 KB, 8-way</text>
</g>
<!-- Additional levels follow same pattern -->
```

## Connection Patterns

### Instruction Fetch Path
Horizontal arrow from L1-I cache to fetch unit:
```xml
<path d="M620 95 L200 95" fill="none" class="arr" marker-end="url(#arrow)"/>
<text class="tx" x="410" y="88" text-anchor="middle" opacity=".6">instruction fetch</text>
```

### Load/Store Path
Complex path from execution ports to L1-D cache:
```xml
<path d="M250 604 L250 640 L580 640 L580 160 L620 160" fill="none" class="arr" marker-end="url(#arrow)"/>
<text class="tx" x="415" y="652" text-anchor="middle" opacity=".6">load / store</text>
```

### Commit Path (dashed)
Dashed line showing write-back from ROB to register file:
```xml
<path d="M550 690 L580 690 L580 445 L595 445" fill="none" class="arr" stroke-dasharray="4 3"/>
<text class="tx" x="590" y="578" opacity=".6" transform="rotate(-90 590 578)">commit</text>
```

### Path Merge (Decode + µop Cache)
Two paths converging before rename:
```xml
<line x1="390" y1="98" x2="430" y2="98" class="arr"/>
<line x1="390" y1="175" x2="430" y2="175" class="arr"/>
<path d="M430 98 L430 175" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
<line x1="430" y1="136" x2="470" y2="136" class="arr" marker-end="url(#arrow)"/>
```

## Text Classes

This diagram uses an additional text class for very small labels:

```css
.tx { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; fill: var(--text-secondary); }
```

Used for:
- Execution port capability labels (ALU, Branch, Load, etc.)
- Connection labels (instruction fetch, load/store, commit)
- DRAM latency annotation

## Color Semantic Mapping

| Color | Stage | Components |
|-------|-------|------------|
| `c-purple` | Front end | Fetch, Branch predictor, Decode |
| `c-teal` | Execution | µop cache, Execution ports |
| `c-coral` | Rename | RAT, Physical RF, Free list |
| `c-amber` | Schedule | Unified scheduler |
| `c-pink` | Retire | Reorder buffer |
| `c-blue` | Memory | L1-I, L1-D, L2, DRAM |
| `c-gray` | External | Off-chip DRAM |

## Layout Notes

- **ViewBox**: 820×720 (taller than wide for vertical pipeline flow)
- **Main pipeline**: x=40 to x=570 (530px width)
- **Memory sidebar**: x=600 to x=790 (190px width)
- **Stage labels**: x=30, left-aligned, 50% opacity
- **Vertical spacing**: ~80-100px between major stages
- **Container padding**: 20px inside containers
- **Port spacing**: 80px between execution port centers
- **Legend**: Bottom-right of memory sidebar, explains color coding

## Architectural Details Shown

| Component | Specification | Notes |
|-----------|---------------|-------|
| Fetch | 6-wide, 32B/cycle | Typical modern Intel/AMD |
| Decode | 6-wide, x86→µops | Complex decoder |
| µop Cache | 4K entries, 8-wide | Bypass for hot code |
| RAT | 180 physical regs | Supports deep OoO |
| Scheduler | 97 entries | Unified RS |
| Execution | 6 ports | ALU×2, Load, Store×2, Vector |
| ROB | 512 entries, 8-wide | In-order retirement |
| L1-I | 32 KB, 8-way | Instruction cache |
| L1-D | 48 KB, 12-way | Data cache |
| L2 | 1.25 MB, 20-way | Unified |
| DRAM | DDR5-6400, ~80ns | Off-chip |

## When to Use This Pattern

Use this diagram style for:
- CPU/GPU microarchitecture visualization
- Compiler pipeline stages
- Network packet processing pipelines
- Any system with parallel execution units fed by a scheduler
- Hardware designs with multiple functional units
