# Hospital Emergency Department Flow

A multi-path flowchart showing patient journey through an emergency department with priority-based routing using semantic colors (red=critical, amber=urgent, green=stable).

## Key Patterns Used

- **Semantic color coding**: Red/amber/green for priority levels (not arbitrary decoration)
- **Stage labels**: Left-aligned faded labels marking workflow phases
- **Convergent paths**: Multiple entry points merging, then branching, then converging again
- **Nested containers**: Diagnostics grouped in a container with inner nodes
- **Legend**: Color key at bottom explaining priority levels

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 620" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Stage labels -->
  <text class="ts" x="40" y="68" text-anchor="start" opacity=".5">Arrival</text>
  <text class="ts" x="40" y="168" text-anchor="start" opacity=".5">Assessment</text>
  <text class="ts" x="40" y="288" text-anchor="start" opacity=".5">Priority routing</text>
  <text class="ts" x="40" y="418" text-anchor="start" opacity=".5">Diagnostics</text>
  <text class="ts" x="40" y="518" text-anchor="start" opacity=".5">Outcome</text>

  <!-- Arrival: Ambulance -->
  <g class="node c-gray">
    <rect x="140" y="40" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="220" y="60" text-anchor="middle" dominant-baseline="central">Ambulance</text>
    <text class="ts" x="220" y="80" text-anchor="middle" dominant-baseline="central">Emergency transport</text>
  </g>

  <!-- Arrival: Walk-in -->
  <g class="node c-gray">
    <rect x="380" y="40" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="460" y="60" text-anchor="middle" dominant-baseline="central">Walk-in</text>
    <text class="ts" x="460" y="80" text-anchor="middle" dominant-baseline="central">Self-arrival</text>
  </g>

  <!-- Arrows to Triage -->
  <line x1="220" y1="96" x2="300" y2="140" class="arr" marker-end="url(#arrow)"/>
  <line x1="460" y1="96" x2="380" y2="140" class="arr" marker-end="url(#arrow)"/>

  <!-- Triage -->
  <g class="node c-purple">
    <rect x="240" y="140" width="200" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="160" text-anchor="middle" dominant-baseline="central">Triage</text>
    <text class="ts" x="340" y="180" text-anchor="middle" dominant-baseline="central">Nurse assessment, vitals</text>
  </g>

  <!-- Arrows from Triage to Priority -->
  <line x1="280" y1="196" x2="140" y2="260" class="arr" marker-end="url(#arrow)"/>
  <line x1="340" y1="196" x2="340" y2="260" class="arr" marker-end="url(#arrow)"/>
  <line x1="400" y1="196" x2="540" y2="260" class="arr" marker-end="url(#arrow)"/>

  <!-- Priority: Red - Trauma -->
  <g class="node c-red">
    <rect x="60" y="260" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="140" y="280" text-anchor="middle" dominant-baseline="central">Trauma bay</text>
    <text class="ts" x="140" y="300" text-anchor="middle" dominant-baseline="central">Priority: critical</text>
  </g>

  <!-- Priority: Yellow - Exam rooms -->
  <g class="node c-amber">
    <rect x="260" y="260" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="280" text-anchor="middle" dominant-baseline="central">Exam rooms</text>
    <text class="ts" x="340" y="300" text-anchor="middle" dominant-baseline="central">Priority: urgent</text>
  </g>

  <!-- Priority: Green - Waiting -->
  <g class="node c-green">
    <rect x="460" y="260" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="540" y="280" text-anchor="middle" dominant-baseline="central">Waiting area</text>
    <text class="ts" x="540" y="300" text-anchor="middle" dominant-baseline="central">Priority: stable</text>
  </g>

  <!-- Arrows to Diagnostics -->
  <line x1="140" y1="316" x2="220" y2="390" class="arr" marker-end="url(#arrow)"/>
  <line x1="340" y1="316" x2="340" y2="390" class="arr" marker-end="url(#arrow)"/>
  <line x1="540" y1="316" x2="460" y2="390" class="arr" marker-end="url(#arrow)"/>

  <!-- Diagnostics container -->
  <g class="c-teal">
    <rect x="140" y="390" width="400" height="56" rx="12" stroke-width="0.5"/>
  </g>

  <!-- Labs -->
  <g class="node c-teal">
    <rect x="160" y="400" width="110" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="215" y="418" text-anchor="middle" dominant-baseline="central">Labs</text>
  </g>

  <!-- Imaging -->
  <g class="node c-teal">
    <rect x="285" y="400" width="110" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="418" text-anchor="middle" dominant-baseline="central">Imaging</text>
  </g>

  <!-- Diagnosis -->
  <g class="node c-teal">
    <rect x="410" y="400" width="110" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="465" y="418" text-anchor="middle" dominant-baseline="central">Diagnosis</text>
  </g>

  <!-- Arrows to Outcomes -->
  <line x1="215" y1="446" x2="160" y2="490" class="arr" marker-end="url(#arrow)"/>
  <line x1="340" y1="446" x2="340" y2="490" class="arr" marker-end="url(#arrow)"/>
  <line x1="465" y1="446" x2="520" y2="490" class="arr" marker-end="url(#arrow)"/>

  <!-- Outcome: Admission -->
  <g class="node c-coral">
    <rect x="80" y="490" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="160" y="510" text-anchor="middle" dominant-baseline="central">Admission</text>
    <text class="ts" x="160" y="530" text-anchor="middle" dominant-baseline="central">Inpatient ward</text>
  </g>

  <!-- Outcome: Surgery -->
  <g class="node c-coral">
    <rect x="260" y="490" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="510" text-anchor="middle" dominant-baseline="central">Surgery</text>
    <text class="ts" x="340" y="530" text-anchor="middle" dominant-baseline="central">Operating room</text>
  </g>

  <!-- Outcome: Discharge -->
  <g class="node c-coral">
    <rect x="440" y="490" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="520" y="510" text-anchor="middle" dominant-baseline="central">Discharge</text>
    <text class="ts" x="520" y="530" text-anchor="middle" dominant-baseline="central">Home with instructions</text>
  </g>

  <!-- Legend -->
  <text class="ts" x="140" y="580" opacity=".5">Priority levels</text>
  <g class="c-red"><rect x="140" y="592" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="162" y="604">Critical</text>
  <g class="c-amber"><rect x="240" y="592" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="262" y="604">Urgent</text>
  <g class="c-green"><rect x="340" y="592" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="362" y="604">Stable</text>
</svg>
```

## Color Assignments

| Element | Color | Reason |
|---------|-------|--------|
| Entry points (Ambulance, Walk-in) | `c-gray` | Neutral starting points |
| Triage | `c-purple` | Processing/assessment step |
| Trauma bay | `c-red` | Critical priority (semantic) |
| Exam rooms | `c-amber` | Urgent priority (semantic) |
| Waiting area | `c-green` | Stable priority (semantic) |
| Diagnostics | `c-teal` | Clinical services category |
| Outcomes | `c-coral` | Final disposition category |

## Layout Notes

- **ViewBox**: 680×620 (standard width, extended height for 5 stages)
- **Stage spacing**: ~110-130px between stage rows
- **Diagonal arrows**: Connect nodes across columns naturally
- **Container with inner nodes**: Diagnostics uses outer `c-teal` rect with inner node rects
