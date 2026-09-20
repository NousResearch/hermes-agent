# Feature Film Production Pipeline

A phased workflow showing the five stages of filmmaking, using containers with inner nodes and horizontal sub-flows within a phase.

## Key Patterns Used

- **Phase containers**: Large rounded rectangles with neutral background and dashed borders
- **Inner task nodes**: Smaller colored nodes inside containers for sub-tasks
- **Horizontal flow within container**: Post-production shows sequential pipeline with arrows (Editing → Color → VFX → Sound → Score)
- **Consistent phase spacing**: ~30px gap between phase containers
- **Phase labels with subtitles**: Each container has title + description

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 780" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Phase 1: Development -->
  <g>
    <rect x="40" y="30" width="600" height="110" rx="16" stroke-width="1" stroke-dasharray="6 4" fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="56">Development</text>
    <text class="ts" x="66" y="74">Concept to greenlight</text>
  </g>
  <g class="node c-purple">
    <rect x="70" y="90" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="150" y="108" text-anchor="middle" dominant-baseline="central">Script / screenplay</text>
  </g>
  <g class="node c-purple">
    <rect x="260" y="90" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="108" text-anchor="middle" dominant-baseline="central">Financing / budget</text>
  </g>
  <g class="node c-purple">
    <rect x="450" y="90" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="530" y="108" text-anchor="middle" dominant-baseline="central">Casting leads</text>
  </g>

  <!-- Arrow to Phase 2 -->
  <line x1="340" y1="140" x2="340" y2="170" class="arr" marker-end="url(#arrow)"/>

  <!-- Phase 2: Pre-production -->
  <g>
    <rect x="40" y="170" width="600" height="110" rx="16" stroke-width="1" stroke-dasharray="6 4" fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="196">Pre-production</text>
    <text class="ts" x="66" y="214">Planning and preparation</text>
  </g>
  <g class="node c-teal">
    <rect x="70" y="230" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="150" y="248" text-anchor="middle" dominant-baseline="central">Storyboards</text>
  </g>
  <g class="node c-teal">
    <rect x="260" y="230" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="248" text-anchor="middle" dominant-baseline="central">Location scouting</text>
  </g>
  <g class="node c-teal">
    <rect x="450" y="230" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="530" y="248" text-anchor="middle" dominant-baseline="central">Crew hiring</text>
  </g>

  <!-- Arrow to Phase 3 -->
  <line x1="340" y1="280" x2="340" y2="310" class="arr" marker-end="url(#arrow)"/>

  <!-- Phase 3: Production -->
  <g>
    <rect x="40" y="310" width="600" height="110" rx="16" stroke-width="1" stroke-dasharray="6 4" fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="336">Production</text>
    <text class="ts" x="66" y="354">Principal photography</text>
  </g>
  <g class="node c-coral">
    <rect x="70" y="370" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="150" y="388" text-anchor="middle" dominant-baseline="central">Filming / shooting</text>
  </g>
  <g class="node c-coral">
    <rect x="260" y="370" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="388" text-anchor="middle" dominant-baseline="central">Production sound</text>
  </g>
  <g class="node c-coral">
    <rect x="450" y="370" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="530" y="388" text-anchor="middle" dominant-baseline="central">VFX plates</text>
  </g>

  <!-- Arrow to Phase 4 -->
  <line x1="340" y1="420" x2="340" y2="450" class="arr" marker-end="url(#arrow)"/>

  <!-- Phase 4: Post-production -->
  <g>
    <rect x="40" y="450" width="600" height="150" rx="16" stroke-width="1" stroke-dasharray="6 4" fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="476">Post-production</text>
    <text class="ts" x="66" y="494">Assembly and finishing</text>
  </g>
  <g class="node c-amber">
    <rect x="70" y="510" width="110" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="125" y="528" text-anchor="middle" dominant-baseline="central">Editing</text>
  </g>
  <g class="node c-amber">
    <rect x="195" y="510" width="110" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="250" y="528" text-anchor="middle" dominant-baseline="central">Color grade</text>
  </g>
  <g class="node c-amber">
    <rect x="320" y="510" width="90" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="365" y="528" text-anchor="middle" dominant-baseline="central">VFX</text>
  </g>
  <g class="node c-amber">
    <rect x="425" y="510" width="100" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="475" y="528" text-anchor="middle" dominant-baseline="central">Sound mix</text>
  </g>
  <g class="node c-amber">
    <rect x="540" y="510" width="80" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="580" y="528" text-anchor="middle" dominant-baseline="central">Score</text>
  </g>
  <!-- Flow arrows within post -->
  <line x1="180" y1="528" x2="195" y2="528" class="arr" marker-end="url(#arrow)"/>
  <line x1="305" y1="528" x2="320" y2="528" class="arr" marker-end="url(#arrow)"/>
  <line x1="410" y1="528" x2="425" y2="528" class="arr" marker-end="url(#arrow)"/>
  <line x1="525" y1="528" x2="540" y2="528" class="arr" marker-end="url(#arrow)"/>
  <!-- Final delivery label -->
  <g class="node c-amber">
    <rect x="240" y="556" width="200" height="32" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="572" text-anchor="middle" dominant-baseline="central">Final master / DCP</text>
  </g>
  <line x1="340" y1="546" x2="340" y2="556" class="arr" marker-end="url(#arrow)"/>

  <!-- Arrow to Phase 5 -->
  <line x1="340" y1="600" x2="340" y2="630" class="arr" marker-end="url(#arrow)"/>

  <!-- Phase 5: Distribution -->
  <g>
    <rect x="40" y="630" width="600" height="110" rx="16" stroke-width="1" stroke-dasharray="6 4" fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="656">Distribution</text>
    <text class="ts" x="66" y="674">Release and exhibition</text>
  </g>
  <g class="node c-blue">
    <rect x="70" y="690" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="150" y="708" text-anchor="middle" dominant-baseline="central">Film festivals</text>
  </g>
  <g class="node c-blue">
    <rect x="260" y="690" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="340" y="708" text-anchor="middle" dominant-baseline="central">Theatrical release</text>
  </g>
  <g class="node c-blue">
    <rect x="450" y="690" width="160" height="36" rx="6" stroke-width="0.5"/>
    <text class="ts" x="530" y="708" text-anchor="middle" dominant-baseline="central">Streaming / VOD</text>
  </g>
</svg>
```

## Color Assignments

| Element | Color | Reason |
|---------|-------|--------|
| Phase containers | Neutral (dashed) | Subtle grouping, doesn't compete with content |
| Development tasks | `c-purple` | Creative/concept work |
| Pre-production tasks | `c-teal` | Planning and preparation |
| Production tasks | `c-coral` | Active filming (main event) |
| Post-production tasks | `c-amber` | Processing/refinement |
| Distribution tasks | `c-blue` | Outward delivery/release |

## Layout Notes

- **ViewBox**: 680×780 (standard width, tall for 5 phases)
- **Container style**: Dashed border (`stroke-dasharray="6 4"`), neutral fill (`var(--bg-secondary)`), `stroke-width="1"`
- **Container height**: 110px for 3-node phases, 150px for post-production (more complex)
- **Inner node dimensions**: 160×36px for standard tasks, variable width for post-production sequential flow
- **Phase gap**: 30px between containers
- **Horizontal sub-flow**: Post-production uses tightly packed nodes with arrows between them to show sequence
- **Convergence node**: "Final master / DCP" sits below the horizontal flow, collecting all post outputs
