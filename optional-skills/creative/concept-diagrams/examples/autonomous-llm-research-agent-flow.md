# Autonomous LLM Research Agent Flow

A multi-section flowchart showing Karpathy's autoresearch framework: human-agent handoff, the autonomous experiment loop with keep/discard decision branching, and the modifiable training pipeline. Demonstrates loop-back arrows, convergent decision paths, and semantic color coding for outcomes.

## Key Patterns Used

- **Three-section layout**: Setup row, main loop container, and detail container — each visually distinct
- **Neutral dashed containers**: Loop and training pipeline use `var(--bg-secondary)` fill with dashed borders to recede behind colored content nodes
- **Decision branching with convergence**: "val_bpb improved?" splits into Keep (green) and Discard (red), then both converge back to "Log to results.tsv"
- **Loop-back arrow**: Dashed path with rounded corners on the right side of the container showing infinite repetition
- **Semantic color for outcomes**: Green = improvement (keep), Red = no improvement (discard) — not arbitrary decoration
- **Highlighted key step**: "Run training" uses `c-coral` to visually distinguish the most important step from other `c-teal` actions
- **Horizontal pipeline flow**: Training details section uses left-to-right arrow-connected nodes (GPT → MuonAdamW → Evaluation)
- **Footer metadata**: Fixed constraints shown as subtle centered text below the pipeline nodes
- **Legend row**: Color key at the bottom explaining what each color means

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 920" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- ========================================== -->
  <!-- SECTION 1: SETUP (Human → program.md → AI) -->
  <!-- ========================================== -->

  <text class="ts" x="40" y="30" text-anchor="start" opacity=".5">One-time setup</text>

  <!-- Human -->
  <g class="node c-gray">
    <rect x="60" y="42" width="140" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="130" y="62" text-anchor="middle" dominant-baseline="central">Human</text>
    <text class="ts" x="130" y="82" text-anchor="middle" dominant-baseline="central">Researcher</text>
  </g>

  <!-- Arrow: Human → program.md -->
  <line x1="200" y1="70" x2="250" y2="70" class="arr" marker-end="url(#arrow)"/>

  <!-- program.md -->
  <g class="node c-gray">
    <rect x="250" y="42" width="180" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="62" text-anchor="middle" dominant-baseline="central">program.md</text>
    <text class="ts" x="340" y="82" text-anchor="middle" dominant-baseline="central">Agent instructions</text>
  </g>

  <!-- Arrow: program.md → AI Agent -->
  <line x1="430" y1="70" x2="470" y2="70" class="arr" marker-end="url(#arrow)"/>

  <!-- AI Agent -->
  <g class="node c-purple">
    <rect x="470" y="42" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="550" y="62" text-anchor="middle" dominant-baseline="central">AI agent</text>
    <text class="ts" x="550" y="82" text-anchor="middle" dominant-baseline="central">Claude / Codex</text>
  </g>

  <!-- Arrow: Setup row → Loop (from program.md center down) -->
  <line x1="340" y1="98" x2="340" y2="142" class="arr" marker-end="url(#arrow)"/>

  <!-- ========================================== -->
  <!-- SECTION 2: AUTONOMOUS EXPERIMENT LOOP      -->
  <!-- ========================================== -->

  <!-- Loop container (neutral dashed) -->
  <g>
    <rect x="40" y="142" width="600" height="528" rx="16"
          stroke-width="1" stroke-dasharray="6 4"
          fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="170">Autonomous experiment loop</text>
    <text class="ts" x="66" y="188">~12 experiments/hour — runs until manually stopped</text>
  </g>

  <!-- Step 1: Read code + past results -->
  <g class="node c-teal">
    <rect x="170" y="208" width="280" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="310" y="230" text-anchor="middle" dominant-baseline="central">Read code + past results</text>
  </g>

  <!-- Arrow: S1 → S2 -->
  <line x1="310" y1="252" x2="310" y2="274" class="arr" marker-end="url(#arrow)"/>

  <!-- Step 2: Propose + edit train.py -->
  <g class="node c-teal">
    <rect x="170" y="274" width="280" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="310" y="294" text-anchor="middle" dominant-baseline="central">Propose + edit train.py</text>
    <text class="ts" x="310" y="314" text-anchor="middle" dominant-baseline="central">Arch, optimizer, hyperparameters</text>
  </g>

  <!-- Arrow: S2 → S3 -->
  <line x1="310" y1="330" x2="310" y2="352" class="arr" marker-end="url(#arrow)"/>

  <!-- Step 3: Run training (highlighted — key step) -->
  <g class="node c-coral">
    <rect x="170" y="352" width="280" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="310" y="372" text-anchor="middle" dominant-baseline="central">Run training</text>
    <text class="ts" x="310" y="392" text-anchor="middle" dominant-baseline="central">uv run train.py (5 min budget)</text>
  </g>

  <!-- Arrow: S3 → S4 -->
  <line x1="310" y1="408" x2="310" y2="430" class="arr" marker-end="url(#arrow)"/>

  <!-- Step 4: Decision — val_bpb improved? -->
  <g class="node c-gray">
    <rect x="170" y="430" width="280" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="310" y="452" text-anchor="middle" dominant-baseline="central">val_bpb improved?</text>
  </g>

  <!-- Decision arrows to Keep / Discard -->
  <line x1="240" y1="474" x2="175" y2="508" class="arr" marker-end="url(#arrow)"/>
  <line x1="380" y1="474" x2="445" y2="508" class="arr" marker-end="url(#arrow)"/>

  <!-- Decision labels -->
  <text class="ts" x="195" y="496" opacity=".6">yes</text>
  <text class="ts" x="416" y="496" opacity=".6">no</text>

  <!-- Keep — advance branch -->
  <g class="node c-green">
    <rect x="70" y="508" width="210" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="175" y="528" text-anchor="middle" dominant-baseline="central">Keep</text>
    <text class="ts" x="175" y="548" text-anchor="middle" dominant-baseline="central">Advance git branch</text>
  </g>

  <!-- Discard — git reset -->
  <g class="node c-red">
    <rect x="340" y="508" width="210" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="445" y="528" text-anchor="middle" dominant-baseline="central">Discard</text>
    <text class="ts" x="445" y="548" text-anchor="middle" dominant-baseline="central">Git reset to previous</text>
  </g>

  <!-- Converge arrows: Keep → Log, Discard → Log -->
  <line x1="175" y1="564" x2="250" y2="590" class="arr" marker-end="url(#arrow)"/>
  <line x1="445" y1="564" x2="370" y2="590" class="arr" marker-end="url(#arrow)"/>

  <!-- Step 6: Log to results.tsv -->
  <g class="node c-teal">
    <rect x="170" y="590" width="280" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="310" y="612" text-anchor="middle" dominant-baseline="central">Log to results.tsv</text>
  </g>

  <!-- Loop-back arrow (dashed, right side) -->
  <path d="M 450 612 L 564 612 Q 576 612 576 600 L 576 242 Q 576 230 564 230 L 450 230"
        fill="none" class="arr" stroke-dasharray="4 3" marker-end="url(#arrow)"/>

  <!-- ========================================== -->
  <!-- SECTION 3: TRAINING PIPELINE DETAILS       -->
  <!-- ========================================== -->

  <!-- Connection arrow: Loop → Training details -->
  <line x1="310" y1="670" x2="310" y2="710" class="arr" marker-end="url(#arrow)"/>

  <!-- Training container (neutral dashed) -->
  <g>
    <rect x="40" y="710" width="600" height="170" rx="16"
          stroke-width="1" stroke-dasharray="6 4"
          fill="var(--bg-secondary)" stroke="var(--border)"/>
    <text class="th" x="66" y="738">train.py — modifiable training pipeline</text>
    <text class="ts" x="66" y="756">Runs during each training step — single GPU, single file</text>
  </g>

  <!-- GPT model -->
  <g class="node c-coral">
    <rect x="70" y="774" width="155" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="147" y="794" text-anchor="middle" dominant-baseline="central">GPT model</text>
    <text class="ts" x="147" y="814" text-anchor="middle" dominant-baseline="central">RoPE, FlashAttn3</text>
  </g>

  <!-- Arrow: GPT → MuonAdamW -->
  <line x1="225" y1="802" x2="260" y2="802" class="arr" marker-end="url(#arrow)"/>

  <!-- MuonAdamW optimizer -->
  <g class="node c-coral">
    <rect x="260" y="774" width="155" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="337" y="794" text-anchor="middle" dominant-baseline="central">MuonAdamW</text>
    <text class="ts" x="337" y="814" text-anchor="middle" dominant-baseline="central">Hybrid optimizer</text>
  </g>

  <!-- Arrow: MuonAdamW → Evaluation -->
  <line x1="415" y1="802" x2="450" y2="802" class="arr" marker-end="url(#arrow)"/>

  <!-- Evaluation -->
  <g class="node c-amber">
    <rect x="450" y="774" width="155" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="527" y="794" text-anchor="middle" dominant-baseline="central">Evaluation</text>
    <text class="ts" x="527" y="814" text-anchor="middle" dominant-baseline="central">val_bpb metric</text>
  </g>

  <!-- Footer: fixed constraints -->
  <text class="ts" x="340" y="856" text-anchor="middle" opacity=".5">climbmix-400b data · 8K BPE vocab · 300s budget · 2048 context</text>

  <!-- ========================================== -->
  <!-- LEGEND                                     -->
  <!-- ========================================== -->

  <g class="c-teal"><rect x="40" y="890" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="62" y="902">Agent actions</text>

  <g class="c-coral"><rect x="170" y="890" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="192" y="902">Training run</text>

  <g class="c-green"><rect x="300" y="890" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="322" y="902">Improvement</text>

  <g class="c-red"><rect x="430" y="890" width="14" height="14" rx="3" stroke-width="0.5"/></g>
  <text class="ts" x="452" y="902">No improvement</text>

</svg>
```

## Color Assignments

| Element | Color | Reason |
|---------|-------|--------|
| Human, program.md | `c-gray` | Neutral setup / input nodes |
| AI agent | `c-purple` | The active intelligent actor |
| Loop action steps | `c-teal` | Agent's analytical/editing actions |
| Run training | `c-coral` | Highlighted key step — the 5-min training run |
| Decision check | `c-gray` | Neutral evaluation checkpoint |
| Keep (improved) | `c-green` | Semantic success — val_bpb decreased |
| Discard (not improved) | `c-red` | Semantic failure — no improvement |
| Training pipeline nodes | `c-coral` | Training infrastructure components |
| Evaluation node | `c-amber` | Distinct from training — measurement/metric role |
| Containers | Neutral (dashed) | Subtle grouping that recedes behind content |

## Layout Notes

- **ViewBox**: 680×920 (standard width, tall for 3 sections)
- **Three sections**: Setup row (y=30–98), loop container (y=142–670), training details (y=710–880)
- **Container style**: Dashed border (`stroke-dasharray="6 4"`), neutral fill (`var(--bg-secondary)`), `stroke-width="1"` — not colored, so inner nodes pop
- **Loop-back arrow**: Dashed `<path>` with quadratic curves (`Q`) at corners for smooth rounded turns, running up the right side of the loop container from "Log" back to "Read code"
- **Decision pattern**: Single question node ("val_bpb improved?") with diagonal arrows to Keep/Discard, then convergent diagonal arrows back to "Log to results.tsv"
- **Decision labels**: "yes"/"no" labels placed along the diagonal arrows with `opacity=".6"` to stay subtle
- **Key step highlight**: "Run training" uses `c-coral` while surrounding steps use `c-teal`, drawing the eye to the most important step
- **Horizontal sub-flow**: Training pipeline uses left-to-right arrow-connected nodes (GPT model → MuonAdamW → Evaluation)
- **Footer metadata**: Fixed constraints (data, vocab, budget, context) shown as a single centered `ts` text line with `opacity=".5"`
- **Legend**: Four color swatches at the bottom explaining the semantic meaning of each color used
