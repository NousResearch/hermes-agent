# SN2 Reaction Mechanism

A chemistry diagram showing the bimolecular nucleophilic substitution (SN2) mechanism between hydroxide ion and methyl bromide. Demonstrates molecular structure rendering, electron movement arrows, transition state notation, and reaction energy profiles.

## Key Patterns Used

- **Molecular structures**: Ball-and-stick style atoms with bonds
- **Electron movement**: Curved arrows showing nucleophilic attack
- **Transition state**: Bracketed pentacoordinate intermediate with partial charges
- **Stereochemistry**: Wedge/dash bonds showing 3D configuration
- **Energy profile**: Potential energy vs reaction coordinate plot
- **Annotation boxes**: Key features and mechanistic notes

## Diagram Type

This is a **chemistry mechanism diagram** with:
- **Molecular rendering**: Atoms as colored circles with element symbols
- **Bond notation**: Solid, wedge, dash, and partial (dashed) bonds
- **Reaction arrows**: Curved for electron movement, straight for reaction progress
- **Energy landscape**: Quantitative energy profile below mechanism

## Molecular Structure Elements

### Atom Rendering

```xml
<!-- Carbon atom (dark) -->
<circle cx="0" cy="0" r="14" class="carbon"/>
<text class="chem" x="0" y="5" text-anchor="middle" fill="white" font-weight="500">C</text>

<!-- Oxygen atom (red) -->
<circle cx="0" cy="0" r="14" class="oxygen"/>
<text class="chem" x="0" y="5" text-anchor="middle" fill="white" font-weight="500">O</text>

<!-- Hydrogen atom (light with border) -->
<circle cx="38" cy="0" r="8" class="hydrogen"/>
<text class="chem-sm" x="38" y="4" text-anchor="middle">H</text>

<!-- Bromine atom (brown) -->
<circle cx="52" cy="0" r="16" class="bromine"/>
<text class="chem" x="52" y="5" text-anchor="middle" fill="white" font-weight="500">Br</text>
```

```css
.carbon { fill: #2C2C2A; }
.hydrogen { fill: #F1EFE8; stroke: #888780; stroke-width: 1; }
.oxygen { fill: #E24B4A; }
.bromine { fill: #993C1D; }
.nitrogen { fill: #378ADD; }  /* for other reactions */
```

### Bond Types

```xml
<!-- Single bond (solid) -->
<line x1="14" y1="0" x2="38" y2="0" class="bond"/>

<!-- Wedge bond (coming toward viewer) -->
<polygon class="bond-wedge" points="0,-14 -6,-35 6,-35"/>

<!-- Dash bond (going away from viewer) -->
<line x1="-10" y1="10" x2="-28" y2="28" class="bond-dash"/>

<!-- Partial bond (forming/breaking) -->
<line x1="-40" y1="0" x2="-14" y2="0" class="bond-partial"/>
```

```css
.bond { stroke: var(--text-primary); stroke-width: 2.5; fill: none; stroke-linecap: round; }
.bond-thin { stroke: var(--text-primary); stroke-width: 1.5; fill: none; }
.bond-partial { stroke: var(--text-primary); stroke-width: 2; fill: none; stroke-dasharray: 4 3; }
.bond-wedge { fill: var(--text-primary); stroke: none; }
.bond-dash { stroke: var(--text-primary); stroke-width: 2; fill: none; stroke-dasharray: 2 2; }
```

### Lone Pairs and Charges

```xml
<!-- Lone pair electrons (dots) -->
<circle cx="-8" cy="-18" r="2" fill="var(--text-primary)"/>
<circle cx="0" cy="-18" r="2" fill="var(--text-primary)"/>

<!-- Formal negative charge -->
<text class="charge" x="12" y="-12" fill="#A32D2D" font-weight="bold">⊖</text>

<!-- Partial charges (delta notation) -->
<text class="partial" x="0" y="-18" text-anchor="middle" fill="#A32D2D">δ⁻</text>
<text class="partial" x="0" y="-22" text-anchor="middle" fill="#3B6D11">δ⁺</text>
```

```css
.charge { font-family: "Times New Roman", Georgia, serif; font-size: 12px; }
.partial { font-family: "Times New Roman", Georgia, serif; font-size: 11px; font-style: italic; }
```

### Curved Arrow (Electron Movement)

```xml
<defs>
  <marker id="curved-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 L3,5 Z" class="arrow-fill"/>
  </marker>
</defs>

<!-- Nucleophilic attack arrow -->
<path d="M -5,15 Q 30,60 70,25" class="arrow-curved" marker-end="url(#curved-arrow)"/>
```

```css
.arrow-curved { stroke: #534AB7; stroke-width: 2; fill: none; }
.arrow-fill { fill: #534AB7; }
```

### Transition State Brackets

```xml
<!-- Left bracket -->
<path d="M -75,-70 L -85,-70 L -85,75 L -75,75" class="ts-bracket"/>

<!-- Right bracket -->
<path d="M 95,-70 L 105,-70 L 105,75 L 95,75" class="ts-bracket"/>

<!-- Double dagger symbol -->
<text class="chem" x="115" y="-60" fill="var(--text-primary)">‡</text>
```

```css
.ts-bracket { stroke: var(--text-primary); stroke-width: 1.5; fill: none; }
```

## Energy Profile Diagram

### Axes

```xml
<!-- Y-axis (Energy) -->
<line x1="0" y1="280" x2="0" y2="0" class="axis" marker-end="url(#straight-arrow)"/>
<text class="t" x="-15" y="-10" text-anchor="middle" transform="rotate(-90 -15 140)">Potential Energy</text>

<!-- X-axis (Reaction Coordinate) -->
<line x1="0" y1="280" x2="600" y2="280" class="axis" marker-end="url(#straight-arrow)"/>
<text class="t" x="580" y="305" text-anchor="middle">Reaction Coordinate</text>
```

### Energy Curve

```xml
<!-- Filled area under curve -->
<path class="energy-fill" d="
  M 40,200 
  Q 150,200 250,50 
  Q 350,200 500,220 
  L 500,280 L 40,280 Z
"/>

<!-- Curve line -->
<path class="energy-curve" d="
  M 40,200 
  Q 100,200 150,150
  Q 200,80 250,50 
  Q 300,80 350,150
  Q 400,210 500,220
"/>
```

```css
.energy-curve { stroke: #534AB7; stroke-width: 2.5; fill: none; }
.energy-fill { fill: rgba(83, 74, 183, 0.1); }
```

### Energy Levels and Annotations

```xml
<!-- Reactants level -->
<line x1="20" y1="200" x2="80" y2="200" stroke="#3B6D11" stroke-width="2"/>
<text class="ts" x="50" y="218" text-anchor="middle">Reactants</text>

<!-- Transition state peak -->
<circle cx="250" cy="50" r="5" fill="#534AB7"/>
<line x1="250" y1="50" x2="250" y2="280" class="energy-level"/>
<text class="ts" x="250" y="30" text-anchor="middle" fill="#534AB7" font-weight="500">Transition State [‡]</text>

<!-- Products level (lower = exergonic) -->
<line x1="470" y1="220" x2="530" y2="220" stroke="#3B6D11" stroke-width="2"/>

<!-- Activation energy arrow -->
<line x1="100" y1="200" x2="100" y2="55" class="delta-arrow" marker-end="url(#delta-arrow)"/>
<text class="ts" x="85" y="125" text-anchor="end" fill="#3B6D11">E<tspan baseline-shift="sub" font-size="8">a</tspan></text>
```

```css
.energy-level { stroke: var(--text-secondary); stroke-width: 1; stroke-dasharray: 4 2; fill: none; }
.delta-arrow { stroke: #3B6D11; stroke-width: 1.5; fill: none; }
.delta-fill { fill: #3B6D11; }
```

## Chemistry Text Styles

```css
/* Chemistry notation (serif font for formulas) */
.chem { font-family: "Times New Roman", Georgia, serif; font-size: 16px; fill: var(--text-primary); }
.chem-sm { font-family: "Times New Roman", Georgia, serif; font-size: 12px; fill: var(--text-primary); }
.chem-lg { font-family: "Times New Roman", Georgia, serif; font-size: 18px; fill: var(--text-primary); }
```

## Subscript/Superscript in SVG

```xml
<!-- Subscript using tspan -->
<text class="ts">E<tspan baseline-shift="sub" font-size="8">a</tspan></text>

<!-- Superscript for charges -->
<text class="chem-sm">OH⁻</text>  <!-- Using Unicode superscript minus -->
<text class="chem-sm">CH₃Br</text>  <!-- Using Unicode subscript 3 -->
```

## Color Coding

| Element | Color | Hex |
|---------|-------|-----|
| Carbon | Dark gray | #2C2C2A |
| Hydrogen | Light cream | #F1EFE8 |
| Oxygen | Red | #E24B4A |
| Bromine | Brown | #993C1D |
| Nitrogen | Blue | #378ADD |
| Electron arrows | Purple | #534AB7 |
| Positive charge | Green | #3B6D11 |
| Negative charge | Red | #A32D2D |

## Layout Notes

- **ViewBox**: 800×680 (landscape for mechanism + energy profile)
- **Mechanism section**: y=60-300, showing reactants → TS → products
- **Energy profile**: y=320-630, with axes and curve
- **Atom sizes**: C/O/Br ~12-16px radius, H ~7-8px radius
- **Bond lengths**: ~25-40px between atom centers
- **Spacing**: ~140px between mechanism stages

## When to Use This Pattern

Use this diagram style for:
- Organic reaction mechanisms (SN1, SN2, E1, E2, additions, eliminations)
- Reaction energy profiles and kinetics
- Stereochemistry illustrations
- Enzyme mechanism diagrams
- Transition state theory visualization
- Any chemistry concept requiring molecular structures
