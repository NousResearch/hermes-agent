# Electricity Grid: Generation to Consumption

A left-to-right flow diagram showing electricity from multiple generation sources through transmission and distribution networks to end consumers. Demonstrates multi-stage flow layout, voltage level visual hierarchy, and smart grid data overlay.

## Key Patterns Used

- **Multi-stage horizontal flow**: Four distinct columns (Generation → Transmission → Distribution → Consumption)
- **Stage dividers**: Vertical dashed lines separating each phase
- **Voltage level hierarchy**: Different line weights/colors for HV, MV, LV
- **Smart grid data overlay**: Dashed data flow lines from control center
- **Capacity labels**: Power ratings on generation sources
- **Multiple source convergence**: Four generators feeding into single transmission grid

## New Shape Techniques

### Nuclear Plant (cooling tower + reactor)
```xml
<!-- Cooling tower (hyperbolic curve) -->
<path class="nuclear-tower" d="M 25 80 Q 15 60 20 40 Q 25 20 40 15 Q 55 20 60 40 Q 65 60 55 80 Z"/>
<!-- Steam clouds -->
<ellipse class="nuclear-steam" cx="40" cy="8" rx="12" ry="6"/>
<!-- Reactor dome -->
<rect class="nuclear-building" x="65" y="45" width="40" height="35" rx="3"/>
<ellipse class="nuclear-building" cx="85" cy="45" rx="20" ry="8"/>
```

### Gas Peaker Plant (with flames)
```xml
<rect class="gas-plant" x="0" y="25" width="70" height="40" rx="3"/>
<!-- Smokestacks -->
<rect class="gas-stack" x="15" y="5" width="8" height="25" rx="1"/>
<!-- Flame -->
<path class="gas-flame" d="M 19 5 Q 17 0 19 -3 Q 21 0 19 5"/>
<!-- Turbine housing -->
<ellipse class="gas-plant" cx="55" cy="45" rx="12" ry="8"/>
```

### Transmission Pylon with Insulators
```xml
<!-- Tapered tower -->
<polygon class="pylon" points="20,0 25,0 30,80 15,80"/>
<!-- Cross arms -->
<line class="pylon-arm" x1="5" y1="10" x2="40" y2="10"/>
<line class="pylon-arm" x1="8" y1="25" x2="37" y2="25"/>
<!-- Insulators (where lines attach) -->
<circle class="insulator" cx="8" cy="10" r="3"/>
<circle class="insulator" cx="37" cy="10" r="3"/>
```

### Transformer Symbol
```xml
<!-- Two coils with core -->
<circle class="transformer-coil" cx="25" cy="25" r="12"/>
<circle class="transformer-coil" cx="55" cy="25" r="12"/>
<rect class="transformer-core" x="35" y="15" width="10" height="20" rx="2"/>
<!-- Busbars -->
<line x1="0" y1="15" x2="-10" y2="15" stroke="#EF9F27" stroke-width="3"/>
```

### Pole-mounted Transformer
```xml
<rect class="pole" x="18" y="0" width="4" height="60"/>
<line x1="10" y1="8" x2="30" y2="8" stroke="#854F0B" stroke-width="2"/>
<rect class="dist-transformer" x="8" y="15" width="24" height="18" rx="2"/>
<line class="lv-line" x1="20" y1="33" x2="20" y2="60"/>
```

### House with Roof
```xml
<rect class="home" x="0" y="25" width="35" height="30" rx="2"/>
<polygon class="home-roof" points="0,25 17,8 35,25"/>
<!-- Door -->
<rect x="8" y="35" width="8" height="15" fill="#085041"/>
<!-- Window -->
<rect x="22" y="32" width="8" height="8" fill="#9FE1CB"/>
```

### Factory Building
```xml
<rect class="factory" x="0" y="15" width="90" height="50" rx="3"/>
<!-- Smokestacks -->
<rect class="factory-stack" x="15" y="0" width="10" height="20"/>
<!-- Windows row -->
<rect x="10" y="30" width="15" height="12" fill="#F5C4B3"/>
<rect x="30" y="30" width="15" height="12" fill="#F5C4B3"/>
<!-- Loading dock -->
<rect x="55" y="50" width="30" height="15" fill="#993C1D"/>
```

### EV Charger with Car
```xml
<!-- Charging station -->
<rect class="ev-charger" x="20" y="0" width="25" height="45" rx="3"/>
<rect x="24" y="5" width="17" height="12" rx="1" fill="#3C3489"/>
<!-- Cable -->
<path d="M 32 20 Q 32 35 45 40" stroke="#534AB7" stroke-width="2" fill="none"/>
<circle cx="45" cy="40" r="4" fill="#534AB7"/>
<!-- Status light -->
<circle cx="32" cy="38" r="3" fill="#97C459"/>

<!-- EV Car -->
<path class="ev-car" d="M 5 20 L 5 12 Q 5 5 15 5 L 45 5 Q 55 5 55 12 L 55 20 Z"/>
<!-- Windows -->
<rect x="10" y="8" width="15" height="8" rx="2" fill="#534AB7"/>
<!-- Wheels -->
<circle cx="15" cy="22" r="5" fill="#2C2C2A"/>
<!-- Charging bolt icon -->
<path d="M 28 12 L 32 8 L 30 11 L 34 11 L 30 16 L 32 13 Z" fill="#97C459"/>
```

## Voltage Level Line Styles

```css
/* High voltage (transmission) - thick, bright */
.hv-line { stroke: #EF9F27; stroke-width: 2.5; fill: none; }

/* Medium voltage (distribution) - medium */
.mv-line { stroke: #BA7517; stroke-width: 2; fill: none; }

/* Low voltage (consumer) - thin, darker */
.lv-line { stroke: #854F0B; stroke-width: 1.5; fill: none; }

/* Smart grid data - dashed purple */
.data-flow { stroke: #7F77DD; stroke-width: 1; fill: none; stroke-dasharray: 3 2; opacity: 0.7; }
```

## Flow Arrow Marker

```xml
<defs>
  <marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" 
          markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 Z" fill="#EF9F27"/>
  </marker>
</defs>
<!-- Usage -->
<line x1="140" y1="105" x2="210" y2="105" class="hv-line" marker-end="url(#flow-arrow)"/>
```

## CSS Classes

```css
/* Generation */
.nuclear-tower { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.nuclear-building { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1; }
.solar-panel { fill: #3C3489; stroke: #534AB7; stroke-width: 0.5; }
.wind-tower { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.wind-blade { fill: #F1EFE8; stroke: #888780; stroke-width: 0.5; }
.gas-plant { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
.gas-flame { fill: #EF9F27; }

/* Transmission */
.pylon { fill: #5F5E5A; stroke: #444441; stroke-width: 0.5; }
.insulator { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.5; }
.substation { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.transformer-coil { fill: none; stroke: #185FA5; stroke-width: 1.5; }

/* Distribution */
.pole { fill: #854F0B; stroke: #633806; stroke-width: 0.5; }
.dist-transformer { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1; }

/* Consumption */
.home { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1; }
.home-roof { fill: #0F6E56; stroke: #085041; stroke-width: 0.5; }
.factory { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
.ev-charger { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1; }
.ev-car { fill: #3C3489; stroke: #534AB7; stroke-width: 0.5; }

/* Smart grid */
.smart-grid { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1.5; }
```

## Layout Notes

- **ViewBox**: 820×520 (wide for 4-column layout)
- **Column widths**: ~200px per stage
- **Stage dividers**: Vertical dashed lines at x=200, 420, 620
- **Stage labels**: Top of diagram, uppercase for emphasis
- **Flow direction**: Left-to-right with arrows showing power flow
- **Data overlay**: Smart grid data lines use different style (dashed purple) to distinguish from power lines
- **Capacity labels**: Show MW ratings on generators for context
- **Voltage labels**: Show transformation ratios at substations
