# Commercial Aircraft Structure

A physical/structural diagram showing an aircraft side profile using appropriate SVG shapes beyond rectangles - paths, polygons, ellipses for realistic representation.

## Key Patterns Used

- **Path elements**: Curved fuselage body with nose cone using quadratic bezier curves
- **Polygon elements**: Tapered wing shape, triangular stabilizers, control surfaces
- **Ellipse elements**: Engines (cylinders), wheels (circles)
- **Line elements**: Landing gear struts, leader lines for labels
- **Dashed strokes**: Interior sections (fuel tank), movable control surfaces (rudder, elevator)
- **Layered composition**: Cabin sections drawn inside the fuselage shape
- **Leader lines with labels**: Connect labels to components they describe

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg">

  <!-- FUSELAGE - main body cylinder with nose cone -->
  <path class="fuselage" d="
    M 80 180
    Q 40 180 40 200
    Q 40 220 80 220
    L 560 220
    Q 580 220 580 200
    Q 580 180 560 180
    Z
  "/>
  
  <!-- Nose cone -->
  <path class="fuselage" d="
    M 80 180
    Q 50 180 35 200
    Q 50 220 80 220
  " fill="none" stroke-width="1"/>

  <!-- COCKPIT windows -->
  <path class="cockpit" d="
    M 45 190
    L 75 185
    L 75 200
    L 50 200
    Z
  "/>
  <line x1="55" y1="188" x2="55" y2="200" stroke="#534AB7" stroke-width="0.5"/>
  <line x1="65" y1="186" x2="65" y2="200" stroke="#534AB7" stroke-width="0.5"/>

  <!-- CABIN SECTIONS (inside fuselage) -->
  <!-- First class -->
  <rect class="first-class" x="85" y="183" width="50" height="34" rx="2"/>
  <text class="tl" x="110" y="203" text-anchor="middle">First</text>
  
  <!-- Business class -->
  <rect class="business-class" x="140" y="183" width="80" height="34" rx="2"/>
  <text class="tl" x="180" y="203" text-anchor="middle">Business</text>
  
  <!-- Economy class -->
  <rect class="economy-class" x="225" y="183" width="200" height="34" rx="2"/>
  <text class="tl" x="325" y="203" text-anchor="middle">Economy</text>

  <!-- CARGO HOLD (lower section indication) -->
  <line x1="85" y1="217" x2="520" y2="217" class="leader"/>
  <text class="tl" x="300" y="228" text-anchor="middle" opacity=".6">Cargo hold below deck</text>

  <!-- WING - main wing shape -->
  <polygon class="wing" points="
    200,220
    120,300
    130,305
    160,305
    340,235
    340,220
  "/>
  
  <!-- Wing fuel tank (dashed interior) -->
  <polygon class="fuel-tank" points="
    210,225
    150,280
    160,283
    180,283
    310,232
    310,225
  "/>
  <text class="tl" x="220" y="260" opacity=".7">Fuel</text>

  <!-- Flaps (trailing edge) -->
  <polygon class="flap" points="
    130,300
    120,305
    160,310
    165,305
  "/>
  <text class="tl" x="143" y="320">Flaps</text>

  <!-- ENGINE under wing -->
  <ellipse class="engine" cx="175" cy="285" rx="25" ry="12"/>
  <ellipse cx="155" cy="285" rx="8" ry="10" fill="none" stroke="#993C1D" stroke-width="0.5"/>
  <!-- Engine pylon -->
  <line x1="175" y1="273" x2="190" y2="245" stroke="#5F5E5A" stroke-width="2"/>
  <text class="tl" x="175" y="308" text-anchor="middle">Engine</text>

  <!-- TAIL SECTION -->
  <!-- Vertical stabilizer -->
  <polygon class="tail-v" points="
    520,180
    560,100
    580,100
    580,180
  "/>
  <text class="tl" x="565" y="150" text-anchor="middle">Vertical</text>
  <text class="tl" x="565" y="162" text-anchor="middle">stabilizer</text>
  
  <!-- Rudder -->
  <polygon points="575,105 590,105 590,178 580,178" fill="none" stroke="#185FA5" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text class="tl" x="595" y="145" opacity=".6">Rudder</text>

  <!-- Horizontal stabilizer -->
  <polygon class="tail-h" points="
    500,195
    460,175
    465,170
    580,170
    580,180
    520,195
  "/>
  <text class="tl" x="510" y="166">Horizontal stabilizer</text>
  
  <!-- Elevator -->
  <polygon points="462,174 450,168 455,163 467,169" fill="none" stroke="#185FA5" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text class="tl" x="440" y="158" opacity=".6">Elevator</text>

  <!-- LANDING GEAR -->
  <!-- Nose gear -->
  <line class="gear" x1="100" y1="220" x2="100" y2="260" stroke-width="3"/>
  <ellipse class="wheel" cx="100" cy="268" rx="8" ry="10"/>
  <text class="tl" x="100" y="290" text-anchor="middle">Nose gear</text>

  <!-- Main gear (under wing/fuselage junction) -->
  <line class="gear" x1="280" y1="220" x2="280" y2="270" stroke-width="4"/>
  <line class="gear" x1="268" y1="265" x2="292" y2="265" stroke-width="3"/>
  <ellipse class="wheel" cx="268" cy="278" rx="10" ry="12"/>
  <ellipse class="wheel" cx="292" cy="278" rx="10" ry="12"/>
  <text class="tl" x="280" y="302" text-anchor="middle">Main gear</text>

  <!-- LABELS with leader lines -->
  <!-- Cockpit label -->
  <line class="leader" x1="60" y1="175" x2="60" y2="140"/>
  <text class="ts" x="60" y="132" text-anchor="middle">Cockpit</text>

  <!-- Wing label -->
  <line class="leader" x1="250" y1="250" x2="290" y2="330"/>
  <text class="ts" x="290" y="345" text-anchor="middle">Wing structure</text>
  <text class="tl" x="290" y="358" text-anchor="middle">Spars, ribs, skin</text>

  <!-- Fuselage label -->
  <line class="leader" x1="400" y1="180" x2="400" y2="140"/>
  <text class="ts" x="400" y="132" text-anchor="middle">Fuselage</text>
  <text class="tl" x="400" y="145" text-anchor="middle">Pressure vessel</text>

</svg>
```

## CSS Classes for Physical Diagrams

When creating physical/structural diagrams, define semantic classes for each component type:

```css
/* Structure shapes */
.fuselage { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.wing { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.tail-v { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.tail-h { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }

/* Interior sections */
.cockpit { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1; }
.first-class { fill: #FBEAF0; stroke: #993556; stroke-width: 0.5; }
.business-class { fill: #FAECE7; stroke: #993C1D; stroke-width: 0.5; }
.economy-class { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }
.cargo { fill: #D3D1C7; stroke: #5F5E5A; stroke-width: 0.5; }

/* Systems */
.engine { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
.fuel-tank { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.5; stroke-dasharray: 3 2; }
.flap { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }

/* Mechanical */
.gear { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }
.wheel { fill: #2C2C2A; stroke: #1a1a18; stroke-width: 0.5; }
```

## Shape Selection Guide

| Physical form | SVG element | Example |
|---------------|-------------|---------|
| Curved body | `<path>` with Q (quadratic) or C (cubic) curves | Fuselage, nose cone |
| Tapered/angular | `<polygon>` | Wings, stabilizers |
| Cylindrical | `<ellipse>` | Engines, wheels, tanks |
| Linear structure | `<line>` | Struts, pylons, gear legs |
| Internal sections | `<rect>` inside parent shape | Cabin classes |
| Dashed boundaries | `stroke-dasharray` on any shape | Fuel tanks, control surfaces |

## Layout Notes

- **ViewBox**: 680×400 (wider aspect ratio suits side profile)
- **Layering**: Draw outer structures first, then interior details on top
- **Leader lines**: Use `.leader` class (dashed) to connect labels to components
- **Text sizes**: Use `.tl` (10px) for component labels, `.ts` (12px) for section labels
- **Semantic colors**: Group by system (structure=blue, propulsion=coral, fuel=amber, etc.)
