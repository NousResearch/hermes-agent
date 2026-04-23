# Smart City Infrastructure

A multi-system integration diagram showing interconnected city infrastructure (power, water, transport) connected through a central IoT platform with a citizen dashboard on top. Demonstrates hub-spoke layout, diverse physical shapes, and UI mockups.

## Key Patterns Used

- **Hub-spoke layout**: Central IoT platform with radiating data connections to subsystems
- **Connection dots**: Visual indicators where data lines attach to the central hub
- **Dashboard/UI mockup**: Screen with mini-charts, gauges, and status indicators
- **Multi-system integration**: Three independent systems unified by central platform
- **Semantic line styles**: Different stroke styles for data (dashed), power, water, roads
- **Physical infrastructure shapes**: Solar panels, wind turbines, dams, pipes, roads, vehicles

## New Shape Techniques

### Solar Panels (angled polygons with grid lines)
```xml
<polygon class="solar-panel" points="0,25 35,8 38,12 3,29"/>
<line class="solar-frame" x1="12" y1="22" x2="24" y2="13"/>
<line x1="19" y1="29" x2="19" y2="40" stroke="#5F5E5A" stroke-width="2"/>
```

### Wind Turbine (tower + nacelle + blades)
```xml
<!-- Tapered tower -->
<polygon class="wind-tower" points="20,70 30,70 28,25 22,25"/>
<!-- Nacelle -->
<rect class="wind-hub" x="18" y="20" width="14" height="8" rx="2"/>
<!-- Hub -->
<circle class="wind-hub" cx="25" cy="18" r="5"/>
<!-- Blades (rotated ellipses) -->
<ellipse class="wind-blade" cx="25" cy="5" rx="3" ry="13"/>
<ellipse class="wind-blade" cx="14" cy="26" rx="3" ry="13" transform="rotate(-120, 25, 18)"/>
<ellipse class="wind-blade" cx="36" cy="26" rx="3" ry="13" transform="rotate(120, 25, 18)"/>
```

### Battery with Charge Level
```xml
<rect class="battery" x="0" y="0" width="45" height="65" rx="5"/>
<!-- Terminals -->
<rect x="10" y="-6" width="10" height="8" rx="2" fill="#27500A"/>
<rect x="25" y="-6" width="10" height="8" rx="2" fill="#27500A"/>
<!-- Charge level fill -->
<rect class="battery-level" x="5" y="12" width="35" height="48" rx="3"/>
<text x="22" y="42" text-anchor="middle" fill="#173404" style="font-size:10px">85%</text>
```

### Dam/Reservoir with Water Waves
```xml
<!-- Dam wall -->
<polygon class="reservoir-wall" points="0,60 10,0 70,0 80,60"/>
<!-- Water behind dam -->
<polygon class="water" points="12,10 68,10 68,55 75,55 75,58 5,58 5,55 12,55"/>
<!-- Wave effect -->
<path d="M 15 25 Q 25 22 35 25 Q 45 28 55 25" fill="none" stroke="#378ADD" stroke-width="1" opacity="0.5"/>
```

### Pipe Network with Joints and Valves
```xml
<path class="pipe" d="M 80 85 L 110 85"/>
<circle class="pipe-joint" cx="10" cy="30" r="8"/>
<circle class="valve" cx="190" cy="85" r="6"/>
<!-- Distribution branches -->
<path class="pipe-thin" d="M 18 30 L 50 30"/>
<path class="pipe-thin" d="M 10 22 L 10 5 L 50 5"/>
```

### Road Intersection with Lane Markings
```xml
<!-- Road surface -->
<line class="road" x1="0" y1="50" x2="170" y2="50"/>
<line class="road-mark" x1="10" y1="50" x2="160" y2="50"/>
<!-- Cross road -->
<line class="road" x1="85" y1="0" x2="85" y2="100"/>
<line class="road-mark" x1="85" y1="10" x2="85" y2="90"/>
<!-- Embedded sensors -->
<circle class="sensor" cx="40" cy="50" r="5"/>
```

### Traffic Light with Signal States
```xml
<rect class="traffic-light" x="0" y="0" width="14" height="32" rx="3"/>
<circle class="light-red" cx="7" cy="8" r="4"/>
<circle class="light-off" cx="7" cy="16" r="4"/>
<circle class="light-off" cx="7" cy="24" r="4"/>
```

### Bus with Windows and Wheels
```xml
<rect class="bus" x="0" y="0" width="55" height="28" rx="6"/>
<!-- Windows -->
<rect class="bus-window" x="5" y="5" width="12" height="12" rx="2"/>
<rect class="bus-window" x="20" y="5" width="12" height="12" rx="2"/>
<!-- Wheels with hubcaps -->
<circle cx="14" cy="30" r="6" fill="#2C2C2A"/>
<circle cx="14" cy="30" r="3" fill="#5F5E5A"/>
```

### Dashboard UI Mockup
```xml
<!-- Monitor frame -->
<rect class="dashboard" x="0" y="0" width="200" height="120" rx="8"/>
<!-- Screen -->
<rect class="screen" x="10" y="10" width="180" height="85" rx="4"/>
<!-- Mini bar chart -->
<rect class="screen-content" x="18" y="18" width="50" height="35" rx="2"/>
<rect class="screen-chart" x="22" y="38" width="8" height="12"/>
<rect class="screen-chart" x="33" y="32" width="8" height="18"/>
<!-- Gauge -->
<circle class="screen-bar" cx="100" cy="35" r="12"/>
<text x="100" y="39" text-anchor="middle" fill="#E8E6DE" style="font-size:8px">78%</text>
<!-- Status indicators -->
<circle cx="35" cy="74" r="6" fill="#97C459"/>
<circle cx="75" cy="74" r="6" fill="#97C459"/>
<circle cx="115" cy="74" r="6" fill="#EF9F27"/>
```

### Hexagonal IoT Hub with Connection Points
```xml
<!-- Outer hexagon -->
<polygon class="iot-hex" points="0,-45 39,-22 39,22 0,45 -39,22 -39,-22"/>
<!-- Inner hexagon -->
<polygon class="iot-inner" points="0,-20 17,-10 17,10 0,20 -17,10 -17,-10"/>
<!-- Connection dots on data lines -->
<circle cx="321" cy="248" r="4" fill="#7F77DD"/>
```

## CSS Classes for Infrastructure

```css
/* Power system */
.solar-panel { fill: #3C3489; stroke: #534AB7; stroke-width: 0.5; }
.solar-frame { fill: none; stroke: #EEEDFE; stroke-width: 0.5; }
.wind-tower { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.wind-blade { fill: #F1EFE8; stroke: #888780; stroke-width: 0.5; }
.battery { fill: #27500A; stroke: #3B6D11; stroke-width: 1.5; }
.battery-level { fill: #97C459; }
.power-line { stroke: #EF9F27; stroke-width: 2; fill: none; }

/* Water system */
.reservoir-wall { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.water { fill: #85B7EB; stroke: #378ADD; stroke-width: 0.5; }
.pipe { fill: none; stroke: #378ADD; stroke-width: 4; stroke-linecap: round; }
.pipe-joint { fill: #185FA5; stroke: #0C447C; stroke-width: 1; }
.valve { fill: #0C447C; stroke: #185FA5; stroke-width: 1; }

/* Transport */
.road { stroke: #888780; stroke-width: 8; fill: none; stroke-linecap: round; }
.road-mark { stroke: #F1EFE8; stroke-width: 1; fill: none; stroke-dasharray: 6 4; }
.traffic-light { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }
.light-red { fill: #E24B4A; }
.light-green { fill: #97C459; }
.light-off { fill: #2C2C2A; }
.bus { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1.5; }

/* Data/IoT */
.data-line { stroke: #7F77DD; stroke-width: 2; fill: none; stroke-dasharray: 4 3; }
.iot-hex { fill: #EEEDFE; stroke: #534AB7; stroke-width: 2; }

/* Dashboard */
.dashboard { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1.5; }
.screen { fill: #1a1a18; }
.screen-chart { fill: #5DCAA5; }
```

## Layout Notes

- **ViewBox**: 720×620 (wider for three-column system layout)
- **Hub position**: Central IoT at (360, 270) - geometric center
- **Data lines**: Use quadratic curves or L-shaped paths, add connection dots at hub attachment points
- **System spacing**: ~200px width per system section
- **Vertical layers**: Dashboard (top) → IoT Hub (middle) → Systems (bottom)
- **Component grouping**: Use `<g transform="translate(x,y)">` for each major component for easy positioning
