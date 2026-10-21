# Dashboard Patterns

Building blocks for UI/dashboard mockups inside a concept diagram — admin panels, monitoring dashboards, control interfaces, status displays.

## Pattern

A "screen" is a rounded dark rect inside a lighter "frame" rect, with chart/gauge/indicator elements nested on top.

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
<circle cx="35" cy="74" r="6" fill="#97C459"/> <!-- green = ok -->
<circle cx="75" cy="74" r="6" fill="#EF9F27"/> <!-- amber = warning -->
<circle cx="115" cy="74" r="6" fill="#E24B4A"/> <!-- red = alert -->
```

## CSS

```css
.dashboard      { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1.5; }
.screen         { fill: #1a1a18; }
.screen-content { fill: #2C2C2A; }
.screen-chart   { fill: #5DCAA5; }
.screen-bar     { fill: #7F77DD; }
.screen-alert   { fill: #E24B4A; }
```

## Tips

- Dashboard screens stay dark in both light and dark mode — they represent actual monitor glass.
- Keep on-screen text small (`font-size:8px` or `10px`) and high-contrast (near-white fill on dark).
- Use the status triad green/amber/red consistently — OK / warning / alert.
- A single dashboard usually sits on top of an infrastructure hub diagram as a unified view (see `examples/smart-city-infrastructure.md`).
