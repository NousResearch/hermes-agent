# Journey of a Banana: From Tree to Smoothie

A narrative journey diagram following a single banana across 3,000 miles and 3 weeks, from harvest in Costa Rica to a smoothie in the consumer's kitchen. Demonstrates storytelling through visualization, winding path layout, and progressive state changes.

## Key Patterns Used

- **Winding journey path**: S-curve connecting all stages visually
- **Location markers**: Country flags and place names for geographic context
- **Progressive state changes**: Banana color changes (green → yellow → brown → frozen → smoothie)
- **Narrative details**: Fun elements like spider check, stickers, price tags
- **Timeline**: Bottom timeline showing duration of journey
- **Environmental context**: Ocean waves, gas clouds, store awning

## New Shape Techniques

### Banana (curved fruit shape)
```xml
<!-- Green banana -->
<path class="banana-green" d="M 5 0 Q 0 10 3 20 Q 6 25 10 20 Q 13 10 8 0 Z"/>

<!-- Yellow banana -->
<path class="banana-yellow" d="M 0 5 Q -6 18 0 32 Q 7 40 15 30 Q 20 15 12 5 Z"/>

<!-- Brown overripe banana with spots -->
<path class="banana-brown" d="M 0 5 Q -5 15 0 28 Q 6 35 14 26 Q 18 14 12 5 Z"/>
<circle class="banana-spots" cx="5" cy="15" r="1.5"/>
<circle class="banana-spots" cx="9" cy="20" r="1"/>
```

### Banana Tree
```xml
<!-- Trunk -->
<rect class="tree-trunk" x="55" y="50" width="15" height="60" rx="3"/>
<!-- Leaves (rotated ellipses) -->
<ellipse class="tree-leaf" cx="62" cy="45" rx="40" ry="15" transform="rotate(-20, 62, 45)"/>
<ellipse class="tree-leaf" cx="62" cy="50" rx="35" ry="12" transform="rotate(25, 62, 50)"/>
<!-- Banana bunch hanging -->
<g transform="translate(40, 55)">
  <path class="banana-green" d="M 5 0 Q 0 10 3 20 Q 6 25 10 20 Q 13 10 8 0 Z"/>
  <path class="banana-green" d="M 12 2 Q 8 12 11 22 Q 14 27 18 22 Q 21 12 16 2 Z"/>
  <rect class="stem" x="8" y="-5" width="12" height="8" rx="2"/>
</g>
```

### Cargo Ship
```xml
<!-- Ocean waves -->
<path class="ocean" d="M 0 90 Q 30 85 60 90 Q 90 95 120 90 Q 150 85 180 90 L 180 110 L 0 110 Z" opacity="0.5"/>
<!-- Hull -->
<path class="ship-hull" d="M 20 90 L 30 60 L 160 60 L 170 90 Q 150 95 95 95 Q 40 95 20 90 Z"/>
<!-- Deck -->
<rect class="ship-deck" x="40" y="45" width="110" height="18" rx="2"/>
<!-- Reefer containers -->
<rect class="container" x="45" y="25" width="30" height="22" rx="2"/>
<!-- Refrigeration symbol -->
<text x="60" y="40" text-anchor="middle" fill="#185FA5" style="font-size:10px">❄</text>
<!-- Smoke stack -->
<rect x="145" y="35" width="8" height="15" fill="#444441"/>
```

### Inspector Figure
```xml
<!-- Body -->
<rect class="inspector" x="10" y="20" width="25" height="35" rx="3"/>
<!-- Head -->
<circle class="inspector" cx="22" cy="12" r="10"/>
<!-- Hat -->
<rect x="12" y="2" width="20" height="6" rx="2" fill="#534AB7"/>
<!-- Clipboard -->
<rect class="clipboard" x="38" y="28" width="15" height="20" rx="2"/>
<line x1="42" y1="34" x2="50" y2="34" stroke="#888780" stroke-width="1"/>
```

### Spider with "No" Symbol
```xml
<circle cx="15" cy="15" r="18" fill="none" stroke="#A32D2D" stroke-width="2"/>
<line x1="3" y1="3" x2="27" y2="27" stroke="#A32D2D" stroke-width="2"/>
<!-- Spider body -->
<ellipse class="spider" cx="15" cy="15" rx="4" ry="5"/>
<ellipse class="spider" cx="15" cy="10" rx="3" ry="3"/>
<!-- Legs -->
<line x1="12" y1="14" x2="5" y2="10" stroke="#2C2C2A" stroke-width="1"/>
<line x1="18" y1="14" x2="25" y2="10" stroke="#2C2C2A" stroke-width="1"/>
```

### Blender with Smoothie
```xml
<!-- Blender jar -->
<path class="blender" d="M 5 5 L 0 45 L 35 45 L 30 5 Z"/>
<!-- Smoothie inside (wavy top) -->
<path class="smoothie" d="M 3 20 L 0 45 L 35 45 L 32 20 Q 25 18 17 22 Q 10 18 3 20 Z"/>
<!-- Blender base -->
<rect class="blender" x="-2" y="45" width="40" height="12" rx="3"/>
<!-- Lid -->
<rect x="8" y="0" width="20" height="8" rx="2" fill="#AFA9EC" stroke="#534AB7"/>
<!-- Banana chunks floating -->
<ellipse cx="12" cy="32" rx="4" ry="2" fill="#FAC775"/>
```

### Winding Journey Path
```xml
<path class="journey-path" d="
  M 80 100 
  L 200 100 
  Q 280 100 280 150 
  L 280 180
  Q 280 220 320 220
  L 520 220
  Q 560 220 560 260
  L 560 320
  Q 560 360 520 360
  L 280 360
  ...
"/>
```

## CSS Classes

```css
/* Journey */
.journey-path { stroke: #D3D1C7; stroke-width: 3; fill: none; stroke-linecap: round; }

/* Banana ripeness stages */
.banana-green { fill: #97C459; stroke: #3B6D11; stroke-width: 0.5; }
.banana-yellow { fill: #FAC775; stroke: #BA7517; stroke-width: 0.5; }
.banana-brown { fill: #854F0B; stroke: #633806; stroke-width: 0.5; }
.banana-spots { fill: #633806; }

/* Environment elements */
.tree-trunk { fill: #854F0B; stroke: #633806; stroke-width: 1; }
.tree-leaf { fill: #97C459; stroke: #3B6D11; stroke-width: 0.5; }
.ocean { fill: #85B7EB; }
.ship-hull { fill: #5F5E5A; stroke: #444441; stroke-width: 1; }
.container { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.gas-cloud { fill: #C0DD97; stroke: #97C459; stroke-width: 0.5; opacity: 0.6; }

/* Buildings */
.packhouse { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.warehouse { fill: #FAEEDA; stroke: #854F0B; stroke-width: 1; }
.store { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1; }

/* Kitchen */
.counter { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
.blender { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1; }
.smoothie { fill: #FAC775; }
.freezer { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }

/* Details */
.sticker { fill: #378ADD; stroke: #185FA5; stroke-width: 0.3; }
.spider { fill: #2C2C2A; stroke: #1a1a18; stroke-width: 0.3; }
```

## Layout Notes

- **ViewBox**: 850×680 (tall for winding path)
- **Path style**: S-curve winding path connects all 7 stages
- **Location labels**: Country flags + place names anchor geographic context
- **State progression**: Same object (banana) shown in different states throughout
- **Timeline**: Horizontal timeline at bottom shows journey duration
- **Narrative elements**: Fun details (spider, stickers, price tags) add storytelling value
- **Environmental context**: Ocean waves, gas clouds, awnings create sense of place
