# Modern Onshore Wind Turbine Structure

A physical/structural cross-section diagram showing all major components of a modern wind turbine from underground foundation to blade tips.

## Key Patterns Used

- **Underground section**: Soil layers, deep concrete foundation with rebar reinforcement grid, spread footing
- **Cross-section view**: Tower wall thickness shown, internal components visible
- **Tapered tower**: Path elements creating realistic tower silhouette that narrows toward top
- **Internal access**: Ladder with rungs, elevator shaft inside tower
- **Cable routing**: Power cables running from nacelle down through tower to transformer
- **Nacelle cutaway**: Gearbox, generator, brake, yaw system all visible inside housing
- **Rotor assembly**: Hub with pitch motors at blade roots, three composite blades with gradient fill
- **Ground level marker**: Clear separation between above/below ground
- **Component color coding**: Each system type has distinct color (blue=generator, gold=gearbox, red=brake, green=yaw, purple=pitch)
- **Legend bar**: Quick reference for color meanings

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 920" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <!-- Blade gradient for 3D effect -->
    <linearGradient id="bladeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#D3D1C7"/>
      <stop offset="50%" style="stop-color:#F1EFE8"/>
      <stop offset="100%" style="stop-color:#B4B2A9"/>
    </linearGradient>
  </defs>

  <!-- ===== GROUND LEVEL LINE ===== -->
  <line x1="40" y1="680" x2="640" y2="680" stroke="#3B6D11" stroke-width="2"/>
  <text class="tl" x="45" y="675">Ground level</text>

  <!-- ===== UNDERGROUND: FOUNDATION ===== -->
  
  <!-- Soil layers -->
  <rect x="120" y="680" width="300" height="180" class="soil"/>
  <rect x="120" y="780" width="300" height="80" class="soil-dark"/>
  
  <!-- Deep concrete foundation -->
  <path d="M170 680 L170 820 L200 850 L340 850 L370 820 L370 680 Z" class="concrete"/>
  <!-- Foundation base spread -->
  <path d="M140 820 L170 820 L200 850 L340 850 L370 820 L400 820 L400 860 L140 860 Z" class="concrete-dark"/>
  
  <!-- Rebar reinforcement -->
  <g class="rebar">
    <line x1="185" y1="700" x2="185" y2="840"/>
    <line x1="210" y1="700" x2="210" y2="845"/>
    <line x1="235" y1="700" x2="235" y2="848"/>
    <line x1="260" y1="700" x2="260" y2="848"/>
    <line x1="285" y1="700" x2="285" y2="848"/>
    <line x1="310" y1="700" x2="310" y2="845"/>
    <line x1="335" y1="700" x2="335" y2="840"/>
    <!-- Horizontal rebar -->
    <line x1="175" y1="720" x2="365" y2="720"/>
    <line x1="175" y1="760" x2="365" y2="760"/>
    <line x1="175" y1="800" x2="365" y2="800"/>
    <line x1="155" y1="835" x2="385" y2="835"/>
  </g>
  
  <!-- Foundation labels -->
  <line x1="410" y1="770" x2="480" y2="770" class="leader"/>
  <text class="ts" x="485" y="766">Deep concrete foundation</text>
  <text class="tl" x="485" y="778">Reinforced with steel rebar</text>
  <text class="tl" x="485" y="790">15-25m deep typical</text>
  
  <line x1="400" y1="850" x2="480" y2="870" class="leader"/>
  <text class="ts" x="485" y="866">Foundation spread footing</text>
  <text class="tl" x="485" y="878">Distributes load to soil</text>

  <!-- ===== TOWER BASE ===== -->
  
  <!-- Tower base flange -->
  <ellipse cx="270" cy="680" rx="70" ry="12" class="concrete-dark"/>
  <rect x="200" y="668" width="140" height="12" class="tower"/>
  
  <!-- Transformer at base -->
  <g transform="translate(470, 640)">
    <rect x="0" y="0" width="50" height="40" rx="3" class="transformer"/>
    <!-- Cooling fins -->
    <rect x="52" y="5" width="4" height="30" class="transformer-fin"/>
    <rect x="58" y="5" width="4" height="30" class="transformer-fin"/>
    <rect x="64" y="5" width="4" height="30" class="transformer-fin"/>
    <!-- Connection box -->
    <rect x="10" y="-8" width="30" height="10" rx="2" class="transformer-fin"/>
  </g>
  <line x1="470" y1="660" x2="430" y2="640" class="leader"/>
  <text class="ts" x="385" y="636" text-anchor="end">Transformer</text>
  <text class="tl" x="385" y="648" text-anchor="end">Steps up voltage for grid</text>

  <!-- ===== TUBULAR STEEL TOWER ===== -->
  
  <!-- Tower outer shell (tapered) -->
  <path d="M200 680 L220 200 L320 200 L340 680 Z" class="tower"/>
  
  <!-- Tower inner surface (cutaway) -->
  <path d="M215 680 L232 210 L308 210 L325 680 Z" class="tower-inner"/>
  
  <!-- Tower section joints -->
  <line x1="205" y1="550" x2="335" y2="550" class="tower-section"/>
  <line x1="210" y1="420" x2="330" y2="420" class="tower-section"/>
  <line x1="215" y1="300" x2="325" y2="300" class="tower-section"/>
  
  <!-- Internal ladder (left side) -->
  <g transform="translate(225, 220)">
    <!-- Ladder rails -->
    <line x1="0" y1="0" x2="8" y2="450" class="ladder"/>
    <line x1="15" y1="0" x2="23" y2="450" class="ladder"/>
    <!-- Rungs -->
    <g class="ladder-rung">
      <line x1="1" y1="20" x2="22" y2="21"/>
      <line x1="1" y1="50" x2="22" y2="52"/>
      <line x1="2" y1="80" x2="22" y2="83"/>
      <line x1="2" y1="110" x2="23" y2="114"/>
      <line x1="2" y1="140" x2="23" y2="145"/>
      <line x1="3" y1="170" x2="23" y2="176"/>
      <line x1="3" y1="200" x2="24" y2="207"/>
      <line x1="3" y1="230" x2="24" y2="238"/>
      <line x1="4" y1="260" x2="24" y2="269"/>
      <line x1="4" y1="290" x2="25" y2="300"/>
      <line x1="4" y1="320" x2="25" y2="331"/>
      <line x1="5" y1="350" x2="25" y2="362"/>
      <line x1="5" y1="380" x2="26" y2="393"/>
      <line x1="6" y1="410" x2="26" y2="424"/>
      <line x1="6" y1="440" x2="27" y2="455"/>
    </g>
  </g>
  
  <!-- Elevator shaft (right side) -->
  <rect x="280" y="230" width="25" height="430" rx="2" class="elevator"/>
  <text class="tl" x="292" y="450" text-anchor="middle" transform="rotate(-90, 292, 450)" fill="#185FA5">ELEVATOR</text>
  
  <!-- Electrical cables running down -->
  <path d="M270 220 C270 300 268 400 268 500 C268 600 268 650 310 665 L470 665" class="cable"/>
  <path d="M260 225 C258 350 256 500 256 600 C256 650 256 670 256 680" class="cable-thin"/>
  
  <!-- Tower labels -->
  <line x1="340" y1="350" x2="400" y2="320" class="leader"/>
  <text class="ts" x="405" y="316">Tubular steel tower</text>
  <text class="tl" x="405" y="328">80-120m height typical</text>
  <text class="tl" x="405" y="340">Tapered for strength</text>
  
  <line x1="248" y1="400" x2="130" y2="380" class="leader"/>
  <text class="ts" x="125" y="376" text-anchor="end">Internal ladder</text>
  <text class="tl" x="125" y="388" text-anchor="end">Service access</text>
  
  <line x1="305" y1="500" x2="400" y2="520" class="leader"/>
  <text class="ts" x="405" y="516">Service elevator</text>
  
  <line x1="268" y1="580" x2="130" y2="600" class="leader"/>
  <text class="ts" x="125" y="596" text-anchor="end">Power cables</text>
  <text class="tl" x="125" y="608" text-anchor="end">To transformer</text>

  <!-- ===== NACELLE ===== -->
  
  <g transform="translate(270, 160)">
    <!-- Nacelle base/bedplate -->
    <rect x="-60" y="30" width="120" height="15" class="nacelle"/>
    
    <!-- Yaw bearing -->
    <ellipse cx="0" cy="42" rx="35" ry="6" class="bearing"/>
    
    <!-- Yaw motors -->
    <rect x="-55" y="32" width="12" height="18" rx="2" class="yaw"/>
    <rect x="43" y="32" width="12" height="18" rx="2" class="yaw"/>
    
    <!-- Nacelle housing -->
    <path d="M-65 30 L-70 -10 L-65 -35 L70 -35 L85 -10 L85 30 Z" class="nacelle-cover"/>
    
    <!-- Main shaft -->
    <rect x="-90" y="-8" width="35" height="16" rx="2" fill="#888780" stroke="#5F5E5A" stroke-width="0.5"/>
    
    <!-- Gearbox -->
    <rect x="-55" y="-25" width="40" height="45" rx="3" class="gearbox"/>
    <text class="tl" x="-35" y="5" text-anchor="middle" fill="#633806">GEAR</text>
    
    <!-- Generator -->
    <rect x="-10" y="-20" width="50" height="38" rx="4" class="generator"/>
    <ellipse cx="15" cy="0" rx="15" ry="15" fill="none" stroke="#0C447C" stroke-width="1"/>
    <text class="tl" x="15" y="4" text-anchor="middle" fill="#E6F1FB">GEN</text>
    
    <!-- Brake disc -->
    <rect x="45" y="-12" width="8" height="24" rx="1" class="brake"/>
    
    <!-- Electrical cabinet -->
    <rect x="58" y="-25" width="20" height="35" rx="2" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
    
    <!-- Anemometer on top -->
    <line x1="60" y1="-35" x2="60" y2="-50" stroke="#5F5E5A" stroke-width="1"/>
    <ellipse cx="60" cy="-52" rx="8" ry="3" fill="#D3D1C7" stroke="#888780" stroke-width="0.5"/>
  </g>
  
  <!-- Nacelle labels -->
  <line x1="215" y1="135" x2="130" y2="115" class="leader"/>
  <text class="ts" x="125" y="111" text-anchor="end">Gearbox</text>
  <text class="tl" x="125" y="123" text-anchor="end">Speed multiplier</text>
  
  <line x1="285" y1="145" x2="400" y2="125" class="leader"/>
  <text class="ts" x="405" y="121">Generator</text>
  <text class="tl" x="405" y="133">Converts rotation to electricity</text>
  
  <line x1="315" y1="155" x2="400" y2="165" class="leader"/>
  <text class="ts" x="405" y="161">Brake system</text>
  
  <line x1="215" y1="200" x2="130" y2="220" class="leader"/>
  <text class="ts" x="125" y="216" text-anchor="end">Yaw motors</text>
  <text class="tl" x="125" y="228" text-anchor="end">Rotate nacelle to face wind</text>
  
  <line x1="330" y1="108" x2="400" y2="90" class="leader"/>
  <text class="ts" x="405" y="86">Anemometer</text>
  <text class="tl" x="405" y="98">Wind speed sensor</text>

  <!-- ===== ROTOR HUB & BLADES ===== -->
  
  <!-- Hub -->
  <g transform="translate(180, 152)">
    <!-- Hub body -->
    <ellipse cx="0" cy="0" rx="25" ry="30" class="hub"/>
    <!-- Hub nose cone -->
    <path d="M-25 -20 Q-50 0 -25 20 Q-30 0 -25 -20" class="hub-cap"/>
    
    <!-- Blade roots with pitch motors -->
    <!-- Blade 1 (up) -->
    <g transform="translate(-10, -25) rotate(-80)">
      <ellipse cx="0" cy="0" rx="12" ry="8" class="blade-root"/>
      <rect x="-8" y="-5" width="10" height="10" rx="2" class="pitch-motor"/>
    </g>
    
    <!-- Blade 2 (lower left) -->
    <g transform="translate(-18, 18) rotate(40)">
      <ellipse cx="0" cy="0" rx="12" ry="8" class="blade-root"/>
      <rect x="-8" y="-5" width="10" height="10" rx="2" class="pitch-motor"/>
    </g>
    
    <!-- Blade 3 (lower right) -->
    <g transform="translate(5, 22) rotate(160)">
      <ellipse cx="0" cy="0" rx="12" ry="8" class="blade-root"/>
      <rect x="-8" y="-5" width="10" height="10" rx="2" class="pitch-motor"/>
    </g>
  </g>
  
  <!-- Blade 1 (pointing up-left) -->
  <path d="M165 125 Q140 80 130 40 Q125 20 115 15 Q110 18 112 25 Q115 50 125 90 Q140 120 158 128 Z" class="blade" fill="url(#bladeGrad)"/>
  
  <!-- Blade 2 (pointing down-left) -->
  <path d="M158 175 Q120 200 80 230 Q60 245 55 255 Q60 258 68 252 Q95 235 130 210 Q155 190 163 178 Z" class="blade" fill="url(#bladeGrad)"/>
  
  <!-- Blade 3 (pointing down-right, partially visible) -->
  <path d="M188 175 Q195 200 205 230 Q210 250 215 255 Q220 252 218 245 Q212 220 202 195 Q192 175 186 172 Z" class="blade" fill="url(#bladeGrad)"/>
  
  <!-- Blade labels -->
  <line x1="115" y1="35" x2="60" y2="35" class="leader"/>
  <text class="ts" x="55" y="31" text-anchor="end">Composite blade</text>
  <text class="tl" x="55" y="43" text-anchor="end">Fiberglass/carbon fiber</text>
  <text class="tl" x="55" y="55" text-anchor="end">40-80m length each</text>
  
  <line x1="170" y1="130" x2="130" y2="155" class="leader"/>
  <text class="ts" x="85" y="151" text-anchor="end">Pitch motor</text>
  <text class="tl" x="85" y="163" text-anchor="end">Adjusts blade angle</text>
  
  <line x1="180" y1="152" x2="130" y2="180" class="leader"/>
  <text class="ts" x="85" y="183" text-anchor="end">Rotor hub</text>

  <!-- ===== LEGEND ===== -->
  <g transform="translate(40, 895)">
    <rect x="0" y="-15" width="600" height="30" rx="4" fill="none" stroke="#D3D1C7" stroke-width="0.5"/>
    
    <rect x="15" y="-5" width="12" height="12" rx="2" class="generator"/>
    <text class="tl" x="32" y="5">Generator</text>
    
    <rect x="95" y="-5" width="12" height="12" rx="2" class="gearbox"/>
    <text class="tl" x="112" y="5">Gearbox</text>
    
    <rect x="170" y="-5" width="12" height="12" rx="2" class="brake"/>
    <text class="tl" x="187" y="5">Brake</text>
    
    <rect x="230" y="-5" width="12" height="12" rx="2" class="yaw"/>
    <text class="tl" x="247" y="5">Yaw system</text>
    
    <rect x="320" y="-5" width="12" height="12" rx="2" class="pitch-motor"/>
    <text class="tl" x="337" y="5">Pitch motor</text>
    
    <line x1="415" y1="1" x2="435" y2="1" class="cable" style="stroke-width:2"/>
    <text class="tl" x="440" y="5">Power cable</text>
    
    <rect x="515" y="-5" width="12" height="12" rx="2" class="transformer"/>
    <text class="tl" x="532" y="5">Transformer</text>
  </g>

</svg>
```

## CSS Classes

```css
/* Foundation */
.concrete { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.concrete-dark { fill: #888780; stroke: #5F5E5A; stroke-width: 1; }
.rebar { stroke: #854F0B; stroke-width: 1.5; fill: none; }
.soil { fill: #8B7355; stroke: #5F5E5A; stroke-width: 0.5; }
.soil-dark { fill: #6B5344; }

/* Tower */
.tower { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.tower-inner { fill: #D3D1C7; stroke: #888780; stroke-width: 0.5; }
.tower-section { stroke: #888780; stroke-width: 0.5; stroke-dasharray: 2 4; }
.ladder { stroke: #5F5E5A; stroke-width: 1; fill: none; }
.ladder-rung { stroke: #888780; stroke-width: 0.8; }
.elevator { fill: #E6F1FB; stroke: #185FA5; stroke-width: 0.5; }
.cable { stroke: #E24B4A; stroke-width: 2; fill: none; }
.cable-thin { stroke: #E24B4A; stroke-width: 1.5; fill: none; }

/* Nacelle */
.nacelle { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.nacelle-cover { fill: #D3D1C7; stroke: #5F5E5A; stroke-width: 1; }
.gearbox { fill: #BA7517; stroke: #633806; stroke-width: 0.5; }
.generator { fill: #378ADD; stroke: #0C447C; stroke-width: 0.5; }
.brake { fill: #E24B4A; stroke: #791F1F; stroke-width: 0.5; }
.yaw { fill: #5DCAA5; stroke: #085041; stroke-width: 0.5; }
.bearing { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }

/* Rotor */
.hub { fill: #D3D1C7; stroke: #5F5E5A; stroke-width: 1; }
.hub-cap { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.blade { fill: #F1EFE8; stroke: #888780; stroke-width: 1; }
.blade-root { fill: #D3D1C7; stroke: #5F5E5A; stroke-width: 0.5; }
.pitch-motor { fill: #7F77DD; stroke: #3C3489; stroke-width: 0.5; }

/* Transformer */
.transformer { fill: #27500A; stroke: #173404; stroke-width: 1; }
.transformer-fin { fill: #3B6D11; stroke: #27500A; stroke-width: 0.5; }
```
