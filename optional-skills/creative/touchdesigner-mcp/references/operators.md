# TouchDesigner Operator Reference

## Operator Families Overview

TouchDesigner has 6 operator families. Each family processes a specific data type and is color-coded in the UI. Operators can only connect to others of the SAME family (with cross-family converters as the bridge).

## TOPs — Texture Operators (Purple)

2D image/texture processing on the GPU. The workhorse of visual output.

### Generators (create images from nothing)

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Noise TOP | `noiseTop` | `type` (0-6), `monochrome`, `seed`, `period`, `harmonics`, `exponent`, `amp`, `offset`, `resolutionw/h` | Procedural noise textures — Perlin, Simplex, Sparse, etc. Foundation of generative art. |
| Constant TOP | `constantTop` | `colorr/g/b/a`, `resolutionw/h` | Solid color. Use as background or blend input. |
| Text TOP | `textTop` | `text`, `fontsizex`, `fontfile`, `alignx/y`, `colorr/g/b` | Render text to texture. Supports multi-line, word wrap. |
| Ramp TOP | `rampTop` | `type` (0=horizontal, 1=vertical, 2=radial, 3=circular), `phase`, `period` | Gradient textures for masking, color mapping. |
| Circle TOP | `circleTop` | `radiusx/y`, `centerx/y`, `width` | Circles, rings, ellipses. |
| Rectangle TOP | `rectangleTop` | `sizex/y`, `centerx/y`, `softness` | Rectangles with optional softness. |
| GLSL TOP | `glslTop` | `dat` (points to shader DAT), `resolutionw/h`, `outputformat`, custom uniforms | Custom fragment shaders. Most powerful TOP for custom visuals. |
| GLSL Multi TOP | `glslmultiTop` | `dat`, `numinputs`, `numoutputs`, `numcomputepasses` | Multi-pass GLSL with compute shaders. Advanced. |
| Render TOP | `renderTop` | `camera`, `geometry`, `lights`, `resolutionw/h` | Renders 3D scenes (SOPs + MATs + Camera/Light COMPs). |

### Filters (modify a single input)

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Level TOP | `levelTop` | `opacity`, `brightness1/2`, `gamma1/2`, `contrast`, `invert`, `blacklevel/whitelevel` | Brightness, contrast, gamma, levels. Essential color correction. |
| Blur TOP | `blurTop` | `sizex/y`, `type` (0=Gaussian, 1=Box, 2=Bartlett) | Gaussian/box blur. |
| Transform TOP | `transformTop` | `tx/ty`, `sx/sy`, `rz`, `pivotx/y`, `extend` (0=Hold, 1=Zero, 2=Repeat, 3=Mirror) | Translate, scale, rotate textures. |
| HSV Adjust TOP | `hsvadjustTop` | `hueoffset`, `saturationmult`, `valuemult` | HSV color adjustments. |
| Lookup TOP | `lookupTop` | (input: texture + lookup table) | Color remapping via lookup table texture. |
| Edge TOP | `edgeTop` | `type` (0=Sobel, 1=Frei-Chen) | Edge detection. |
| Displace TOP | `displaceTop` | `scalex/y` | Pixel displacement using a second input as displacement map. |
| Flip TOP | `flipTop` | `flipx`, `flipy`, `flop` (diagonal) | Mirror/flip textures. |
| Crop TOP | `cropTop` | `cropleft/right/top/bottom` | Crop region of texture. |
| Resolution TOP | `resolutionTop` | `resolutionw/h`, `outputresolution` | Resize textures. |
| Null TOP | `nullTop` | (none significant) | Pass-through. Use for organization, referencing, feedback delay. |
| Cache TOP | `cacheTop` | `length`, `step` | Store N frames of history. Useful for trails, time effects. |

### Compositors (combine multiple inputs)

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Composite TOP | `compositeTop` | `operand` (0-31: Over, Add, Multiply, Screen, etc.) | Blend two textures with standard compositing modes. |
| Over TOP | `overTop` | (simple alpha compositing) | Layer with alpha. Simpler than Composite. |
| Add TOP | `addTop` | (additive blend) | Additive blending. Great for glow, light effects. |
| Multiply TOP | `multiplyTop` | (multiplicative blend) | Multiply blend. Good for masking, darkening. |
| Switch TOP | `switchTop` | `index` (0-based) | Switch between multiple inputs by index. |
| Cross TOP | `crossTop` | `cross` (0.0-1.0) | Crossfade between two inputs. |

### I/O (input/output)

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Movie File In TOP | `moviefileinTop` | `file`, `speed`, `trim`, `index` | Load video files, image sequences. |
| Movie File Out TOP | `moviefileoutTop` | `file`, `type` (codec), `record` (toggle) | Record/export video files. |
| NDI In TOP | `ndiinTop` | `sourcename` | Receive NDI video streams. |
| NDI Out TOP | `ndioutTop` | `sourcename` | Send NDI video streams. |
| Syphon Spout In/Out TOP | `syphonspoutinTop` / `syphonspoutoutTop` | `servername` | Inter-app texture sharing. |
| Video Device In TOP | `videodeviceinTop` | `device` | Webcam/capture card input. |
| Feedback TOP | `feedbackTop` | `top` (path to the TOP to feed back) | One-frame delay feedback. Essential for recursive effects. |

### Converters

| Operator | Type Name | Direction | Use |
|----------|-----------|-----------|-----|
| CHOP to TOP | `choptopTop` | CHOP -> TOP | Visualize channel data as texture (waveform, spectrum display). |
| TOP to CHOP | `topchopChop` | TOP -> CHOP | Sample texture pixels as channel data. |

## CHOPs — Channel Operators (Green)

Time-varying numeric data: audio, animation curves, sensor data, control signals.

### Generators

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Constant CHOP | `constantChop` | `name0/value0`, `name1/value1`... | Static named channels. Control panel for parameters. |
| LFO CHOP | `lfoChop` | `frequency`, `type` (0=Sin, 1=Tri, 2=Square, 3=Ramp, 4=Pulse), `amp`, `offset`, `phase` | Low frequency oscillator. Animation driver. |
| Noise CHOP | `noiseChop` | `type`, `roughness`, `period`, `amp`, `seed`, `channels` | Smooth random motion. Organic animation. |
| Pattern CHOP | `patternChop` | `type` (0=Sine, 1=Triangle, ...), `length`, `cycles` | Generate waveform patterns. |
| Timer CHOP | `timerChop` | `length`, `play`, `cue`, `cycles` | Countdown/count-up timer with cue points. |
| Count CHOP | `countChop` | `threshold`, `limittype`, `limitmin/max` | Event counter with wrapping/clamping. |

### Audio

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Audio File In CHOP | `audiofileinChop` | `file`, `volume`, `play`, `speed`, `trim` | Play audio files. |
| Audio Device In CHOP | `audiodeviceinChop` | `device`, `channels` | Live microphone/line input. |
| Audio Spectrum CHOP | `audiospectrumChop` | `size` (FFT size), `outputformat` (0=Power, 1=Magnitude) | FFT frequency analysis. |
| Audio Band EQ CHOP | `audiobandeqChop` | `bands`, `gaindb` per band | Frequency band isolation. |
| Audio Device Out CHOP | `audiodeviceoutChop` | `device` | Audio playback output. |

### Math/Logic

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Math CHOP | `mathChop` | `preoff`, `gain`, `postoff`, `chanop` (0=Off, 1=Add, 2=Subtract, 3=Multiply...) | Math operations on channels. The Swiss army knife. |
| Logic CHOP | `logicChop` | `preop` (0=Off, 1=AND, 2=OR, 3=XOR, 4=NAND), `convert` | Boolean logic on channels. |
| Filter CHOP | `filterChop` | `type` (0=Low Pass, 1=Band Pass, 2=High Pass, 3=Notch), `cutofffreq`, `filterwidth` | Smooth, dampen, filter signals. |
| Lag CHOP | `lagChop` | `lag1/2`, `overshoot1/2` | Smooth transitions with overshoot. |
| Limit CHOP | `limitChop` | `type` (0=Clamp, 1=Loop, 2=ZigZag), `min/max` | Clamp or wrap channel values. |
| Speed CHOP | `speedChop` | (none significant) | Integrate values (velocity to position, acceleration to velocity). |
| Trigger CHOP | `triggerChop` | `attack`, `peak`, `decay`, `sustain`, `release` | ADSR envelope from trigger events. |
| Select CHOP | `selectChop` | `chop` (path), `channames` | Reference channels from another CHOP. |
| Merge CHOP | `mergeChop` | `align` (0=Extend, 1=Trim to First, 2=Trim to Shortest) | Combine channels from multiple CHOPs. |
| Null CHOP | `nullChop` | (none significant) | Pass-through for organization and referencing. |

### Input Devices

| Operator | Type Name | Use |
|----------|-----------|-----|
| Mouse In CHOP | `mouseinChop` | Mouse position, buttons, wheel. |
| Keyboard In CHOP | `keyboardinChop` | Keyboard key states. |
| MIDI In CHOP | `midiinChop` | MIDI note/CC input. |
| OSC In CHOP | `oscinChop` | OSC message input (network). |

## SOPs — Surface Operators (Blue)

3D geometry: points, polygons, NURBS, meshes.

### Generators

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Grid SOP | `gridSop` | `rows`, `cols`, `sizex/y`, `type` (0=Polygon, 1=Mesh, 2=NURBS) | Flat grid mesh. Foundation for displacement, instancing. |
| Sphere SOP | `sphereSop` | `type`, `rows`, `cols`, `radius` | Sphere geometry. |
| Box SOP | `boxSop` | `sizex/y/z` | Box geometry. |
| Torus SOP | `torusSop` | `radiusx/y`, `rows`, `cols` | Donut shape. |
| Circle SOP | `circleSop` | `type`, `radius`, `divs` | Circle/ring geometry. |
| Line SOP | `lineSop` | `dist`, `points` | Line segments. |
| Text SOP | `textSop` | `text`, `fontsizex`, `fontfile`, `extrude` | 3D text geometry. |

### Modifiers

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Transform SOP | `transformSop` | `tx/ty/tz`, `rx/ry/rz`, `sx/sy/sz` | Transform geometry (translate, rotate, scale). |
| Noise SOP | `noiseSop` | `type`, `amp`, `period`, `roughness` | Deform geometry with noise. |
| Sort SOP | `sortSop` | `ptsort`, `primsort` | Reorder points/primitives. |
| Facet SOP | `facetSop` | `unique`, `consolidate`, `computenormals` | Normals, consolidation, unique points. |
| Merge SOP | `mergeSop` | (none significant) | Combine multiple geometry inputs. |
| Null SOP | `nullSop` | (none significant) | Pass-through. |

## DATs — Data Operators (White)

Text, tables, scripts, network data.

### Core

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Table DAT | `tableDat` | (edit content directly) | Spreadsheet-like data tables. |
| Text DAT | `textDat` | (edit content directly) | Arbitrary text content. Shader code, configs, scripts. |
| Script DAT | `scriptDat` | `language` (0=Python, 1=C++) | Custom callbacks and DAT processing. |
| CHOP Execute DAT | `chopexecDat` | `chop` (path to watch), callbacks | Trigger Python on CHOP value changes. |
| DAT Execute DAT | `datexecDat` | `dat` (path to watch) | Trigger Python on DAT content changes. |
| Panel Execute DAT | `panelexecDat` | `panel` | Trigger Python on UI panel events. |

### I/O

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Web DAT | `webDat` | `url`, `fetchmethod` (0=GET, 1=POST) | HTTP requests. API integration. |
| TCP/IP DAT | `tcpipDat` | `address`, `port`, `mode` | TCP networking. |
| OSC In DAT | `oscinDat` | `port` | Receive OSC as text messages. |
| Serial DAT | `serialDat` | `port`, `baudrate` | Serial port communication (Arduino, etc.). |
| File In DAT | `fileinDat` | `file` | Read text files. |
| File Out DAT | `fileoutDat` | `file`, `write` | Write text files. |

### Conversions

| Operator | Type Name | Direction | Use |
|----------|-----------|-----------|-----|
| DAT to CHOP | `dattochopChop` | DAT -> CHOP | Convert table data to channels. |
| CHOP to DAT | `choptodatDat` | CHOP -> DAT | Convert channel data to table rows. |
| SOP to DAT | `soptodatDat` | SOP -> DAT | Extract geometry data as table. |

## MATs — Material Operators (Yellow)

Materials for 3D rendering in Render TOP / Geometry COMP.

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Phong MAT | `phongMat` | `diff_colorr/g/b`, `spec_colorr/g/b`, `shininess`, `colormap`, `normalmap` | Classic Phong shading. Simple, fast. |
| PBR MAT | `pbrMat` | `basecolorr/g/b`, `metallic`, `roughness`, `normalmap`, `emitcolorr/g/b` | Physically-based rendering. Realistic materials. |
| GLSL MAT | `glslMat` | `dat` (shader DAT), custom uniforms | Custom vertex + fragment shaders for 3D. |
| Constant MAT | `constMat` | `colorr/g/b`, `colormap` | Flat unlit color/texture. No shading. |
| Point Sprite MAT | `pointspriteMat` | `colormap`, `scale` | Render points as camera-facing sprites. Great for particles. |
| Wireframe MAT | `wireframeMat` | `colorr/g/b`, `width` | Wireframe rendering. |
| Depth MAT | `depthMat` | `near`, `far` | Render depth buffer as grayscale. |

## COMPs — Component Operators (Gray)

Containers, 3D scene elements, UI components.

### 3D Scene

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Geometry COMP | `geometryComp` | `material` (path), `instancechop` (path), `instancing` (toggle) | Renders geometry with material. Instancing host. |
| Camera COMP | `cameraComp` | `tx/ty/tz`, `rx/ry/rz`, `fov`, `near/far` | Camera for Render TOP. |
| Light COMP | `lightComp` | `lighttype` (0=Point, 1=Directional, 2=Spot, 3=Cone), `dimmer`, `colorr/g/b` | Lighting for 3D scenes. |
| Ambient Light COMP | `ambientlightComp` | `dimmer`, `colorr/g/b` | Ambient lighting. |
| Environment Light COMP | `envlightComp` | `envmap` | Image-based lighting (IBL). |

### Containers

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Container COMP | `containerComp` | `w`, `h`, `bgcolor1/2/3` | UI container. Holds other COMPs for panel layouts. |
| Base COMP | `baseComp` | (none significant) | Generic container. Networks-inside-networks. |
| Replicator COMP | `replicatorComp` | `template`, `operatorsdat` | Clone a template operator N times from a table. |

### Utilities

| Operator | Type Name | Key Parameters | Use |
|----------|-----------|---------------|-----|
| Window COMP | `windowComp` | `winw/h`, `winoffsetx/y`, `monitor`, `borders` | Output window for display/projection. |
| Select COMP | `selectComp` | `rowcol`, `panel` | Select and display content from elsewhere. |
| Engine COMP | `engineComp` | `tox`, `externaltox` | Load external .tox components. Sub-process isolation. |

## Cross-Family Converter Summary

| From | To | Operator | Type Name |
|------|-----|----------|-----------|
| CHOP | TOP | CHOP to TOP | `choptopTop` |
| TOP | CHOP | TOP to CHOP | `topchopChop` |
| DAT | CHOP | DAT to CHOP | `dattochopChop` |
| CHOP | DAT | CHOP to DAT | `choptodatDat` |
| SOP | CHOP | SOP to CHOP | `soptochopChop` |
| CHOP | SOP | CHOP to SOP | `choptosopSop` |
| SOP | DAT | SOP to DAT | `soptodatDat` |
| DAT | SOP | DAT to SOP | `dattosopSop` |
| SOP | TOP | (use Render TOP + Geometry COMP) | — |
| TOP | SOP | TOP to SOP | `toptosopSop` |
