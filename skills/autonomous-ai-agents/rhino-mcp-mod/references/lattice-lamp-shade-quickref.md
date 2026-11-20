# Lattice Lamp Shade - Quick Reference Card

**Source:** YouTube "I Bypassed the Slicer to Print mid Air" (qne2eZXdWoc)  
**Topic:** Direct G-code from Grasshopper for mid-air lattice printing

---

## Grasshopper Components for G-code Generation

### Core Geometry Pipeline
```
Surface → Populate3D → Voronoi3D → MeshEdges → Pipe → CurveToPolyline → GCodeWriter
```

### Essential Components

| Component | Input | Output | Key Settings |
|-----------|-------|--------|--------------|
| `Populate3D` | BoundingBox, Count | Points | Seed, Distribution |
| `Voronoi3D` | Points, Bounds | Cells | Radius, Padding |
| `MeshEdges` | Mesh | Curves | Naked/Interior edges |
| `Pipe` | Curves, Radius | Meshes/Surfaces | Cap, Radius profile |
| `CurveToPolyline` | Curves, Tolerance | Polylines | Max segment length |
| `GCodeWriter` | Polylines, Settings | .gcode file | See settings below |

---

## GCodeWriter Settings Template

```json
{
  "machine": {
    "nozzle_diameter": 0.4,
    "filament_diameter": 1.75,
    "bed_size": [250, 250, 300]
  },
  "material": {
    "type": "PLA",
    "hotend_temp": 200,
    "bed_temp": 60,
    "fan_speed": 255,
    "retraction_distance": 2.0,
    "retraction_speed": 25,
    "extrusion_multiplier": 1.0
  },
  "print": {
    "xy_speed": 1200,
    "z_speed": 300,
    "first_layer_speed": 600,
    "bridge_speed": 600,
    "travel_speed": 9000
  },
  "geometry": {
    "segment_length": 0.8,
    "min_feature_size": 0.4,
    "bridge_angle_threshold": 45
  }
}
```

---

## Mid-Air Printing Checklist

- [ ] Fan at 100% (M106 S255)
- [ ] Hotend temp ≤ 205°C for PLA
- [ ] Print speed 15-25 mm/s for bridges
- [ ] Retraction 1.5-2.5mm @ 25mm/s
- [ ] Z-hop OFF for mid-air segments
- [ ] Combing: "Within Infill" or "All"
- [ ] Path order: Bridges → Perimeters → Infill
- [ ] Cooling: Part cooling aimed at bridge zone

---

## Rhino/GH Commands Quick Reference

```python
# In GhPython component
import Rhino.Geometry as rg

# Curve to polyline (for G-code)
polyline = curve.ToPolyline(
    0.8,    # max segment length
    0.8,    # min segment length  
    0.01,   # max angle deviation
    15      # max segments per curve
)

# Extrusion calculation
nozzle = 0.4
filament = 1.75
multiplier = 1.0
length = segment.Length
vol = 3.14159 * (nozzle/2)**2 * length * multiplier
e_inc = vol / (3.14159 * (filament/2)**2)

# G-code line
gcode = f"G1 X{pt.X:.3f} Y{pt.Y:.3f} Z{pt.Z:.3f} E{e_total:.5f} F{feedrate}"
```

---

## File Structure for Project

```
/project
├── gh/
│   ├── lattice_lamp.gh          # Main definition
│   ├── components/
│   │   ├── GCodeWriter.ghuser   # Custom component
│   │   └── PathOptimizer.ghuser
│   └── data/
│       ├── settings.json        # Machine/material settings
│       └── toolpaths.json       # Intermediate storage
├── gcode/
│   └── lamp_shade_v1.gcode      # Output
├── stl/
│   └── lamp_shade_ref.stl       # Reference geometry
└── docs/
    └── print_log.md             # Print parameters & results
```

---

## Useful Rhino/GH Plugins for This Workflow

| Plugin | Purpose |
|--------|---------|
| **Silkworm** | G-code generation from GH |
| **Firefly** | Hardware/Arduino integration |
| **Kangaroo2** | Form-finding for lamp form |
| **Karamba3D** | Structural analysis of lattice |
| **NGon** | Mesh topology for clean lattices |
| **Pufferfish** | Curve/tween utilities |
| **Heteroptera** | Graph/pattern tools |

---

## Video Timestamp Notes (qne2eZXdWoc)

| Time | Topic |
|------|-------|
| 3:51 | Introduction - why bypass slicer |
| 5:23 | Grasshopper lattice generation |
| 8:45 | Mid-air printing theory |
| 12:10 | G-code writer component |
| 15:30 | Path ordering optimization |
| 18:22 | First test print |
| 22:45 | Lamp shade parameters |
| 26:10 | Cooling & retraction tuning |
| 31:00 | Final lamp shade result |

---

## Troubleshooting Quick Fixes

| Symptom | Fix |
|---------|-----|
| Stringy bridges | ↓ Temp 5°C, ↑ Retraction 0.5mm |
| Sagging mid-air | ↑ Fan, ↓ Speed 5mm/s, ↓ Temp |
| Nozzle clogs | Clean nozzle, check extruder gear |
| G-code too large | ↑ Segment length to 1.0-1.5mm |
| Layer shifts | Check belt tension, ↓ Travel speed |

---

*Generated from video analysis - add to Rhino/GH knowledge base*