# Blender-MCP Installation and Troubleshooting

## Server Install

### Via uv (recommended)
```bash
uvx blender-mcp
```

### Via pip
```bash
pip install blender-mcp
```

### Via source
```bash
git clone https://github.com/ahujasid/blender-mcp.git
cd blender-mcp
pip install -e .
```

## Addon Install

### Manual Copy
```bash
mkdir -p ~/.config/blender/4.3/scripts/addons
cp /path/to/blender-mcp/addon.py ~/.config/blender/4.3/scripts/addons/
```

### Via Blender UI
1. Open Blender
2. **Edit -> Preferences -> Add-ons -> Install**
3. Select `addon.py`
4. Check "Interface: Blender MCP"

### Connect
1. In 3D View, press N to open sidebar
2. Click "Connect to Claude"
3. The addon starts a socket server (default localhost:9876)

## Hermes Config

```yaml
mcp_servers:
  blender:
    command: uvx
    args: [blender-mcp]
```

Restart Hermes after adding.

## Common Issues

| Issue | Fix |
|-------|-----|
| Addon greyed out, cannot check | Restart Blender -- cache stale after manual copy |
| `No module named 'numpy'` | `sudo pip install numpy --break-system-packages` (Blender uses system Python) |
| glTF export fails silently | Same as above -- numpy required for glTF I/O |
| Connection refused | Ensure addon is enabled and "Connect to Claude" clicked |
| Timeout on complex requests | Break into smaller steps, or increase MCP timeout |
| `bsdf.inputs['Emission']` error | Blender 4.x renamed to `Emission Color` / `Emission Strength` |
| `ShaderNodeTexGrid` missing | Use `ShaderNodeTexWave` with `wave_type='BANDS'` |
| Undo wipes scene | Each MCP call is a separate undo context; save before risky ops |

## Capabilities

- Scene inspection (objects, materials, lighting)
- Object creation/deletion/modification
- Material control (Principled BSDF, emission, transparency)
- Python execution in Blender context
- Poly Haven asset download (HDRIs, textures, models)
- Hyper3D Rodin 3D generation
- Viewport screenshots
- GLB export
