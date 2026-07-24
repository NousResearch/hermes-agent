#!/usr/bin/env python3
"""Quality gates for the game-asset pipeline.

Each gate is a function returning (ok: bool, message: str). Used by
trellis_gen.py, skintokens_rig.py, and the game-asset-pipeline orchestrator.
No third-party deps beyond what the pipeline already requires (Pillow optional
for image gates; GLB parsing is done with stdlib struct/json).
"""
import json
import struct
from pathlib import Path


def gate_alpha_mask(path, min_alpha_coverage=0.05, max_alpha_coverage=0.98):
    """Image has a real alpha channel isolating a subject (not all opaque/transparent)."""
    try:
        from PIL import Image
    except ImportError:
        return True, "Pillow not installed; alpha gate skipped"
    try:
        im = Image.open(path).convert("RGBA")
        a = im.getchannel("A")
        hist = a.histogram()
        total = im.width * im.height
        # count pixels above half-alpha as subject: rembg/u2net produces
        # semi-transparent masks where little is fully opaque
        subject = sum(hist[128:]) / total
        if subject < min_alpha_coverage:
            return False, f"subject coverage {subject:.1%} too low — bad mask or empty image?"
        if subject > max_alpha_coverage:
            return False, f"opaque coverage {subject:.1%} — image is NOT alpha-masked (would bake background into 3D)"
        return True, f"alpha mask ok ({subject:.0%} subject coverage)"
    except Exception as e:
        return False, f"alpha gate error: {e}"


def _parse_glb_json(path):
    with open(path, "rb") as f:
        magic, version, _length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError("not a GLB (bad magic)")
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # JSON
            raise ValueError("first chunk is not JSON")
        return json.loads(f.read(chunk_len))


def gate_glb_geometry(path, min_verts=1000, max_verts=2_000_000):
    """GLB parses, has meshes with a sane vertex count and at least one material/texture."""
    try:
        g = _parse_glb_json(path)
        meshes = g.get("meshes", [])
        if not meshes:
            return False, "no meshes in GLB"
        verts = 0
        for m in meshes:
            for prim in m.get("primitives", []):
                idx = prim.get("attributes", {}).get("POSITION")
                if idx is not None:
                    verts += g["accessors"][idx].get("count", 0)
        if verts < min_verts:
            return False, f"only {verts} verts — generation likely failed"
        if verts > max_verts:
            return False, f"{verts} verts — decimation target not applied?"
        has_tex = bool(g.get("images") or g.get("textures"))
        note = "textured" if has_tex else "NO TEXTURES"
        return (has_tex, f"{verts} verts, {note}") if not has_tex else (True, f"{verts} verts, textured")
    except Exception as e:
        return False, f"glb gate error: {e}"


def gate_glb_rig(path, min_bones=5):
    """GLB contains a skin with joints and skin weights (JOINTS_0/WEIGHTS_0)."""
    try:
        g = _parse_glb_json(path)
        skins = g.get("skins", [])
        if not skins:
            return False, "no skins in GLB — not rigged"
        joints = max(len(s.get("joints", [])) for s in skins)
        if joints < min_bones:
            return False, f"only {joints} joints — rig looks degenerate"
        has_weights = any(
            "JOINTS_0" in prim.get("attributes", {}) and "WEIGHTS_0" in prim.get("attributes", {})
            for m in g.get("meshes", []) for prim in m.get("primitives", [])
        )
        if not has_weights:
            return False, "skin exists but no JOINTS_0/WEIGHTS_0 attributes"
        return True, f"rigged: {joints} joints, weights present"
    except Exception as e:
        return False, f"rig gate error: {e}"


def gate_full_body_for_rig(path):
    """DEPRECATED — bounding-box proportions cannot distinguish cropped humanoids
    from legitimately wide creatures (ants/spiders). Verified: waist-up wizard
    (fails rigging) and fire ant (rigs fine) have identical 1.0 ratios.
    Kept as a no-op so existing callers don't break; rely on the auto-voxel
    retry in skintokens_rig.py instead. Real defense: require full-body source
    images when rigging is intended (documented in the skills)."""
    return True, "proportion gate disabled (unreliable heuristic)"


ENGINE_PRESETS = {
    # target: (decimation, texture_size)
    # NOTE: TRELLIS.2 /extract_glb enforces decimation >= 100000
    "godot-mobile": (100000, 1024),
    "godot": (100000, 2048),
    "unity": (100000, 2048),
    "blender": (300000, 2048),
    "film": (500000, 4096),
}


def apply_engine_preset(args):
    """Mutate an argparse namespace with --target presets, unless flags were explicitly set."""
    target = getattr(args, "target", None)
    if not target:
        return args
    if target not in ENGINE_PRESETS:
        raise SystemExit(f"unknown --target '{target}'. Choices: {', '.join(ENGINE_PRESETS)}")
    dec, tex = ENGINE_PRESETS[target]
    if getattr(args, "decimation", None) in (None, 300000):
        args.decimation = dec
    if getattr(args, "texture_size", None) in (None, 2048):
        args.texture_size = tex
    return args
