#!/usr/bin/env python3
"""Rig quality gate: does this GLB actually carry a usable skeleton?

Self-contained on purpose. The image-to-3D skill ships its own gates module;
this one covers only the rig check so this skill installs and runs on its own
(no cross-skill import, no home-directory path assumptions).

Stdlib only: a GLB's JSON chunk is parsed with struct + json.
"""
import json
import struct


def _parse_glb_json(path):
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError("not a GLB (bad magic)")
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # 'JSON'
            raise ValueError("first chunk is not JSON")
        return json.loads(f.read(chunk_len))


def gate_glb_rig(path, min_bones=5):
    """Return (ok, message): a skin exists, has joints, and weights are bound."""
    try:
        g = _parse_glb_json(path)
        skins = g.get("skins", [])
        if not skins:
            return False, "no skins in GLB — not rigged"
        joints = max(len(s.get("joints", [])) for s in skins)
        if joints < min_bones:
            return False, f"only {joints} joints — rig looks degenerate"
        has_weights = any(
            "JOINTS_0" in prim.get("attributes", {})
            and "WEIGHTS_0" in prim.get("attributes", {})
            for m in g.get("meshes", [])
            for prim in m.get("primitives", [])
        )
        if not has_weights:
            return False, "skin exists but no JOINTS_0/WEIGHTS_0 attributes"
        return True, f"rigged: {joints} joints, weights present"
    except Exception as e:
        return False, f"rig gate error: {e}"
