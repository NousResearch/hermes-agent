#!/usr/bin/env python3
"""
mesh_studio.py — Pure Python Procedural 3D Mesh Engine & Game Asset Synthesizer.

Zero external dependencies: uses Python standard library only (math, struct, json, os, sys, argparse).
Generates binary glTF 2.0 (.glb), Wavefront (.obj/.mtl), Godot 4 (.tscn), Three.js HTML previews,
and headless Blender automation scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vector3 & Math Helpers
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vec3:
        l = self.length()
        if l < 1e-8:
            return Vec3(0, 1, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)


@dataclass
class ColorRGB:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def to_tuple_rgb(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def to_tuple_rgba(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)


@dataclass
class Mesh:
    name: str = "mesh"
    vertices: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    colors: List[ColorRGB] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    material_name: str = "default_mat"
    base_color: ColorRGB = field(default_factory=lambda: ColorRGB(0.8, 0.8, 0.8, 1.0))
    roughness: float = 0.5
    metallic: float = 0.0

    def calculate_normals(self) -> None:
        """Compute flat or smooth per-vertex normals based on face triangles."""
        norm_accum = [Vec3(0, 0, 0) for _ in self.vertices]
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break
            idx0, idx1, idx2 = self.indices[i], self.indices[i + 1], self.indices[i + 2]
            if max(idx0, idx1, idx2) >= len(self.vertices):
                continue
            v0, v1, v2 = self.vertices[idx0], self.vertices[idx1], self.vertices[idx2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_norm = edge1.cross(edge2).normalized()
            norm_accum[idx0] = norm_accum[idx0] + face_norm
            norm_accum[idx1] = norm_accum[idx1] + face_norm
            norm_accum[idx2] = norm_accum[idx2] + face_norm

        self.normals = [n.normalized() for n in norm_accum]

    def transform(self, offset: Vec3 = Vec3(0, 0, 0), scale: Vec3 = Vec3(1, 1, 1)) -> Mesh:
        """Apply in-place translation and scaling."""
        for v in self.vertices:
            v.x = v.x * scale.x + offset.x
            v.y = v.y * scale.y + offset.y
            v.z = v.z * scale.z + offset.z
        return self

    def merge(self, other: Mesh) -> Mesh:
        """Merge another mesh into this mesh."""
        base_idx = len(self.vertices)
        self.vertices.extend([Vec3(v.x, v.y, v.z) for v in other.vertices])
        self.normals.extend([Vec3(n.x, n.y, n.z) for n in other.normals])
        if other.colors:
            self.colors.extend([ColorRGB(c.r, c.g, c.b, c.a) for c in other.colors])
        elif self.colors:
            self.colors.extend([other.base_color for _ in other.vertices])
        self.indices.extend([i + base_idx for i in other.indices])
        return self


# ---------------------------------------------------------------------------
# Procedural Geometry Generators
# ---------------------------------------------------------------------------

def create_cube(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    color: ColorRGB = ColorRGB(0.7, 0.7, 0.7),
) -> Mesh:
    mesh = Mesh(name="cube", base_color=color)
    hw, hh, hd = width * 0.5, height * 0.5, depth * 0.5

    faces = [
        ([Vec3(-hw, -hh, hd), Vec3(hw, -hh, hd), Vec3(hw, hh, hd), Vec3(-hw, hh, hd)], Vec3(0, 0, 1)),
        ([Vec3(hw, -hh, -hd), Vec3(-hw, -hh, -hd), Vec3(-hw, hh, -hd), Vec3(hw, hh, -hd)], Vec3(0, 0, -1)),
        ([Vec3(-hw, hh, hd), Vec3(hw, hh, hd), Vec3(hw, hh, -hd), Vec3(-hw, hh, -hd)], Vec3(0, 1, 0)),
        ([Vec3(-hw, -hh, -hd), Vec3(hw, -hh, -hd), Vec3(hw, -hh, hd), Vec3(-hw, -hh, hd)], Vec3(0, -1, 0)),
        ([Vec3(hw, -hh, hd), Vec3(hw, -hh, -hd), Vec3(hw, hh, -hd), Vec3(hw, hh, hd)], Vec3(1, 0, 0)),
        ([Vec3(-hw, -hh, -hd), Vec3(-hw, -hh, hd), Vec3(-hw, hh, hd), Vec3(-hw, hh, -hd)], Vec3(-1, 0, 0)),
    ]

    for verts, norm in faces:
        base = len(mesh.vertices)
        for v in verts:
            mesh.vertices.append(v)
            mesh.normals.append(norm)
            mesh.colors.append(color)
        mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return mesh


def create_cylinder(
    radius_top: float = 0.5,
    radius_bottom: float = 0.5,
    height: float = 1.0,
    radial_segments: int = 16,
    color: ColorRGB = ColorRGB(0.6, 0.6, 0.6),
) -> Mesh:
    mesh = Mesh(name="cylinder", base_color=color)
    hh = height * 0.5

    for i in range(radial_segments):
        theta0 = (i / radial_segments) * math.tau
        theta1 = ((i + 1) / radial_segments) * math.tau

        c0, s0 = math.cos(theta0), math.sin(theta0)
        c1, s1 = math.cos(theta1), math.sin(theta1)

        v0 = Vec3(c0 * radius_bottom, -hh, s0 * radius_bottom)
        v1 = Vec3(c1 * radius_bottom, -hh, s1 * radius_bottom)
        v2 = Vec3(c1 * radius_top, hh, s1 * radius_top)
        v3 = Vec3(c0 * radius_top, hh, s0 * radius_top)

        n0 = Vec3(c0, 0, s0).normalized()
        n1 = Vec3(c1, 0, s1).normalized()

        base = len(mesh.vertices)
        mesh.vertices.extend([v0, v1, v2, v3])
        mesh.normals.extend([n0, n1, n1, n0])
        mesh.colors.extend([color] * 4)
        mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    top_center_idx = len(mesh.vertices)
    mesh.vertices.append(Vec3(0, hh, 0))
    mesh.normals.append(Vec3(0, 1, 0))
    mesh.colors.append(color)
    for i in range(radial_segments):
        th = (i / radial_segments) * math.tau
        mesh.vertices.append(Vec3(math.cos(th) * radius_top, hh, math.sin(th) * radius_top))
        mesh.normals.append(Vec3(0, 1, 0))
        mesh.colors.append(color)

    for i in range(radial_segments):
        curr = top_center_idx + 1 + i
        nxt = top_center_idx + 1 + ((i + 1) % radial_segments)
        mesh.indices.extend([top_center_idx, curr, nxt])

    bot_center_idx = len(mesh.vertices)
    mesh.vertices.append(Vec3(0, -hh, 0))
    mesh.normals.append(Vec3(0, -1, 0))
    mesh.colors.append(color)
    for i in range(radial_segments):
        th = (i / radial_segments) * math.tau
        mesh.vertices.append(Vec3(math.cos(th) * radius_bottom, -hh, math.sin(th) * radius_bottom))
        mesh.normals.append(Vec3(0, -1, 0))
        mesh.colors.append(color)

    for i in range(radial_segments):
        curr = bot_center_idx + 1 + i
        nxt = bot_center_idx + 1 + ((i + 1) % radial_segments)
        mesh.indices.extend([bot_center_idx, nxt, curr])

    return mesh


def create_uv_sphere(
    radius: float = 0.5,
    rings: int = 12,
    sectors: int = 16,
    color: ColorRGB = ColorRGB(0.4, 0.7, 0.9),
) -> Mesh:
    mesh = Mesh(name="sphere", base_color=color)

    for r in range(rings + 1):
        phi = (r / rings) * math.pi
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        for s in range(sectors + 1):
            theta = (s / sectors) * math.tau
            sin_th = math.sin(theta)
            cos_th = math.cos(theta)

            x = cos_th * sin_phi
            y = cos_phi
            z = sin_th * sin_phi

            norm = Vec3(x, y, z)
            vert = norm * radius
            mesh.vertices.append(vert)
            mesh.normals.append(norm)
            mesh.colors.append(color)

    for r in range(rings):
        for s in range(sectors):
            cur = r * (sectors + 1) + s
            nxt = cur + sectors + 1
            mesh.indices.extend([cur, nxt, cur + 1, cur + 1, nxt, nxt + 1])

    return mesh


def create_cone(
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 16,
    color: ColorRGB = ColorRGB(0.2, 0.8, 0.3),
) -> Mesh:
    return create_cylinder(radius_top=0.0, radius_bottom=radius, height=height, radial_segments=segments, color=color)


# ---------------------------------------------------------------------------
# Procedural Game Assets
# ---------------------------------------------------------------------------

def generate_lowpoly_tree(
    trunk_radius: float = 0.15,
    trunk_height: float = 1.2,
    foliage_tiers: int = 3,
    foliage_radius: float = 0.8,
    seed: int = 42,
) -> Mesh:
    rng = random.Random(seed)
    trunk_col = ColorRGB(0.45, 0.28, 0.15)
    foliage_col = ColorRGB(0.18, 0.55, 0.22)

    tree = create_cylinder(
        radius_top=trunk_radius * 0.75,
        radius_bottom=trunk_radius,
        height=trunk_height,
        radial_segments=6,
        color=trunk_col,
    )
    tree.transform(offset=Vec3(0, trunk_height * 0.5, 0))

    cur_y = trunk_height * 0.65
    cur_r = foliage_radius
    cone_h = 0.9

    for tier in range(foliage_tiers):
        tier_col = ColorRGB(
            foliage_col.r + (rng.random() - 0.5) * 0.05,
            foliage_col.g + (tier * 0.04),
            foliage_col.b + (rng.random() - 0.5) * 0.05,
        )
        cone = create_cone(
            radius=cur_r,
            height=cone_h,
            segments=7,
            color=tier_col,
        )
        cone.transform(offset=Vec3(0, cur_y + cone_h * 0.5, 0))
        tree.merge(cone)

        cur_y += cone_h * 0.55
        cur_r *= 0.75
        cone_h *= 0.85

    tree.name = "procedural_tree"
    return tree


def generate_rock(
    radius: float = 0.7,
    roughness: float = 0.35,
    seed: int = 1337,
) -> Mesh:
    rng = random.Random(seed)
    rock_col = ColorRGB(0.48, 0.46, 0.44)
    sphere = create_uv_sphere(radius=radius, rings=6, sectors=8, color=rock_col)

    for v in sphere.vertices:
        noise = 1.0 + (rng.random() * 2.0 - 1.0) * roughness
        v.x *= noise
        v.y *= noise * (0.75 + rng.random() * 0.3)
        v.z *= noise

    sphere.calculate_normals()
    sphere.name = "procedural_rock"
    return sphere


def generate_castle_tower(
    radius: float = 1.0,
    height: float = 3.5,
    battlements: int = 8,
    seed: int = 101,
) -> Mesh:
    stone_col = ColorRGB(0.55, 0.55, 0.58)
    roof_col = ColorRGB(0.65, 0.22, 0.18)

    tower = create_cylinder(
        radius_top=radius,
        radius_bottom=radius * 1.08,
        height=height,
        radial_segments=12,
        color=stone_col,
    )
    tower.transform(offset=Vec3(0, height * 0.5, 0))

    ledge_h = 0.2
    ledge = create_cylinder(
        radius_top=radius * 1.2,
        radius_bottom=radius * 1.15,
        height=ledge_h,
        radial_segments=12,
        color=ColorRGB(0.5, 0.5, 0.52),
    )
    ledge.transform(offset=Vec3(0, height + ledge_h * 0.5, 0))
    tower.merge(ledge)

    cren_w = (math.tau * radius * 1.18) / (battlements * 2)
    cren_h = 0.45
    cren_d = 0.2
    for b in range(battlements):
        th = (b / battlements) * math.tau
        cx = math.cos(th) * (radius * 1.1)
        cz = math.sin(th) * (radius * 1.1)
        cren = create_cube(width=cren_w, height=cren_h, depth=cren_d, color=stone_col)
        cren.transform(offset=Vec3(cx, height + ledge_h + cren_h * 0.5, cz))
        tower.merge(cren)

    roof = create_cone(
        radius=radius * 0.9,
        height=1.4,
        segments=12,
        color=roof_col,
    )
    roof.transform(offset=Vec3(0, height + ledge_h + 0.7, 0))
    tower.merge(roof)

    tower.name = "castle_tower"
    return tower


def generate_fantasy_sword(
    blade_length: float = 2.2,
    blade_width: float = 0.2,
    guard_width: float = 0.8,
) -> Mesh:
    steel_col = ColorRGB(0.85, 0.88, 0.92)
    gold_col = ColorRGB(0.85, 0.65, 0.18)
    leather_col = ColorRGB(0.35, 0.22, 0.12)

    blade = create_cube(width=blade_width, height=blade_length, depth=0.04, color=steel_col)
    blade.transform(offset=Vec3(0, blade_length * 0.5 + 0.15, 0))

    guard = create_cube(width=guard_width, height=0.1, depth=0.12, color=gold_col)
    guard.transform(offset=Vec3(0, 0.1, 0))
    blade.merge(guard)

    grip = create_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.5, radial_segments=6, color=leather_col)
    grip.transform(offset=Vec3(0, -0.2, 0))
    blade.merge(grip)

    pommel = create_uv_sphere(radius=0.08, rings=4, sectors=6, color=gold_col)
    pommel.transform(offset=Vec3(0, -0.48, 0))
    blade.merge(pommel)

    blade.name = "fantasy_sword"
    return blade


def generate_dungeon_tile(
    size: float = 2.0,
    wall_height: float = 1.5,
    north_wall: bool = True,
    east_wall: bool = False,
    south_wall: bool = False,
    west_wall: bool = True,
) -> Mesh:
    floor_col = ColorRGB(0.38, 0.38, 0.4)
    wall_col = ColorRGB(0.48, 0.45, 0.43)

    tile = create_cube(width=size, height=0.15, depth=size, color=floor_col)
    tile.transform(offset=Vec3(0, -0.075, 0))

    wall_thickness = 0.2
    hw = size * 0.5
    hh = wall_height * 0.5

    if north_wall:
        w = create_cube(width=size, height=wall_height, depth=wall_thickness, color=wall_col)
        w.transform(offset=Vec3(0, hh, -hw + wall_thickness * 0.5))
        tile.merge(w)

    if south_wall:
        w = create_cube(width=size, height=wall_height, depth=wall_thickness, color=wall_col)
        w.transform(offset=Vec3(0, hh, hw - wall_thickness * 0.5))
        tile.merge(w)

    if west_wall:
        w = create_cube(width=wall_thickness, height=wall_height, depth=size, color=wall_col)
        w.transform(offset=Vec3(-hw + wall_thickness * 0.5, hh, 0))
        tile.merge(w)

    if east_wall:
        w = create_cube(width=wall_thickness, height=wall_height, depth=size, color=wall_col)
        w.transform(offset=Vec3(hw - wall_thickness * 0.5, hh, 0))
        tile.merge(w)

    tile.name = "dungeon_tile"
    return tile


def generate_crate(size: float = 1.0) -> Mesh:
    wood_col = ColorRGB(0.55, 0.38, 0.22)
    metal_col = ColorRGB(0.3, 0.32, 0.35)

    base_crate = create_cube(width=size * 0.96, height=size * 0.96, depth=size * 0.96, color=wood_col)

    f = size * 0.08
    edge1 = create_cube(width=size, height=f, depth=f, color=metal_col)
    edge1.transform(offset=Vec3(0, size * 0.48, size * 0.48))
    base_crate.merge(edge1)

    edge2 = create_cube(width=size, height=f, depth=f, color=metal_col)
    edge2.transform(offset=Vec3(0, -size * 0.48, size * 0.48))
    base_crate.merge(edge2)

    base_crate.name = "cargo_crate"
    return base_crate


def generate_terrain(
    grid_size: int = 16,
    cell_size: float = 0.5,
    height_scale: float = 1.2,
    seed: int = 777,
) -> Mesh:
    rng = random.Random(seed)
    mesh = Mesh(name="procedural_terrain")

    w = grid_size + 1
    heights: List[float] = []
    colors: List[ColorRGB] = []

    for z in range(w):
        for x in range(w):
            nx = (x / grid_size) * math.tau * 1.5
            nz = (z / grid_size) * math.tau * 1.5
            h = (math.sin(nx) * math.cos(nz) + math.sin(nx * 2.3 + 0.5) * 0.5) * height_scale
            h += (rng.random() - 0.5) * 0.15
            heights.append(h)

            if h > height_scale * 0.6:
                colors.append(ColorRGB(0.9, 0.92, 0.95))
            elif h > height_scale * 0.1:
                colors.append(ColorRGB(0.45, 0.42, 0.38))
            elif h > -height_scale * 0.2:
                colors.append(ColorRGB(0.25, 0.6, 0.22))
            else:
                colors.append(ColorRGB(0.78, 0.72, 0.45))

    offset_x = (grid_size * cell_size) * -0.5
    offset_z = (grid_size * cell_size) * -0.5

    for z in range(w):
        for x in range(w):
            idx = z * w + x
            px = offset_x + x * cell_size
            py = heights[idx]
            pz = offset_z + z * cell_size
            mesh.vertices.append(Vec3(px, py, pz))
            mesh.colors.append(colors[idx])
            mesh.normals.append(Vec3(0, 1, 0))

    for z in range(grid_size):
        for x in range(grid_size):
            i0 = z * w + x
            i1 = i0 + 1
            i2 = (z + 1) * w + x
            i3 = i2 + 1
            mesh.indices.extend([i0, i2, i1, i1, i2, i3])

    mesh.calculate_normals()
    return mesh


# ---------------------------------------------------------------------------
# Exporters: Binary glTF (.glb), Wavefront OBJ, Godot (.tscn), HTML Preview
# ---------------------------------------------------------------------------

def export_obj(mesh: Mesh, filepath: Path) -> Tuple[Path, Path]:
    obj_path = filepath.with_suffix(".obj")
    mtl_path = filepath.with_suffix(".mtl")

    with open(mtl_path, "w", encoding="utf-8") as fm:
        fm.write(f"# Material created by Hermes procedural-3d-studio\n")
        fm.write(f"newmtl {mesh.material_name}\n")
        fm.write(f"Kd {mesh.base_color.r:.4f} {mesh.base_color.g:.4f} {mesh.base_color.b:.4f}\n")
        fm.write(f"Ka 0.1 0.1 0.1\n")
        fm.write(f"Ks 0.2 0.2 0.2\n")
        fm.write(f"d {mesh.base_color.a:.4f}\n")
        fm.write(f"illum 2\n")

    with open(obj_path, "w", encoding="utf-8") as fo:
        fo.write(f"# Wavefront OBJ exported by Hermes procedural-3d-studio\n")
        fo.write(f"mtllib {mtl_path.name}\n")
        fo.write(f"o {mesh.name}\n\n")

        has_colors = bool(mesh.colors and len(mesh.colors) == len(mesh.vertices))
        for i, v in enumerate(mesh.vertices):
            if has_colors:
                c = mesh.colors[i]
                fo.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f} {c.r:.4f} {c.g:.4f} {c.b:.4f}\n")
            else:
                fo.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")

        for n in mesh.normals:
            fo.write(f"vn {n.x:.6f} {n.y:.6f} {n.z:.6f}\n")

        fo.write(f"\nusemtl {mesh.material_name}\n")
        fo.write(f"s 1\n")

        has_normals = bool(mesh.normals and len(mesh.normals) == len(mesh.vertices))
        for i in range(0, len(mesh.indices), 3):
            i0, i1, i2 = mesh.indices[i] + 1, mesh.indices[i + 1] + 1, mesh.indices[i + 2] + 1
            if has_normals:
                fo.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            else:
                fo.write(f"f {i0} {i1} {i2}\n")

    return obj_path, mtl_path


def export_glb(mesh: Mesh, filepath: Path) -> Path:
    glb_path = filepath.with_suffix(".glb")

    pos_data = bytearray()
    min_pos = [float("inf")] * 3
    max_pos = [float("-inf")] * 3
    for v in mesh.vertices:
        pos_data.extend(struct.pack("<3f", v.x, v.y, v.z))
        min_pos[0] = min(min_pos[0], v.x)
        min_pos[1] = min(min_pos[1], v.y)
        min_pos[2] = min(min_pos[2], v.z)
        max_pos[0] = max(max_pos[0], v.x)
        max_pos[1] = max(max_pos[1], v.y)
        max_pos[2] = max(max_pos[2], v.z)

    norm_data = bytearray()
    for n in mesh.normals:
        norm_data.extend(struct.pack("<3f", n.x, n.y, n.z))

    color_data = bytearray()
    has_colors = bool(mesh.colors and len(mesh.colors) == len(mesh.vertices))
    if has_colors:
        for c in mesh.colors:
            color_data.extend(struct.pack("<4f", c.r, c.g, c.b, c.a))

    idx_data = bytearray()
    min_idx = min(mesh.indices) if mesh.indices else 0
    max_idx = max(mesh.indices) if mesh.indices else 0
    for idx in mesh.indices:
        idx_data.extend(struct.pack("<I", idx))

    def pad4(b: bytearray) -> bytearray:
        rem = len(b) % 4
        if rem:
            b.extend(b"\x00" * (4 - rem))
        return b

    pos_data = pad4(pos_data)
    norm_data = pad4(norm_data)
    if has_colors:
        color_data = pad4(color_data)
    idx_data = pad4(idx_data)

    buffer_bytes = bytearray()
    pos_offset = len(buffer_bytes)
    buffer_bytes.extend(pos_data)

    norm_offset = len(buffer_bytes)
    buffer_bytes.extend(norm_data)

    color_offset = 0
    if has_colors:
        color_offset = len(buffer_bytes)
        buffer_bytes.extend(color_data)

    idx_offset = len(buffer_bytes)
    buffer_bytes.extend(idx_data)

    buffer_views = [
        {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_data), "target": 34962},
        {"buffer": 0, "byteOffset": norm_offset, "byteLength": len(norm_data), "target": 34962},
    ]
    if has_colors:
        buffer_views.append({"buffer": 0, "byteOffset": color_offset, "byteLength": len(color_data), "target": 34962})
    buffer_views.append({"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_data), "target": 34963})

    accessors = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(mesh.vertices),
            "type": "VEC3",
            "min": min_pos,
            "max": max_pos,
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(mesh.normals),
            "type": "VEC3",
        },
    ]

    attributes = {"POSITION": 0, "NORMAL": 1}
    next_view_idx = 2

    if has_colors:
        accessors.append({
            "bufferView": next_view_idx,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(mesh.colors),
            "type": "VEC4",
        })
        attributes["COLOR_0"] = next_view_idx
        next_view_idx += 1

    idx_accessor_idx = len(accessors)
    accessors.append({
        "bufferView": next_view_idx,
        "byteOffset": 0,
        "componentType": 5125,
        "count": len(mesh.indices),
        "type": "SCALAR",
        "min": [min_idx],
        "max": [max_idx],
    })

    gltf_dict: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Hermes procedural-3d-studio"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0, "name": mesh.name}],
        "meshes": [{
            "name": mesh.name,
            "primitives": [{
                "attributes": attributes,
                "indices": idx_accessor_idx,
                "material": 0,
            }],
        }],
        "materials": [{
            "name": mesh.material_name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(mesh.base_color.to_tuple_rgba()),
                "metallicFactor": mesh.metallic,
                "roughnessFactor": mesh.roughness,
            },
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffer_bytes)}],
    }

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    rem = len(json_bytes) % 4
    if rem:
        json_bytes += b" " * (4 - rem)

    glb_header_len = 12
    json_chunk_header_len = 8
    bin_chunk_header_len = 8
    total_length = glb_header_len + json_chunk_header_len + len(json_bytes) + bin_chunk_header_len + len(buffer_bytes)

    glb_file = bytearray()
    glb_file.extend(b"glTF")
    glb_file.extend(struct.pack("<II", 2, total_length))

    glb_file.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    glb_file.extend(json_bytes)

    glb_file.extend(struct.pack("<II", len(buffer_bytes), 0x004E4942))
    glb_file.extend(buffer_bytes)

    with open(glb_path, "wb") as f:
        f.write(glb_file)

    return glb_path


def export_godot_scene(mesh: Mesh, glb_path: Path, output_path: Path) -> Path:
    tscn_path = output_path.with_suffix(".tscn")
    rel_glb = os.path.relpath(glb_path, tscn_path.parent).replace("\\", "/")

    content = f"""[gd_scene load_steps=2 format=3 uid="uid://mesh_{abs(hash(mesh.name)) % 1000000}"]

[ext_resource type="PackedScene" path="{rel_glb}" id="1_mesh"]

[node name="{mesh.name.capitalize()}" type="Node3D"]

[node name="ModelInstance" parent="." instance=ExtResource("1_mesh")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
"""
    with open(tscn_path, "w", encoding="utf-8") as f:
        f.write(content)
    return tscn_path


def export_threejs_preview(mesh: Mesh, glb_path: Path, output_path: Path) -> Path:
    import html
    import json

    html_path = output_path.with_suffix(".html")
    rel_glb = os.path.relpath(glb_path, html_path.parent).replace("\\", "/")
    escaped_name = html.escape(mesh.name)
    json_rel_glb = json.dumps(rel_glb)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>3D Preview - {escaped_name}</title>
  <style>
    body {{ margin: 0; overflow: hidden; background: #121418; font-family: sans-serif; }}
    #info {{ position: absolute; top: 12px; left: 16px; color: #e0e0e0; z-index: 10; pointer-events: none; }}
    #info h1 {{ margin: 0 0 4px 0; font-size: 18px; font-weight: 600; color: #fff; }}
    #info p {{ margin: 0; font-size: 12px; color: #9aa0a6; }}
    #controls-hint {{ position: absolute; bottom: 12px; left: 16px; color: #70757a; font-size: 11px; }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
</head>
<body>
  <div id="info">
    <h1>{escaped_name}</h1>
    <p>Vertices: {len(mesh.vertices)} | Triangles: {len(mesh.indices) // 3}</p>
  </div>
  <div id="controls-hint">Left click + drag to orbit | Right click to pan | Scroll to zoom</div>

  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121418);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(3, 3, 4);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x6688cc, 0.6);
    dirLight2.position.set(-5, -2, -5);
    scene.add(dirLight2);

    const grid = new THREE.GridHelper(10, 20, 0x333a42, 0x22262c);
    grid.position.y = 0;
    scene.add(grid);

    const loader = new GLTFLoader();
    loader.load({json_rel_glb}, (gltf) => {{
      const model = gltf.scene;
      scene.add(model);

      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.set(center.x + maxDim * 1.5, center.y + maxDim * 1.2, center.z + maxDim * 1.8);
      controls.target.copy(center);
      controls.update();
    }});

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def export_blender_script(mesh: Mesh, filepath: Path) -> Path:
    py_path = filepath.with_suffix(".blender.py")
    script_content = f'''"""
Headless Blender script for {mesh.name}.
Run with: blender --background --python {py_path.name}
"""
import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

glb_path = "{filepath.with_suffix('.glb').as_posix()}"
bpy.ops.import_scene.gltf(filepath=glb_path)

cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (3.5, -3.5, 2.8)
cam_obj.rotation_euler = (math.radians(65), 0, math.radians(45))
bpy.context.scene.camera = cam_obj

light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
light_data.energy = 3.5
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(45), math.radians(30), 0)

bpy.context.scene.render.engine = 'CYCLES' if bpy.app.version >= (3, 0, 0) else 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.filepath = "{filepath.with_suffix('.png').as_posix()}"

print(f"[Blender] Rendering {mesh.name} snapshot...")
bpy.ops.render.render(write_still=True)
print(f"[Blender] Render saved to {filepath.with_suffix('.png').as_posix()}")
'''
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    return py_path


# ---------------------------------------------------------------------------
# CLI Argument Parser & Dispatcher
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_studio",
        description="Pure Python Procedural 3D Mesh Engine & Game Asset Synthesizer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prim = sub.add_parser("primitive", help="Generate 3D primitive (cube, sphere, cylinder, cone)")
    p_prim.add_argument("type", choices=["cube", "sphere", "cylinder", "cone"], help="Primitive shape")
    p_prim.add_argument("-o", "--output", default="primitive_mesh", help="Base output filename")
    p_prim.add_argument("--size", type=float, default=1.0, help="Size/diameter")

    p_tree = sub.add_parser("tree", help="Generate procedural stylized low-poly tree")
    p_tree.add_argument("-o", "--output", default="tree", help="Output filename")
    p_tree.add_argument("--tiers", type=int, default=3, help="Foliage tiers")
    p_tree.add_argument("--height", type=float, default=1.2, help="Trunk height")
    p_tree.add_argument("--seed", type=int, default=42, help="Random seed")

    p_rock = sub.add_parser("rock", help="Generate procedural deformed rock/boulder")
    p_rock.add_argument("-o", "--output", default="rock", help="Output filename")
    p_rock.add_argument("--radius", type=float, default=0.7, help="Base radius")
    p_rock.add_argument("--roughness", type=float, default=0.35, help="Facet roughness")
    p_rock.add_argument("--seed", type=int, default=1337, help="Random seed")

    p_tower = sub.add_parser("tower", help="Generate medieval castle tower with battlements")
    p_tower.add_argument("-o", "--output", default="castle_tower", help="Output filename")
    p_tower.add_argument("--height", type=float, default=3.5, help="Tower height")
    p_tower.add_argument("--battlements", type=int, default=8, help="Number of battlements")

    p_sword = sub.add_parser("sword", help="Generate fantasy sword weapon model")
    p_sword.add_argument("-o", "--output", default="sword", help="Output filename")
    p_sword.add_argument("--length", type=float, default=2.2, help="Blade length")

    p_dung = sub.add_parser("dungeon", help="Generate modular 3D dungeon tile")
    p_dung.add_argument("-o", "--output", default="dungeon_tile", help="Output filename")
    p_dung.add_argument("--walls", default="nw", help="Wall configuration string (e.g. 'nw', 'nesw', 'none')")

    p_crate = sub.add_parser("crate", help="Generate reinforced low-poly cargo crate")
    p_crate.add_argument("-o", "--output", default="crate", help="Output filename")
    p_crate.add_argument("--size", type=float, default=1.0, help="Crate cube size")

    p_terr = sub.add_parser("terrain", help="Generate procedural heightmap terrain")
    p_terr.add_argument("-o", "--output", default="terrain", help="Output filename")
    p_terr.add_argument("--grid-size", type=int, default=16, help="Grid subdivision size")
    p_terr.add_argument("--scale", type=float, default=1.2, help="Height multiplier")
    p_terr.add_argument("--seed", type=int, default=777, help="Random seed")

    for p in [p_prim, p_tree, p_rock, p_tower, p_sword, p_dung, p_crate, p_terr]:
        p.add_argument("--format", choices=["glb", "obj", "all"], default="all", help="Export file format")
        p.add_argument("--preview", action="store_true", default=True, help="Generate interactive HTML WebGL preview")
        p.add_argument("--godot", action="store_true", default=True, help="Generate Godot 4 .tscn scene file")
        p.add_argument("--blender", action="store_true", default=True, help="Generate headless Blender script")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mesh: Optional[Mesh] = None

    if args.command == "primitive":
        if args.type == "cube":
            mesh = create_cube(width=args.size, height=args.size, depth=args.size)
        elif args.type == "sphere":
            mesh = create_uv_sphere(radius=args.size * 0.5)
        elif args.type == "cylinder":
            mesh = create_cylinder(radius_top=args.size * 0.5, radius_bottom=args.size * 0.5, height=args.size)
        elif args.type == "cone":
            mesh = create_cone(radius=args.size * 0.5, height=args.size)
    elif args.command == "tree":
        mesh = generate_lowpoly_tree(trunk_height=args.height, foliage_tiers=args.tiers, seed=args.seed)
    elif args.command == "rock":
        mesh = generate_rock(radius=args.radius, roughness=args.roughness, seed=args.seed)
    elif args.command == "tower":
        mesh = generate_castle_tower(height=args.height, battlements=args.battlements)
    elif args.command == "sword":
        mesh = generate_fantasy_sword(blade_length=args.length)
    elif args.command == "dungeon":
        w_str = args.walls.lower()
        mesh = generate_dungeon_tile(
            north_wall="n" in w_str,
            east_wall="e" in w_str,
            south_wall="s" in w_str,
            west_wall="w" in w_str,
        )
    elif args.command == "crate":
        mesh = generate_crate(size=args.size)
    elif args.command == "terrain":
        mesh = generate_terrain(grid_size=args.grid_size, height_scale=args.scale, seed=args.seed)

    if mesh is None:
        sys.exit(1)

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    created_files = []

    if args.format in ("glb", "all"):
        glb_file = export_glb(mesh, out_base)
        created_files.append(glb_file)
        if args.preview:
            created_files.append(export_threejs_preview(mesh, glb_file, out_base))
        if args.godot:
            created_files.append(export_godot_scene(mesh, glb_file, out_base))
        if args.blender:
            created_files.append(export_blender_script(mesh, out_base))

    if args.format in ("obj", "all"):
        obj_f, mtl_f = export_obj(mesh, out_base)
        created_files.extend([obj_f, mtl_f])

    print(f"3D Asset Generated: '{mesh.name}' ({len(mesh.vertices)} verts, {len(mesh.indices)//3} tris)")
    for f in created_files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
