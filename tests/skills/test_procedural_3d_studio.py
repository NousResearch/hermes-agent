"""Tests for optional-skills/creative/procedural-3d-studio/scripts/mesh_studio.py"""

import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "procedural-3d-studio"
    / "scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))

import mesh_studio as ms


def test_vector_and_math():
    v1 = ms.Vec3(1, 2, 3)
    v2 = ms.Vec3(4, 5, 6)
    v_add = v1 + v2
    assert v_add == ms.Vec3(5, 7, 9)

    v_sub = v2 - v1
    assert v_sub == ms.Vec3(3, 3, 3)

    v_mul = v1 * 2.0
    assert v_mul == ms.Vec3(2, 4, 6)

    assert v1.dot(v2) == 1 * 4 + 2 * 5 + 3 * 6
    norm = ms.Vec3(0, 5, 0).normalized()
    assert norm.y == pytest.approx(1.0)


def test_primitives_geometry():
    cube = ms.create_cube(1.0, 1.0, 1.0)
    assert len(cube.vertices) == 24
    assert len(cube.indices) == 36

    sphere = ms.create_uv_sphere(radius=0.5, rings=6, sectors=8)
    assert len(sphere.vertices) > 0
    assert len(sphere.indices) > 0

    cylinder = ms.create_cylinder(radius_top=0.5, radius_bottom=0.5, height=1.0, radial_segments=8)
    assert len(cylinder.vertices) > 0
    assert len(cylinder.indices) > 0


def test_procedural_game_assets():
    tree = ms.generate_lowpoly_tree(trunk_radius=0.2, trunk_height=1.5, foliage_tiers=3, seed=42)
    assert tree.name == "procedural_tree"
    assert len(tree.vertices) > 30

    rock = ms.generate_rock(radius=0.8, roughness=0.3, seed=123)
    assert rock.name == "procedural_rock"
    assert len(rock.vertices) > 0

    tower = ms.generate_castle_tower(radius=1.0, height=3.0, battlements=6)
    assert tower.name == "castle_tower"
    assert len(tower.vertices) > 50

    sword = ms.generate_fantasy_sword(blade_length=2.0)
    assert sword.name == "fantasy_sword"
    assert len(sword.vertices) > 20

    tile = ms.generate_dungeon_tile(size=2.0, wall_height=1.5, north_wall=True, west_wall=True)
    assert tile.name == "dungeon_tile"
    assert len(tile.vertices) > 20

    crate = ms.generate_crate(size=1.0)
    assert crate.name == "cargo_crate"
    assert len(crate.vertices) > 24

    terrain = ms.generate_terrain(grid_size=8, cell_size=0.5, height_scale=1.0, seed=555)
    assert terrain.name == "procedural_terrain"
    assert len(terrain.vertices) == 9 * 9
    assert len(terrain.indices) == 8 * 8 * 6


def test_export_glb_binary_integrity():
    tree = ms.generate_lowpoly_tree(trunk_height=1.0, foliage_tiers=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "test_tree"
        glb_file = ms.export_glb(tree, out_base)

        assert glb_file.exists()
        assert glb_file.stat().st_size > 100

        with open(glb_file, "rb") as f:
            data = f.read()

        # Check glTF 2.0 Binary Header
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        assert magic == b"glTF"
        assert version == 2
        assert length == len(data)

        # Check JSON Chunk header
        json_len, json_type = struct.unpack_from("<II", data, 12)
        assert json_type == 0x4E4F534A  # 'JSON'
        assert 12 + 8 + json_len <= len(data)


def test_export_obj_and_mtl():
    cube = ms.create_cube()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "test_cube"
        obj_file, mtl_file = ms.export_obj(cube, out_base)

        assert obj_file.exists()
        assert mtl_file.exists()

        obj_text = obj_file.read_text(encoding="utf-8")
        assert "v " in obj_text
        assert "f " in obj_text
        assert f"mtllib {mtl_file.name}" in obj_text


def test_export_godot_and_threejs():
    rock = ms.generate_rock()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "test_rock"
        glb_file = ms.export_glb(rock, out_base)
        tscn_file = ms.export_godot_scene(rock, glb_file, out_base)
        html_file = ms.export_threejs_preview(rock, glb_file, out_base)
        blender_file = ms.export_blender_script(rock, out_base)

        assert tscn_file.exists()
        assert html_file.exists()
        assert blender_file.exists()

        assert "[gd_scene" in tscn_file.read_text(encoding="utf-8")
        assert "GLTFLoader" in html_file.read_text(encoding="utf-8")
        assert "import bpy" in blender_file.read_text(encoding="utf-8")


def test_cli_parser_build():
    parser = ms.build_parser()
    args = parser.parse_args(["tree", "--tiers", "4", "--height", "2.0"])
    assert args.command == "tree"
    assert args.tiers == 4
    assert args.height == 2.0
