"""Regression for #122593: workspace snapshots carry every member build input.

A nested project's uv.lock and nested dist assets must survive `_copy_core_inputs`
byte-identical — they are tracked build inputs — while the root's own lock and
build outputs stay out of the copy (the root lock is seeded separately) and junk
stays excluded at every depth.
"""
from pathlib import Path

from pm.workspace import _copy_core_inputs


def test_snapshot_keeps_member_build_inputs_and_drops_only_root_outputs(tmp_path):
    source = tmp_path / "core"
    (source / "pm" / "dist").mkdir(parents=True)
    (source / "plugins" / "demo" / "dashboard" / "dist").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["pm", "plugins"]\n', encoding="utf-8")
    (source / "uv.lock").write_text("root lock is seeded, never copied\n", encoding="utf-8")
    (source / "pm" / "pyproject.toml").write_text('[project]\nname = "pm"\nversion = "0"\n', encoding="utf-8")
    (source / "pm" / "uv.lock").write_text("member lock bytes\n", encoding="utf-8")
    (source / "pm" / "dist" / "asset.js").write_text("member dist asset\n", encoding="utf-8")
    (source / "plugins" / "demo" / "dashboard" / "dist" / "index.js").write_text("served asset\n", encoding="utf-8")
    for name in ("build", "dist", "release"):
        (source / name).mkdir()
        (source / name / "output.bin").write_text("root build output\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / "pm" / "venv").mkdir()
    (source / "pm" / "__pycache__").mkdir()
    (source / "plugins" / "demo" / "node_modules").mkdir(parents=True)

    destination = tmp_path / "snapshot"
    destination.mkdir()
    _copy_core_inputs(source, destination)

    for relative in ("pm/pyproject.toml", "pm/uv.lock", "pm/dist/asset.js",
                     "plugins/demo/dashboard/dist/index.js"):
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
    assert not (destination / "uv.lock").exists(), "the root lock is seeded, never copied"
    for name in ("build", "dist", "release", ".git"):
        assert not (destination / name).exists()
    for relative in ("pm/venv", "pm/__pycache__", "plugins/demo/node_modules"):
        assert not (destination / relative).exists()
