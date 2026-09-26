"""Tests for check_deps.py — focuses on parsing logic that doesn't need a server."""

from __future__ import annotations

from check_deps import (
    NODE_TO_PACKAGE,
    model_present,
    normalize_for_match,
    suggest_install_command,
)


class TestNormalizeForMatch:
    def test_basic(self):
        s = normalize_for_match("model.safetensors")
        assert "model.safetensors" in s
        assert "model" in s

    def test_subfolder(self):
        s = normalize_for_match("subdir/model.pt")
        assert "subdir/model.pt" in s
        assert "model.pt" in s
        assert "model" in s


class TestModelPresent:
    def test_exact_match(self):
        assert model_present("a.safetensors", {"a.safetensors", "b.safetensors"}) is True

    def test_extension_difference(self):
        # User said "model" but installed is "model.safetensors"
        assert model_present("model", {"model.safetensors"}) is True
        # Reverse direction — also matches
        assert model_present("model.safetensors", {"model"}) is True

    def test_subfolder_match(self):
        # Installed list has "subdir/model.safetensors", workflow asks "model.safetensors"
        assert model_present("model.safetensors", {"subdir/model.safetensors"}) is True

    def test_missing(self):
        assert model_present("missing.safetensors", {"a.safetensors", "b.safetensors"}) is False

    def test_empty_installed(self):
        assert model_present("anything.safetensors", set()) is False


class TestSuggestInstallCommand:
    def test_known_node(self):
        cmd = suggest_install_command("VHS_VideoCombine")
        assert cmd == "comfy node install comfyui-videohelpersuite"

    def test_unknown_node(self):
        assert suggest_install_command("SomeRandomNodeName123") is None


class TestNodePackageMap:
    def test_no_duplicates(self):
        # Each node should map to exactly one package
        keys = list(NODE_TO_PACKAGE.keys())
        assert len(keys) == len(set(keys))

    def test_packages_are_safe_for_shell(self):
        # Registry slugs must be alphanumerics + hyphens/underscores only
        # (passed straight to `comfy node install <pkg>`).
        import re
        safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
        for pkg in NODE_TO_PACKAGE.values():
            assert safe.match(pkg), f"Unsafe package slug: {pkg!r}"
