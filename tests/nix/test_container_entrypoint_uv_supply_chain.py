"""Regression tests for container first-boot uv provisioning."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NIXOS_MODULE = REPO_ROOT / "nix" / "nixosModules.nix"
HERMES_PACKAGE = REPO_ROOT / "nix" / "hermes-agent.nix"


def test_container_entrypoint_does_not_pipe_remote_uv_installer_to_shell() -> None:
    """Container startup must not execute a network-fetched uv installer."""
    module_text = NIXOS_MODULE.read_text(encoding="utf-8")

    assert "https://astral.sh/uv/install.sh" not in module_text
    assert "curl -LsSf" not in module_text
    assert "| sh" not in module_text


def test_container_uses_nix_built_uv_for_python_venv() -> None:
    """The writable venv should be seeded by pinned Nix store tooling."""
    module_text = NIXOS_MODULE.read_text(encoding="utf-8")
    package_text = HERMES_PACKAGE.read_text(encoding="utf-8")

    assert 'uv,' in package_text
    assert "${pkgs.uv}/bin/uv" in module_text
    assert "${pkgs.python312}/bin/python3" in module_text
    assert "$_UV_BIN venv --python $_PYTHON_BIN --seed $TARGET_HOME/.venv" in module_text
