# nix/python.nix — uv2nix virtual environment builder
{
  python311,
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./..; };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      python = python311;
    }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
      ]);
in
pythonSet.mkVirtualEnv "hermes-agent-env" {
  hermes-agent = [ "all" ];
}
