# nix/python.nix — uv2nix virtual environment builder
{
  python311,
  stdenv,
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

  # nixos-24.11 reports darwinSdkVersion "11.0" on aarch64-darwin, but
  # packages like onnxruntime only ship macosx_14_0 wheels.  Raise the
  # version so pyproject-nix's wheel-tag check accepts them — safe because
  # we install pre-built wheels, not compile against the SDK.
  darwinSdkOverride = final: prev:
    lib.optionalAttrs stdenv.hostPlatform.isDarwin {
      stdenv = prev.stdenv // {
        targetPlatform = prev.stdenv.targetPlatform // {
          darwinSdkVersion = "14.0";
        };
      };
    };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      python = python311;
    }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        darwinSdkOverride
        overlay
      ]);
in
pythonSet.mkVirtualEnv "hermes-agent-env" {
  hermes-agent = [ "all" ];
}
