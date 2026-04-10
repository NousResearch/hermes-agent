# nix/python.nix — uv2nix virtual environment builder
{
  python,
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  stdenv,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./..; };
  hacks = callPackage pyproject-nix.build.hacks { };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  isAarch64Darwin = stdenv.hostPlatform.system == "aarch64-darwin";

  # Keep the workspace locked through uv2nix, but supply the local voice stack
  # from nixpkgs so wheel-only transitive artifacts do not break evaluation.
  mkPrebuiltPassthru = dependencies: {
    inherit dependencies;
    optional-dependencies = { };
    dependency-groups = { };
  };

  mkPrebuiltOverride = final: from: dependencies:
    hacks.nixpkgsPrebuilt {
      inherit from;
      prev = {
        nativeBuildInputs = [ final.pyprojectHook ];
        passthru = mkPrebuiltPassthru dependencies;
      };
    };

  pythonPackageOverrides = final: _prev:
    if isAarch64Darwin then {
      numpy = mkPrebuiltOverride final python.pkgs.numpy { };

      av = mkPrebuiltOverride final python.pkgs.av { };

      humanfriendly = mkPrebuiltOverride final python.pkgs.humanfriendly { };

      coloredlogs = mkPrebuiltOverride final python.pkgs.coloredlogs {
        humanfriendly = [ ];
      };

      onnxruntime = mkPrebuiltOverride final python.pkgs.onnxruntime {
        coloredlogs = [ ];
        numpy = [ ];
        packaging = [ ];
      };

      ctranslate2 = mkPrebuiltOverride final python.pkgs.ctranslate2 {
        numpy = [ ];
        pyyaml = [ ];
      };

      faster-whisper = mkPrebuiltOverride final python.pkgs.faster-whisper {
        av = [ ];
        ctranslate2 = [ ];
        huggingface-hub = [ ];
        onnxruntime = [ ];
        tokenizers = [ ];
        tqdm = [ ];
      };
    } else {};

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        pythonPackageOverrides
      ]);
in
pythonSet.mkVirtualEnv "hermes-agent-env" {
  hermes-agent = [ "all" ];
}
