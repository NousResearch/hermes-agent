# nix/python.nix — uv2nix virtual environment builder
{
  python312,
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

  # Legacy alibabacloud packages ship only sdists with setup.py/setup.cfg
  # and no pyproject.toml, so setuptools isn't declared as a build dep.
  buildSystemOverrides = final: prev: builtins.mapAttrs
    (name: _: prev.${name}.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
    }))
    (lib.genAttrs [
      "alibabacloud-credentials-api"
      "alibabacloud-endpoint-util"
      "alibabacloud-gateway-dingtalk"
      "alibabacloud-gateway-spi"
      "alibabacloud-tea"
    ] (_: null));

  pythonPackageOverrides = final: _prev:
    if isAarch64Darwin then {
      numpy = mkPrebuiltOverride final python312.pkgs.numpy { };

      pyarrow = mkPrebuiltOverride final python312.pkgs.pyarrow { };

      av = mkPrebuiltOverride final python312.pkgs.av { };

      humanfriendly = mkPrebuiltOverride final python312.pkgs.humanfriendly { };

      coloredlogs = mkPrebuiltOverride final python312.pkgs.coloredlogs {
        humanfriendly = [ ];
      };

      onnxruntime = mkPrebuiltOverride final python312.pkgs.onnxruntime {
        coloredlogs = [ ];
        numpy = [ ];
        packaging = [ ];
      };

      ctranslate2 = mkPrebuiltOverride final python312.pkgs.ctranslate2 {
        numpy = [ ];
        pyyaml = [ ];
      };

      faster-whisper = mkPrebuiltOverride final python312.pkgs.faster-whisper {
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
      python = python312;
    }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        buildSystemOverrides
        pythonPackageOverrides
      ]);
in
pythonSet.mkVirtualEnv "hermes-agent-env" {
  hermes-agent = [ "all" ];
}
