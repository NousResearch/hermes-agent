# nix/python.nix — uv2nix virtual environment builder
{
  python312,
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  stdenv,
  dependency-groups ? ["all"],
}: let
  workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./..;};
  hacks = callPackage pyproject-nix.build.hacks {};

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  isAarch64Darwin = stdenv.hostPlatform.system == "aarch64-darwin";

  # Keep the workspace locked through uv2nix, but supply the local voice stack
  # from nixpkgs so wheel-only transitive artifacts do not break evaluation.
  mkPrebuiltPassthru = dependencies: {
    inherit dependencies;
    optional-dependencies = {};
    dependency-groups = {};
  };

  mkPrebuiltOverride = final: from: dependencies:
    hacks.nixpkgsPrebuilt {
      inherit from;
      prev = {
        nativeBuildInputs = [final.pyprojectHook];
        passthru = mkPrebuiltPassthru dependencies;
      };
    };

  # Legacy alibabacloud packages ship only sdists with setup.py/setup.cfg
  # and no pyproject.toml, so setuptools isn't declared as a build dep.
  buildSystemOverrides = final: prev:
    builtins.mapAttrs
    (name: _:
      prev.${name}.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
      }))
    (lib.genAttrs [
      "alibabacloud-credentials-api"
      "alibabacloud-endpoint-util"
      "alibabacloud-gateway-dingtalk"
      "alibabacloud-gateway-spi"
      "alibabacloud-tea"
    ] (_: null));

  withoutChecks = package:
    package.overridePythonAttrs (_old: {
      # These packages are consumed as prebuilt runtime inputs for the uv2nix
      # environment. On small aarch64-darwin builders, their upstream check
      # suites either OOM-kill directly (PyAV) or pull heavyweight test stacks
      # such as transformers/datasets/torch through check-only dependencies.
      doCheck = false;
      doInstallCheck = false;
      nativeCheckInputs = [];
      checkInputs = [];
      installCheckInputs = [];
      pythonImportsCheck = [];
    });

  pythonPackageOverrides = final: _prev:
    if isAarch64Darwin
    then let
      numpy = withoutChecks python312.pkgs.numpy;
      pyarrow = withoutChecks python312.pkgs.pyarrow;
      av = withoutChecks python312.pkgs.av;
      humanfriendly = withoutChecks python312.pkgs.humanfriendly;
      coloredlogs = withoutChecks (python312.pkgs.coloredlogs.override {
        inherit humanfriendly;
      });
      huggingface-hub = withoutChecks python312.pkgs.huggingface-hub;
      onnxruntime = withoutChecks (python312.pkgs.onnxruntime.override {
        inherit coloredlogs numpy;
      });
      tokenizers = withoutChecks (python312.pkgs.tokenizers.override {
        inherit huggingface-hub;
      });
      ctranslate2 = withoutChecks (python312.pkgs.ctranslate2.override {
        inherit numpy;
        torch = null;
        transformers = null;
      });
      faster-whisper = withoutChecks (python312.pkgs.faster-whisper.override {
        inherit av ctranslate2 huggingface-hub onnxruntime tokenizers;
      });
    in {
      numpy = mkPrebuiltOverride final numpy {};

      pyarrow = mkPrebuiltOverride final pyarrow {};

      av = mkPrebuiltOverride final av {};

      humanfriendly = mkPrebuiltOverride final humanfriendly {};

      coloredlogs = mkPrebuiltOverride final coloredlogs {
        humanfriendly = [];
      };

      onnxruntime = mkPrebuiltOverride final onnxruntime {
        coloredlogs = [];
        numpy = [];
        packaging = [];
      };

      ctranslate2 = mkPrebuiltOverride final ctranslate2 {
        numpy = [];
        pyyaml = [];
      };

      faster-whisper = mkPrebuiltOverride final faster-whisper {
        av = [];
        ctranslate2 = [];
        huggingface-hub = [];
        onnxruntime = [];
        tokenizers = [];
        tqdm = [];
      };
    }
    else {};

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
    hermes-agent = dependency-groups;
  }
