# nix/python.nix — uv2nix virtual environment builder
{
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  stdenv,
  # Filtered Python source (see lib.nix pythonSrc) — keeps JS/docs/skills
  # edits from invalidating the venv derivation.
  pythonSrc,
  dependency-groups ? [ "all" ],
}:
let
  # The interpreter family comes from pm/lock.json (pythonLock.nix owns the
  # selection); every override below must be built for THAT interpreter.
  pythonLock = callPackage ./pythonLock.nix { };
  python = pythonLock.interpreter;
  pythonPackages = python.pkgs;

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = pythonSrc; };
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

  mkPrebuiltOverride =
    final: from: dependencies:
    hacks.nixpkgsPrebuilt {
      inherit from;
      prev = {
        nativeBuildInputs = [ final.pyprojectHook ];
        passthru = mkPrebuiltPassthru dependencies;
      };
    };

  # Legacy alibabacloud packages ship only sdists with setup.py/setup.cfg
  # and no pyproject.toml, so setuptools isn't declared as a build dep.
  buildSystemOverrides =
    final: prev:
    builtins.mapAttrs
      (
        name: _:
        prev.${name}.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
        })
      )
      (
        lib.genAttrs [
          "alibabacloud-credentials-api"
          "alibabacloud-endpoint-util"
          "alibabacloud-gateway-dingtalk"
          "alibabacloud-gateway-spi"
          "alibabacloud-tea"
        ] (_: null)
      )
    // {
      # The locked sdist has no build-system metadata; setup.py imports
      # setuptools and uses CFFI to compile the bundled libolm.
      python-olm = prev.python-olm.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem {
          setuptools = [ ];
          cffi = [ ];
        };
      });
    };

  pythonPackageOverrides =
    final: _prev:
    if isAarch64Darwin then
      {
        numpy = mkPrebuiltOverride final pythonPackages.numpy { };

        pyarrow = mkPrebuiltOverride final pythonPackages.pyarrow { };

        av = mkPrebuiltOverride final pythonPackages.av { };

        humanfriendly = mkPrebuiltOverride final pythonPackages.humanfriendly { };

        coloredlogs = mkPrebuiltOverride final pythonPackages.coloredlogs {
          humanfriendly = [ ];
        };

        onnxruntime = mkPrebuiltOverride final pythonPackages.onnxruntime {
          coloredlogs = [ ];
          numpy = [ ];
          packaging = [ ];
        };

        ctranslate2 = mkPrebuiltOverride final pythonPackages.ctranslate2 {
          numpy = [ ];
          pyyaml = [ ];
        };

        faster-whisper = mkPrebuiltOverride final pythonPackages.faster-whisper {
          av = [ ];
          ctranslate2 = [ ];
          huggingface-hub = [ ];
          onnxruntime = [ ];
          tokenizers = [ ];
          tqdm = [ ];
        };
      }
    else
      { };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          buildSystemOverrides
          pythonPackageOverrides
          # ``setup.py`` permits wheel/sdist creation only from the sealed
          # Hermes derivation. This is deliberately a derivation environment
          # variable, not a devShell variable: ``nix develop -c uv build``
          # must remain blocked.
          (final: prev: {
            hermes-agent = prev.hermes-agent.overrideAttrs (_old: {
              HERMES_NIX_BUILD = "1";
            });
          })
        ]
      );

  # The editable venv points at the live checkout, so it uses an
  # UNFILTERED workspace rooted at a real path — mkEditablePyprojectOverlay
  # computes relative paths via lib.path.splitRoot, which rejects the
  # filtered pythonSrc (a cleanSourceWith set, not a path).  Filtering
  # buys nothing here anyway: the editable install reads from
  # $HERMES_PYTHON_SRC_ROOT at runtime.
  workspaceRoot = ./..;
  editableWorkspace = uv2nix.lib.workspace.loadWorkspace { inherit workspaceRoot; };
  editableOverlay = editableWorkspace.mkEditablePyprojectOverlay {
    root = "$HERMES_PYTHON_SRC_ROOT"; # resolved at shellHook time
  };

  editableSet = pythonSet.overrideScope (
    lib.composeManyExtensions [
      editableOverlay
      (final: prev: {
        hermes-agent = prev.hermes-agent.overrideAttrs (old: {
          # point straight at the real source instead of the filtered nix store copy
          src = workspaceRoot;
          nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
        });
      })
    ]
  );
in
{
  inherit python;

  # Equivalent to uv's --only-group: use the lock-derived dependency spec,
  # without installing Hermes or its runtime dependencies in the build env.
  iconBuildVenv = pythonSet.mkVirtualEnv "hermes-icon-build-env"
    pythonSet.hermes-agent.dependency-groups.icon-build;

  venv = pythonSet.mkVirtualEnv "hermes-agent-env" {
    hermes-agent = dependency-groups;
  };
  editableVenv = editableSet.mkVirtualEnv "hermes-agent-editable-env" {
    hermes-agent = dependency-groups;
  };
}
