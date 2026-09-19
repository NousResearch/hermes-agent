# nix/lib.nix — Shared helpers for nix stuff
#
# All npm packages in this repo are workspace members sharing a single
# root package-lock.json.  mkNpmPassthru provides the shared npmDeps,
# npmRoot, and npmConfigHook so individual .nix files don't duplicate them.
#
# Source filters (pythonSrc, per-package npm srcs) reduce rebuild scope so
# that e.g. a .tsx change doesn't trigger a Python venv rebuild, and a .py
# change doesn't trigger a TUI/Web/Desktop rebuild.  Each derivation gets a
# filtered src that only includes files it actually needs, while keeping
# the repo-root directory layout intact for buildNpmPackage /
# npmConfigHook workspace resolution.
#
# mkNpmPassthru returns packageJsonPath (e.g. "ui-tui/package.json")
# instead of a per-package devShellHook.  The root devshell hook
# (mkNpmDevShellHook) collects all package.json paths, stamps them,
# and if any changed, runs a single `npm i --package-lock-only` from
# root to update the lockfile, then `npm ci` if the lockfile changed.
{
  lib,
  npm-lockfile-fix,
  importNpmLock,
  writeShellScriptBin,
  writeShellScript,
  coreutils,
  callPackage,
  nodejs_26,
  symlinkJoin,
  buildNpmPackage,
  runCommand,
}:
let
  repoRoot = ./..;

  npm12 = callPackage ./npm-12-0-2.nix { };
  node_gyp_11_4_0 = callPackage ./node-gyp-11-4-0.nix { };
  nodejs_26_npm_12 = symlinkJoin {
    name = "nodejs-26-npm-12";
    paths = [
      npm12
      nodejs_26
    ];
    inherit (nodejs_26) meta passthru;
  };

  nodejs = nodejs_26_npm_12;

  # Patched hook: just a new derivation that copies and patches the script
  patchedNpmConfigHook = runCommand "npm-config-hook-patched" { } ''
    mkdir -p $out/nix-support
    # Copy all support files from the original hook
    cp -r ${importNpmLock.npmConfigHook}/nix-support/* $out/nix-support/

    # Change the node gyp config var to avoid the warning with npm12
    # Replace the node-gyp path with the newer one that supports the new config var
    substituteInPlace $out/nix-support/setup-hook \
      --replace-fail 'npm_config_nodedir' 'npm_package_config_node_gyp_nodedir' \
      --replace-fail 'npm_config_node_gyp' 'npm_config_node_gyp=${node_gyp_11_4_0}/bin/node-gyp'
  '';

  # ── npm workspace discovery ────────────────────────────────────────
  # Single source of truth: the `workspaces` field of the root
  # package.json.  Everything below (workspace package.json discovery,
  # the Python source's JS-dir exclusions) is derived from this so the
  # topology is never duplicated.  Add a workspace to package.json and
  # the nix build picks it up automatically.
  rootPackageJson = builtins.fromJSON (builtins.readFile (repoRoot + "/package.json"));

  # Expand a workspace glob (e.g. "apps/*") into concrete member dirs
  # relative to the repo root.  Only trailing "*" globs are supported —
  # that's all npm uses here.  Literal patterns (e.g. "ui-tui") pass
  # through unchanged.
  expandWorkspace =
    pattern:
    let
      parts = lib.splitString "/" pattern;
    in
    if lib.last parts == "*" then
      let
        parent = lib.concatStringsSep "/" (lib.init parts);
        entries = builtins.readDir (repoRoot + "/${parent}");
        dirs = lib.filterAttrs (_: t: t == "directory") entries;
      in
      map (d: "${parent}/${d}") (builtins.attrNames dirs)
    else
      [ pattern ];

  # All workspace member directories (relative paths), filtered to those
  # that actually carry a package.json — a glob like apps/* may match a
  # dir that isn't really a package.
  workspaceMemberDirs = builtins.filter (d: builtins.pathExists (repoRoot + "/${d}/package.json")) (
    lib.concatMap expandWorkspace rootPackageJson.workspaces
  );

  # Top-level directory of each workspace member, deduplicated.  Used to
  # exclude JS/TS workspace trees from the Python source filter.  E.g.
  # apps/desktop + apps/shared + ui-tui + web → [ "apps" "ui-tui" "web" ].
  jsWorkspaceTopDirs = lib.unique (
    map (d: builtins.head (lib.splitString "/" d)) workspaceMemberDirs
  );

  # ── Source filters for reducing rebuild scope ──────────────────────
  # Changing a .tsx/.mjs file should NOT trigger a Python venv rebuild,
  # and changing a .py file should NOT trigger a TUI/Web/Desktop rebuild.

  # Python source: everything except JS/TS/docs/infra directories.
  pythonSrc = lib.cleanSourceWith {
    src = repoRoot;
    name = "hermes-python-source";
    filter =
      path: type:
      let
        relPath = lib.removePrefix (toString repoRoot + "/") (toString path);
        components = lib.splitString "/" relPath;
        topComponent = if components == [ ] then "" else builtins.head components;
        excludedDirs =
          # JS/TS workspace directories — derived from the npm workspaces
          # so a new workspace member is excluded from the Python source
          # without touching this list.
          jsWorkspaceTopDirs ++ [
            # Documentation
            "docs"
            "website"
            # CI/infra
            "docker"
            ".github"
            # Content/examples
            "infographic"
            "datagen-config-examples"
            # unused packaging infra
            "packaging"
            # Test infrastructure
            "tests"
            # Plan/temp files
            "plans"
            # Nix build definitions (Python build doesn't need these)
            "nix"
            # Skills are shipped via HERMES_BUNDLED_SKILLS /
            # HERMES_OPTIONAL_SKILLS (see hermes-agent.nix), not via the
            # wheel's data_files — setup.py's _data_file_tree returns []
            # for a missing dir, so the wheel builds fine without them.
            # This keeps SKILL.md edits from rebuilding the Python venv.
            "skills"
            "optional-skills"
            # locales/ and optional-mcps/ are bare data dirs (no
            # __init__.py) shipped via symlinks + HERMES_BUNDLED_LOCALES
            # / HERMES_OPTIONAL_MCPS, not via the wheel. Excluding them
            # keeps catalog edits from rebuilding the Python venv.
            "locales"
            "optional-mcps"
          ];
        excludedFiles = [
          # JS root manifests
          "package.json"
          "package-lock.json"
          # Docker files
          "Dockerfile"
          "docker-compose.yml"
          "docker-compose.windows.yml"
          # Nix build definitions — editing the flake shouldn't rebuild
          # the venv.  (Input changes rebuild regardless, via the lock.)
          "flake.nix"
          "flake.lock"
          # Root docs the wheel doesn't consume.  README.md and LICENSE
          # must stay — pyproject.toml references them (readme /
          # license-files).
          "AGENTS.md"
          "CONTRIBUTING.md"
          "SECURITY.md"
          "README.zh-CN.md"
          ".gitignore"
          "setup-hermes.sh"
        ];
      in
      if relPath == "" then
        true
      else if builtins.elem relPath excludedFiles then
        false
      else if builtins.elem topComponent excludedDirs then
        false
      else
        true;
  };

  # Common npm workspace resolution files needed by all npm builds.
  # npm ci requires all workspace package.json files to resolve
  # workspace: protocol dependencies correctly.  Discovered from the
  # root package.json workspaces — root manifests + every member's
  # package.json.
  npmWorkspaceFiles = lib.fileset.unions (
    [
      (repoRoot + "/package.json")
      (repoRoot + "/package-lock.json")
    ]
    ++ map (d: repoRoot + "/${d}/package.json") workspaceMemberDirs
  );

  # npm deps source: just what importNpmLock needs (root manifests +
  # workspace member package.jsons).  Much smaller than the full repo,
  # so changing source files won't invalidate the npmDeps derivation.
  npmDepsSrc = lib.fileset.toSource {
    root = repoRoot;
    fileset = npmWorkspaceFiles;
  };

  # npm dependencies for the workspace, shared by all members. importNpmLock
  # resolves each package from the lockfile's own `integrity` hashes, so the
  # lockfile is the single source of truth — no separate dependency hash to
  # keep in sync with it.
  npmDeps = importNpmLock.importNpmLock {
    npmRoot = npmDepsSrc;
  };

  # Build a per-package npm source: workspace resolution files + the
  # package's own directory tree(s).  Source ROOT is always the repo
  # root, preserving the workspace layout that buildNpmPackage and
  # npmConfigHook expect.  Callers pass the dirs they need (relative to
  # the repo root), so each package owns its own source scope.
  testFileFilter = lib.fileset.fileFilter (file: lib.hasInfix ".test." file.name) repoRoot;
  mkNpmSrc =
    dirs:
    lib.fileset.toSource {
      root = repoRoot;
      fileset = lib.fileset.difference (lib.fileset.union npmWorkspaceFiles (
        lib.fileset.unions (map (d: repoRoot + "/${d}") dirs)
      )) testFileFilter;
    };

  # Returns a buildNpmPackage-compatible function.

  # `dirs` is the single source of truth for what the package contains:
  # its first entry is the package's own folder (→ packageJsonPath), and
  # all entries scope the filtered src.  Packages that import source from
  # another workspace member (file: deps) must list that member's dir too,
  # e.g. apps/desktop depends on apps/shared.
  #
  # Usage:
  #   hermesNpmLib.buildNpmPackage {
  #     dirs = [ "apps/desktop" "apps/shared" ];
  #     buildPhase = '' ... '';
  #     installPhase = '' ... '';
  #   }
  customBuildNpmPackage =
    { dirs, ... }@attrs:
    let
      # The package's own folder is the first dir; it carries the
      # package.json that buildNpmPackage reads.
      folder = builtins.head dirs;

      # Read package.json from the repo (the filtered src is a store path, but we can read the original)
      packageJson = builtins.fromJSON (builtins.readFile (repoRoot + "/${folder}/package.json"));
      defaultPname = packageJson.name or "unknown";
      defaultVersion = packageJson.version or "0.0.0";

      common = {
        inherit nodejs npmDeps;
        # No sourceRoot — the workspace root (with the single package-lock.json)
        # is auto-detected as sourceRoot by nix. npmRoot stays at "."
        # so npmConfigHook finds the lockfile there.
        src = mkNpmSrc dirs;
        npmConfigHook = patchedNpmConfigHook;
        npmRoot = ".";
        ELECTRON_SKIP_BINARY_DOWNLOAD = 1;
        passthru = {
          packageJsonPath = "${folder}/package.json";
        };
      };

      # Remove `dirs` from the passed attrs (buildNpmPackage doesn't need it)
      attrsWithoutDirs = removeAttrs attrs [ "dirs" ];

      finalAttrs =
        common
        // attrsWithoutDirs
        // {
          pname = attrs.pname or defaultPname;
          version = attrs.version or defaultVersion;
        };
    in
    buildNpmPackage finalAttrs;
in
{
  inherit pythonSrc nodejs;
  node-gyp = node_gyp_11_4_0;

  # Regenerate the shared root lockfile from scratch and verify all npm
  # packages still build.  Exposed as a runnable package — `nix run
  # .#update-npm-lockfile` — so it's actually usable, unlike a bin buried
  # in a build sandbox's PATH.  All workspace packages share one lockfile,
  # so there's a single script (not one per package).
  updateNpmLockfile = writeShellScriptBin "update-npm-lockfile" ''
    set -euo pipefail
    # DEBUG=1 nix run .#update-npm-lockfile — trace every command
    [ -n "''${DEBUG:-}" ] && set -x

    REPO_ROOT=$(git rev-parse --show-toplevel)
    cd "$REPO_ROOT"

    rm -rf node_modules/
    ${lib.getExe' nodejs "npm"} cache clean --force
    CI=true ${lib.getExe' nodejs "npm"} install --workspaces
    ${lib.getExe npm-lockfile-fix} ./package-lock.json

    # importNpmLock reads hashes from the lockfile itself — rebuild every
    # npm package to verify the new lockfile resolves offline.
    nix build .#tui .#web .#desktop
    echo "Lockfile updated and all npm packages built."
  '';

  buildNpmPackage = customBuildNpmPackage;

  # Single devshell hook for all npm workspace packages.
  #
  # Takes a list of package.json relative paths (from mkNpmPassthru .passthru.packageJsonPath),
  # stamps all of them, and if any changed:
  #   1. Runs `npm i --package-lock-only` from root to update the lockfile
  #   2. If the lockfile changed, runs `npm ci`
  mkNpmDevShellHook =
    packageJsonPaths:
    writeShellScript "npm-dev-hook" ''
      REPO_ROOT=$(git rev-parse --show-toplevel)

      # Stamp all workspace package.jsons into one file.
      STAMP_DIR=".nix-stamps"
      STAMP="$STAMP_DIR/npm-package-jsons"
      STAMP_VALUE=$(
        ${coreutils}/bin/sha256sum ${
          lib.concatMapStringsSep " " (p: "\"$REPO_ROOT/${p}\"") packageJsonPaths
        } 2>/dev/null | ${coreutils}/bin/sort | ${coreutils}/bin/sha256sum | awk '{print $1}'
      )

      PKG_CHANGED=false
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        PKG_CHANGED=true
        echo "npm: package.json changed, updating lockfile..."
        ( cd "$REPO_ROOT" && ${lib.getExe' nodejs "npm"} i --package-lock-only --silent --no-fund --no-audit 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$STAMP_VALUE" > "$STAMP"
      fi

      # Check if lockfile changed (either from the npm i above or from an
      # external edit).  Runs npm ci if so.
      LOCK_STAMP="$STAMP_DIR/root-lockfile"
      LOCK_STAMP_VALUE=$(sha256sum "$REPO_ROOT/package-lock.json" 2>/dev/null | awk '{print $1}')
      if [ ! -f "$LOCK_STAMP" ] || [ "$(cat "$LOCK_STAMP")" != "$LOCK_STAMP_VALUE" ]; then
        echo "npm: package-lock.json changed, running npm ci..."
        ( cd "$REPO_ROOT" && CI=true ${lib.getExe' nodejs "npm"} ci --silent --no-fund --no-audit 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$LOCK_STAMP_VALUE" > "$LOCK_STAMP"
      fi
    '';
}
