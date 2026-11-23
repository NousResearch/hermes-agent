# nix/lib.nix — Shared helpers for nix stuff
#
# All JS packages in this repo are pnpm workspace members sharing a single
# root pnpm-lock.yaml.  buildPnpmPackage provides the shared pnpmDeps,
# pnpm and pnpmConfigHook so individual .nix files don't duplicate them.
#
# Source filters (pythonSrc, per-package npm srcs) reduce rebuild scope so
# that e.g. a .tsx change doesn't trigger a Python venv rebuild, and a .py
# change doesn't trigger a TUI/Web/Desktop rebuild.  Each derivation gets a
# filtered src that only includes files it actually needs, while keeping
# the repo-root directory layout intact for pnpmConfigHook workspace
# resolution.
#
# buildPnpmPackage returns packageJsonPath (e.g. "ui-tui/package.json")
# instead of a per-package devShellHook.  The root devshell hook
# (mkPnpmDevShellHook) stamps all package.json paths and pnpm-workspace.yaml,
# and if any changed, runs a single `pnpm install --lockfile-only` from
# root to update the lockfile, then `pnpm install --frozen-lockfile` if
# the lockfile changed.
{
  lib,
  stdenv,
  fetchPnpmDeps,
  pnpmConfigHook,
  nodejs-slim,
  writeShellScriptBin,
  writeShellScript,
  coreutils,
  callPackage,
  runCommand,
  # pm/lock.json pins, used for node, npm and pnpm.
  pmPackages,
}:
let
  repoRoot = ./..;

  # Every JS tool comes from pm/lock.json — the node the builds and the
  # shipped app run on, npm wrapped onto it, and the pnpm that installs the
  # workspaces. nixpkgs supplies nothing here; node/npm/pnpm have Linux
  # artifacts in the lock.
  pmNode = pmPackages.node;
  npm12 = callPackage ./npm-pinned.nix { nodejs = pmNode; };
  # nixpkgs' pnpm packages build from the `pnpm` npm package, which since
  # pnpm 11 is only a launcher that downloads the native executable, and the
  # pinned nixpkgs has no pnpm_12 anyway; pm's pnpm is that executable.
  # fetchPnpmDeps runs its store normalizer (pnpm-fixup-state-db) on
  # `pnpm.nodejs-slim`, which only nixpkgs' pnpm carries. That helper only
  # rewrites the fetched store's index, so nixpkgs' node runs it.
  pnpm = pmPackages.pnpm // {
    inherit nodejs-slim;
  };
  node_gyp_11_4_0 = callPackage ./node-gyp-11-4-0.nix { nodejs = nodejs; };
  # bin/npm and bin/npx are the pinned npm; the rest of the tree (node,
  # include/node headers, lib/node_modules/node's own bundled npm) is the
  # pinned node. A plain symlinkJoin cannot express that: both paths carry
  # bin/npm, so assemble it explicitly.
  nodejs = runCommand "nodejs-26-npm-12" {
    # lib.getExe reads meta.mainProgram; the pm derivation carries no meta.
    meta.mainProgram = "node";
  } ''
    cp -rs ${pmNode} $out
    # cp -r copies the store's read-only directory modes; make the farm's own
    # directories writable so bin/npm and bin/npx can be replaced. Only real
    # directories are chmodded, never the symlink targets.
    find $out -type d -exec chmod u+w {} +
    rm -f $out/bin/npm $out/bin/npx
    ln -s ${npm12}/bin/npm $out/bin/npm
    ln -s ${npm12}/bin/npx $out/bin/npx
  '';

  # ── pnpm workspace discovery ───────────────────────────────────────
  # Single source of truth: the `packages` list of pnpm-workspace.yaml.
  # Everything below (workspace package.json discovery, the Python
  # source's JS-dir exclusions) is derived from this so the topology is
  # never duplicated.  Add a workspace to pnpm-workspace.yaml and the nix
  # build picks it up automatically.  Nix reads no YAML: this parses the
  # one shape the file uses (a top-level `packages:` key followed by
  # `  - <dir>` / `  - <dir>/*` lines) and throws on anything else, so a
  # reformatted list cannot silently drop a workspace.
  workspaceYamlLines = lib.splitString "\n" (builtins.readFile (repoRoot + "/pnpm-workspace.yaml"));
  workspacePatterns =
    let
      start = lib.lists.findFirstIndex (line: line == "packages:") (builtins.length workspaceYamlLines) workspaceYamlLines;
      block = lib.drop (start + 1) workspaceYamlLines;
      count = lib.lists.findFirstIndex (line: !lib.hasPrefix "  - " line) (builtins.length block) block;
      items = lib.take count block;
      next = builtins.head (lib.drop count block ++ [ "" ]);
      patterns = map (lib.removePrefix "  - ") items;
      supported = pattern: builtins.match "[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*(/\\*)?" pattern != null;
    in
    if items == [ ] || lib.hasPrefix " " next || lib.hasPrefix "-" next || !(lib.all supported patterns) then
      throw "nix/lib.nix: pnpm-workspace.yaml `packages:` is no longer a block of unquoted `  - <dir>` or `  - <dir>/*` lines; update the parser"
    else
      patterns;

  # Expand a workspace glob (e.g. "apps/*") into concrete member dirs
  # relative to the repo root.  Only trailing "*" globs are supported —
  # that's all pnpm-workspace.yaml uses here.  Literal patterns (e.g.
  # "ui-tui") pass through unchanged.
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
    lib.concatMap expandWorkspace workspacePatterns
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
          # JS/TS workspace directories — derived from the pnpm workspaces
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
          "pnpm-lock.yaml"
          "pnpm-workspace.yaml"
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
      else if lib.hasPrefix "scripts/build/" relPath && lib.hasSuffix ".mjs" relPath then
        # Frontend recipes belong to their product source filters, not Python.
        false
      else if builtins.elem relPath excludedFiles then
        false
      else if builtins.elem topComponent excludedDirs then
        false
      else
        true;
  };

  # Common pnpm workspace resolution files needed by all pnpm builds.
  # A frozen install requires every workspace package.json to match its
  # lockfile importer, and pnpm-workspace.yaml carries the overrides and
  # settings the lockfile was resolved with.  Discovered from
  # pnpm-workspace.yaml — root manifests + every member's package.json.
  npmWorkspaceFiles = lib.fileset.unions (
    [
      (repoRoot + "/package.json")
      (repoRoot + "/pnpm-lock.yaml")
      (repoRoot + "/pnpm-workspace.yaml")
    ]
    ++ map (d: repoRoot + "/${d}/package.json") workspaceMemberDirs
  );

  # pnpm deps source: just what fetchPnpmDeps needs (root manifests +
  # workspace member package.jsons).  Much smaller than the full repo,
  # so changing source files won't invalidate the pnpmDeps derivation.
  pnpmDepsSrc = lib.fileset.toSource {
    root = repoRoot;
    fileset = npmWorkspaceFiles;
  };

  # The pnpm store for the whole lockfile, shared by all members; each
  # build installs only its own slice of it.  nixpkgs has no hashless pnpm
  # importer, so this is a fixed-output derivation pinned by
  # nix/pnpm-deps-hash: after a pnpm-lock.yaml change, run
  # `nix run .#update-pnpm-lockfile` (CI prints the expected hash too).
  pnpmDeps = fetchPnpmDeps {
    pname = "hermes-agent";
    src = pnpmDepsSrc;
    inherit pnpm;
    # pnpm >= 11 keeps the store index in SQLite, which only fetcher
    # version 4 dumps reproducibly.
    fetcherVersion = 4;
    # fetchPnpmDeps passes --registry="$NIX_NPM_REGISTRY", which is empty
    # unless the Nix daemon sets it; pnpm 12 takes an empty registry
    # literally and hangs retrying relative URLs.
    # pnpm checks the root engines.node (engineStrict) against the node on
    # PATH, where fetchPnpmDeps puts nixpkgs' node first: install with the
    # pinned node every build uses.
    prePnpmInstall = ''
      export NIX_NPM_REGISTRY=''${NIX_NPM_REGISTRY:-https://registry.npmjs.org/}
      export PATH=${pmNode}/bin:$PATH
    '';
    hash = lib.fileContents ./pnpm-deps-hash;
  };

  # Build a per-package pnpm source: workspace resolution files + the
  # package's own directory tree(s).  Source ROOT is always the repo
  # root, preserving the workspace layout that pnpmConfigHook expects.
  # Callers pass the dirs they need (relative to the repo root), so each
  # package owns its own source scope.
  testFileFilter = lib.fileset.fileFilter (file: lib.hasInfix ".test." file.name) repoRoot;
  mkNpmSrc =
    dirs:
    lib.fileset.toSource {
      root = repoRoot;
      fileset = lib.fileset.difference (lib.fileset.union npmWorkspaceFiles (
        lib.fileset.unions (map (d: repoRoot + "/${d}") dirs)
      )) testFileFilter;
    };

  # Returns a stdenv.mkDerivation with the workspace's pnpm deps installed.

  # `dirs` is the single source of truth for what the package contains:
  # its first entry is the package's own folder (→ packageJsonPath), and
  # all entries scope the filtered src.  Packages that import source from
  # another workspace member (workspace: deps) must list that member's dir
  # too, e.g. apps/desktop depends on apps/shared.
  #
  # Usage:
  #   hermesNpmLib.buildPnpmPackage {
  #     dirs = [ "apps/desktop" "apps/shared" ];
  #     buildPhase = '' ... '';
  #     installPhase = '' ... '';
  #   }
  customBuildPnpmPackage =
    { dirs, ... }@attrs:
    let
      # The package's own folder is the first dir; it carries the
      # package.json that names the derivation.
      folder = builtins.head dirs;

      # Read package.json from the repo (the filtered src is a store path, but we can read the original)
      packageJson = builtins.fromJSON (builtins.readFile (repoRoot + "/${folder}/package.json"));
      defaultPname = packageJson.name or "unknown";
      defaultVersion = packageJson.version or "0.0.0";

      common = {
        inherit pnpmDeps;
        # No sourceRoot — the workspace root (with the single pnpm-lock.yaml)
        # is auto-detected as sourceRoot by nix, so pnpmConfigHook finds
        # the lockfile there.
        src = mkNpmSrc dirs;
        # Only this member and the workspace members it depends on (pnpm
        # always adds the root project's own dependencies). The braces make
        # `...` select dependencies; pnpm reads a bare `./dir...` as that
        # dir alone.
        pnpmWorkspaces = [ "{./${folder}}..." ];
        ELECTRON_SKIP_BINARY_DOWNLOAD = 1;
        passthru = {
          packageJsonPath = "${folder}/package.json";
        };
      };

      # Remove `dirs` from the passed attrs (mkDerivation doesn't need it)
      attrsWithoutDirs = removeAttrs attrs [ "dirs" ];

      finalAttrs =
        common
        // attrsWithoutDirs
        // {
          pname = attrs.pname or defaultPname;
          version = attrs.version or defaultVersion;
          nativeBuildInputs = [
            nodejs
            pnpm
            pnpmConfigHook
          ]
          ++ attrs.nativeBuildInputs or [ ];
        };
    in
    stdenv.mkDerivation finalAttrs;
in
{
  inherit pythonSrc nodejs pnpm pnpmDeps;
  node-gyp = node_gyp_11_4_0;

  # Bring the shared root lockfile up to date with the manifests, refresh
  # the pnpmDeps hash for it, and verify all pnpm packages still build.
  # Exposed as a runnable package — `nix run .#update-pnpm-lockfile` — so
  # it's actually usable, unlike a bin buried in a build sandbox's PATH.
  # All workspace packages share one lockfile, so there's a single script
  # (not one per package).
  updatePnpmLockfile = writeShellScriptBin "update-pnpm-lockfile" ''
    set -euo pipefail
    # DEBUG=1 nix run .#update-pnpm-lockfile — trace every command
    [ -n "''${DEBUG:-}" ] && set -x

    REPO_ROOT=$(git rev-parse --show-toplevel)
    cd "$REPO_ROOT"

    PATH=${nodejs}/bin:$PATH ${lib.getExe' pnpm "pnpm"} install --lockfile-only

    # pnpmDeps is pinned by one fixed-output hash: build it with an empty
    # hash and take the real one from Nix's mismatch report.
    : > nix/pnpm-deps-hash
    log=$(mktemp)
    trap 'rm -f "$log"' EXIT
    nix build .#pnpm-deps --no-link 2>&1 | tee "$log" || true
    hash=$(sed -n 's/^ *got: *\(sha256-[A-Za-z0-9+/=]*\)$/\1/p' "$log")
    if [ -z "$hash" ]; then
      echo "update-pnpm-lockfile: no hash in the nix build output above" >&2
      exit 1
    fi
    echo "$hash" > nix/pnpm-deps-hash

    nix build .#tui .#web .#webapp .#desktop
    echo "Lockfile and pnpmDeps hash updated and all pnpm packages built."
  '';

  buildPnpmPackage = customBuildPnpmPackage;

  # Single devshell hook for all pnpm workspace packages.
  #
  # Takes a list of package.json relative paths (from buildPnpmPackage .passthru.packageJsonPath),
  # stamps all of them, and if any changed:
  #   1. Runs `pnpm install --lockfile-only` from root to update the lockfile
  #   2. If the lockfile changed, runs `pnpm install --frozen-lockfile`
  mkPnpmDevShellHook =
    packageJsonPaths:
    writeShellScript "pnpm-dev-hook" ''
      REPO_ROOT=$(git rev-parse --show-toplevel)

      # Stamp all workspace package.jsons, and pnpm-workspace.yaml (its
      # overrides, catalog and allowBuilds shape the lock too), into one file.
      STAMP_DIR=".nix-stamps"
      STAMP="$STAMP_DIR/npm-package-jsons"
      STAMP_VALUE=$(
        ${coreutils}/bin/sha256sum "$REPO_ROOT/pnpm-workspace.yaml" ${
          lib.concatMapStringsSep " " (p: "\"$REPO_ROOT/${p}\"") packageJsonPaths
        } 2>/dev/null | ${coreutils}/bin/sort | ${coreutils}/bin/sha256sum | awk '{print $1}'
      )

      PKG_CHANGED=false
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        PKG_CHANGED=true
        echo "pnpm: package.json or pnpm-workspace.yaml changed, updating lockfile..."
        ( cd "$REPO_ROOT" && ${lib.getExe' pnpm "pnpm"} install --lockfile-only --silent 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$STAMP_VALUE" > "$STAMP"
      fi

      # Check if lockfile changed (either from the pnpm install above or
      # from an external edit).  Runs a frozen pnpm install if so.
      LOCK_STAMP="$STAMP_DIR/root-lockfile"
      LOCK_STAMP_VALUE=$(sha256sum "$REPO_ROOT/pnpm-lock.yaml" 2>/dev/null | awk '{print $1}')
      if [ ! -f "$LOCK_STAMP" ] || [ "$(cat "$LOCK_STAMP")" != "$LOCK_STAMP_VALUE" ]; then
        echo "pnpm: pnpm-lock.yaml changed, running pnpm install --frozen-lockfile..."
        ( cd "$REPO_ROOT" && CI=true ${lib.getExe' pnpm "pnpm"} install --frozen-lockfile --trust-lockfile --silent 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$LOCK_STAMP_VALUE" > "$LOCK_STAMP"
      fi
    '';
}
