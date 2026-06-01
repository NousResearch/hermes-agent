# nix/desktop.nix — Hermes Desktop (Electron) app build + wrapper
#
# `hermesAgent` is the fully-built `.#default` package — it ships the
# `hermes` binary with the venv, runtime PATH, bundled skills/plugins, etc.
# already wired up.  We point the desktop at it via the existing
# `HERMES_DESKTOP_HERMES` override env var, so the desktop's resolver
# uses our fully wrapped binary at step 4 ("existing Hermes CLI").
# No reimplementation of the agent resolution in this wrapper.
{ pkgs, lib, stdenv, makeWrapper, hermesNpmLib, electron, hermesAgent, ... }:
let
  src = ../apps;
  npmDeps = pkgs.fetchNpmDeps {
    src = ../apps/desktop;
    # buildNpmPackage uses `npm ci` which is strict — peer deps not in the
    # lockfile cause network fetch attempts.  Fetcher v2 stages the full
    # cache (including peer-only deps) so `npm ci` can resolve them offline.
    fetcherVersion = 2;
    hash = "sha256-7W9ObYz08yDMtybY8+RkUXkKVsJXINLl0qBUB91hpao=";
  };

  npm = hermesNpmLib.mkNpmPassthru { folder = "apps/desktop"; attr = "desktop"; pname = "hermes-desktop"; };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/desktop/package.json"));
  version = packageJson.version;

  # Build the renderer (dist/ + electron/ + package.json).
  renderer = pkgs.buildNpmPackage (npm // {
    pname = "hermes-desktop-renderer";
    inherit src npmDeps version;
    sourceRoot = "apps/desktop";

    doCheck = false;
    # buildNpmPackage uses `npm ci` which fails on peer deps not in the
    # lockfile.  npmDepsFetcherVersion=2 stages the full cache (peer deps
    # included) so the offline `npm ci` resolves them.
    npmDepsFetcherVersion = 2;
    # `--ignore-scripts` skips the electron prebuild download (we use nixpkgs
    # electron instead).  `--legacy-peer-deps` matches the dev workflow —
    # apps/desktop has conflicting peer deps (zod, @testing-library) that
    # the package.json relies on npm 7+ to relax.
    npmFlags = [ "--ignore-scripts" "--legacy-peer-deps" ];
    makeCacheWritable = true;

    buildPhase = ''
      runHook preBuild

      # write-build-stamp.cjs replacement.  Packaged Electron reads this
      # at first-launch to pin the install.ps1 git ref; informational in
      # nix builds (the backend comes from the derivation directly).
      mkdir -p build
      echo '{"schemaVersion":1,"commit":"nix","branch":"nix","dirty":false,"source":"nix"}' > build/install-stamp.json

      # The vite config aliases react/react-dom to ../../node_modules/react
      # (workspace root, where npm dedups them in dev).  In the standalone
      # nix build there is no workspace root, so the deps are installed
      # locally — rewrite the aliases to point at the local copy.
      substituteInPlace vite.config.ts \
        --replace-quiet '../../node_modules/' './node_modules/'

      # vite handles TS transpilation via esbuild — no type-checking.
      # We skip `tsc -b` to avoid type errors in test files that don't
      # ship in the bundle (real upstream peer-dep version mismatches
      # in @testing-library/react v16 — not blocking the build).
      npx vite build --outDir dist

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      cp -r dist electron build $out/
      cp package.json $out/
      runHook postInstall
    '';
  });
in

# Electron wrapper: nixpkgs' electron binary pointed at the renderer dir.
stdenv.mkDerivation {
  pname = "hermes-desktop";
  inherit version;

  dontUnpack = true;
  dontBuild = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/hermes-desktop $out/bin
    cp -r ${renderer}/* $out/share/hermes-desktop/

    # Wrap the nixpkgs electron binary to launch our app.  Set
    # HERMES_DESKTOP_HERMES to the absolute path of the nix-built `hermes`
    # binary so the desktop's resolver step 4 ("existing Hermes CLI on
    # PATH") uses our fully wrapped binary — venv with all deps,
    # bundled skills/plugins, runtime PATH (ripgrep/git/ffmpeg/etc).
    # No reimplementation of the agent resolver in the wrapper.
    makeWrapper ${lib.getExe electron} $out/bin/hermes-desktop \
      --add-flags "$out/share/hermes-desktop" \
      --set HERMES_DESKTOP_HERMES "${lib.getExe hermesAgent}" \
      --set ELECTRON_IS_DEV 0

    runHook postInstall
  '';

  meta = with lib; {
    description = "Native Electron desktop shell for Hermes Agent";
    homepage = "https://github.com/NousResearch/hermes-agent";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "hermes-desktop";
  };
}
