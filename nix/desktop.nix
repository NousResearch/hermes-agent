# nix/desktop.nix — Hermes Desktop (Electron) app build + wrapper
#
# `hermesAgent` is the fully-built `.#default` package — it ships the
# `hermes` binary with the venv, runtime PATH, bundled skills/plugins, etc.
# already wired up.  We point the desktop at it via the existing
# `HERMES_DESKTOP_HERMES` override env var, so the desktop's resolver
# uses our fully wrapped binary at step 4 ("existing Hermes CLI").
# No reimplementation of the agent resolution in this wrapper.
{
  pkgs,
  lib,
  stdenv,
  makeWrapper,
  hermesNpmLib,
  electron,
  hermesAgent,
  ...
}:
let
  npm = hermesNpmLib.mkNpmPassthru {
    folder = "apps/desktop";
    attr = "desktop";
    pname = "hermes-desktop";
  };

  packageJson = builtins.fromJSON (builtins.readFile (npm.src + "/apps/desktop/package.json"));
  version = packageJson.version;

  electronHeaders = pkgs.fetchurl {
    url = "https://artifacts.electronjs.org/headers/dist/v${electron.version}/node-v${electron.version}-headers.tar.gz";
    sha256 = "sha256-zi/QMwRZ0+FwE9XTE+DiSIeJXAwxmLKEaBWD5W3pMOI=";
  };

  # node-pty ships no Electron-tagged prebuild we can trust to match this
  # exact nixpkgs electron version, so it's always compiled from source
  # against Electron's own headers (not whatever Node ran `npm`).
  targetPlatform =
    if stdenv.hostPlatform.isDarwin then
      "darwin"
    else if stdenv.hostPlatform.isLinux then
      "linux"
    else
      throw "hermes-desktop: unsupported host platform for node-pty staging";

  targetArch =
    if stdenv.hostPlatform.isAarch64 then
      "arm64"
    else if stdenv.hostPlatform.isx86_64 then
      "x64"
    else
      throw "hermes-desktop: unsupported host arch for node-pty staging";

  # Build the renderer (dist/ + electron/ + package.json).
  renderer = pkgs.buildNpmPackage (
    npm
    // {
      pname = "hermes-desktop-renderer";
      inherit version;
      doCheck = true;

      buildPhase = ''
        runHook preBuild

        mkdir -p apps/desktop/build

        patchShebangs .

        pushd apps/desktop
          # typecheck :3
          npm exec tsc -b

          # build the renderer bundle
          # vite's emptyOutDir wipes dist/ on every run
          # so it has to be first
          npm exec vite build

          # build the electron bundle
          node scripts/bundle-electron-main.mjs

          # Compile node-pty against Electron's actual ABI (the nixpkgs
          # `electron` we ship). Headers come from a pinned fetchurl input
          # since the sandbox has no network here, so node-gyp's
          # normal --disturl download path can't run.
          mkdir -p "$TMPDIR/electron-headers"
          tar -xzf ${electronHeaders} -C "$TMPDIR/electron-headers" --strip-components=1

          npm rebuild node-pty \
            --build-from-source \
            --runtime=electron \
            --target=${electron.version} \
            --nodedir="$TMPDIR/electron-headers" \
            --disturl="" \
            --offline

          # Target platform/arch come from stdenv.hostPlatform, not the
          # build host's own process.platform/arch.
          node scripts/stage-native-deps.mjs ${targetPlatform} ${targetArch}
        popd

        runHook postBuild
      '';

      checkPhase = ''
        runHook preCheck

        pushd apps/desktop

          npm run postbuild

          # validate staged node-pty native binary is present.
          STAGED_PTY_NODE="./dist/node_modules/node-pty/build/Release/pty.node"

          if [ ! -f "$STAGED_PTY_NODE" ]; then
            echo "FATAL: Missing staged node-pty native binary at $STAGED_PTY_NODE"
            echo "node-pty must be compiled natively"
            exit 1
          fi

        popd

        runHook postCheck
      '';

      installPhase = ''
        runHook preInstall
        mkdir -p $out
        # vite writes to apps/desktop/dist/ (we cd'd there in buildPhase).
        # stage-native-deps.mjs stages node-pty into dist/node_modules/node-pty,
        # so copying dist/ wholesale carries the native dep along with the
        # esbuild bundle that require()s it. apps/desktop/build was created
        # before the cd.
        cp -rn apps/desktop/dist $out/

        echo '{"schemaVersion":1,"commit":"nix-dummy-commit","branch":"nix","dirty":false,"source":"nix"}' > $out/install-stamp.json

        cp -n apps/desktop/package.json $out/
        runHook postInstall
      '';
    }
  );
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

    # Standard nixpkgs pattern for electron-builder apps: patch process.resourcesPath
    # to point to the app's directory. In Nix, unpackaged electron defaults this
    # to the electron distribution's resources path, breaking extraResources lookups.
    substituteInPlace $out/share/hermes-desktop/dist/electron-main.mjs \
      --replace-fail "process.resourcesPath" "'$out/share/hermes-desktop'"

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

  passthru = {
    inherit (renderer.passthru) packageJsonPath;
  };

  meta = with lib; {
    description = "Native Electron desktop shell for Hermes Agent";
    homepage = "https://github.com/NousResearch/hermes-agent";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "hermes-desktop";
  };
}
