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
  installStampFile,
  generatedIcons,
  python3,
  # Environment to bake into the launcher. A GUI launcher reads none of the
  # shell profile, so a variable that an interactive shell exports does not
  # reach an app that the desktop menu starts. The Home Manager module passes
  # HERMES_HOME and HERMES_MANAGED here, which gives the app the same state
  # directory as the services.
  extraEnv ? { },
  # Shell lines to run before the app starts. A secret belongs here and never
  # in extraEnv: makeWrapper writes a --set value into the Nix store, which
  # all users can read. A --run line reads the value from a runtime path at
  # each start instead.
  extraRun ? [ ],
  ...
}:
let
  # Each flag goes on its own continued line, and the leading backslash is
  # inside the generated string. An empty attribute set then adds no text at
  # all, and cannot leave a backslash above a blank line. That fault ends the
  # makeWrapper command early, and the next flag runs as a shell command.
  extraEnvFlags = lib.concatMapStrings (
    name: " \\\n      --set ${name} ${lib.escapeShellArg (toString extraEnv.${name})}"
  ) (lib.attrNames extraEnv);

  extraRunFlags = lib.concatMapStrings (line: " \\\n      --run ${lib.escapeShellArg line}") extraRun;

  electronHeaders = pkgs.fetchurl {
    url = "https://artifacts.electronjs.org/headers/dist/v${electron.version}/node-v${electron.version}-headers.tar.gz";
    sha256 = "sha256-f8bSbLRmtbP93CJAvEBs+sHWDZ1xP2bcpLhC1EnOmZU=";
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
  renderer = hermesNpmLib.buildNpmPackage {
    dirs = [
      "apps/desktop"
      "apps/shared"
      "scripts/build/desktop.mjs"
      "scripts/build/frontend-common.mjs"
    ];
    pname = "hermes-desktop-renderer";

    doCheck = true;

    buildPhase = ''
      runHook preBuild

      patchShebangs .

      # The native provider runs before compilation. Use the headers for
      # the exact Electron runtime shipped by this derivation, offline.
      mkdir -p "$TMPDIR/electron-headers"
      tar -xzf ${electronHeaders} -C "$TMPDIR/electron-headers" --strip-components=1
      ${lib.getExe hermesNpmLib.node-gyp} rebuild \
        --directory=node_modules/node-pty \
        --build-from-source \
        --runtime=electron \
        --target=${electron.version} \
        --arch=${targetArch} \
        --nodedir="$TMPDIR/electron-headers" \
        --disturl="" \
        --offline

      node apps/desktop/scripts/stage-native-deps.mjs \
        --source "$PWD" --out "$TMPDIR/desktop-native-deps" \
        --platform ${targetPlatform} --arch ${targetArch}
      node scripts/build/desktop.mjs \
        --source "$PWD" --out "$PWD/apps/desktop/dist" \
        --icons ${generatedIcons} --stamp ${installStampFile} \
        --native-deps "$TMPDIR/desktop-native-deps" \
        --platform ${targetPlatform} --typecheck

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
      # The shared product contains renderer, main/preload, and native deps.
      cp -rn apps/desktop/dist $out/

      cp ${installStampFile} $out/install-stamp.json

      cp -n apps/desktop/package.json $out/
      runHook postInstall
    '';
  };
in

# Electron wrapper: nixpkgs' electron binary pointed at the renderer dir.
stdenv.mkDerivation {
  pname = "hermes-desktop";
  inherit (renderer) version;

  dontUnpack = true;
  dontBuild = true;

  nativeBuildInputs = [
    makeWrapper
    python3
  ];

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
      --set ELECTRON_IS_DEV 0${extraEnvFlags}${extraRunFlags}

    # XDG launcher entry
    mkdir -p $out/share/applications $out/share/icons/hicolor/1024x1024/apps
    install -m 0644 ${generatedIcons}/apps/desktop/assets/icon.png \
      $out/share/icons/hicolor/1024x1024/apps/hermes.png
    export PYTHONPATH=$(mktemp -d)
    cp ${../hermes_cli/linux_desktop_entry.py} "$PYTHONPATH/linux_desktop_entry.py"
    export DESKTOP_EXEC="$out/bin/hermes-desktop"
    export DESKTOP_ICON="$out/share/icons/hicolor/1024x1024/apps/hermes.png"
    python3 -c 'import os; from linux_desktop_entry import render_desktop_entry; print(render_desktop_entry(os.environ["DESKTOP_EXEC"], os.environ["DESKTOP_ICON"]))' > $out/share/applications/hermes.desktop
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
