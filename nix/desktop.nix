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
  python3,
  ...
}:
let
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
    ];
    pname = "hermes-desktop-renderer";

    doCheck = true;

    buildPhase = ''
      runHook preBuild

      mkdir -p apps/desktop/build

      patchShebangs .

      pushd apps/desktop
        # typecheck :3
        npm exec -- tsc -b

        # build the renderer bundle
        # vite's emptyOutDir wipes dist/ on every run
        # so it has to be first
        npm exec -- vite build

        # build the electron bundle
        node scripts/bundle-electron-main.mjs

        # Compile node-pty against Electron's actual ABI (the nixpkgs
        # `electron` we ship). Headers come from a pinned fetchurl input
        # since the sandbox has no network here, so node-gyp's
        # normal --disturl download path can't run.
        mkdir -p "$TMPDIR/electron-headers"
        tar -xzf ${electronHeaders} -C "$TMPDIR/electron-headers" --strip-components=1

        ${lib.getExe hermesNpmLib.node-gyp} rebuild \
          --directory=../../node_modules/node-pty \
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
  };

  # Generate Info.plist for the macOS .app bundle (XML plist format).
  infoPlist = pkgs.writeText "Info.plist" ''
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>CFBundleDevelopmentRegion</key>
      <string>en</string>
      <key>CFBundleDisplayName</key>
      <string>Hermes</string>
      <key>CFBundleExecutable</key>
      <string>Hermes</string>
      <key>CFBundleIconFile</key>
      <string>icon.icns</string>
      <key>CFBundleIdentifier</key>
      <string>com.nousresearch.hermes</string>
      <key>CFBundleInfoDictionaryVersion</key>
      <string>6.0</string>
      <key>CFBundleName</key>
      <string>Hermes</string>
      <key>CFBundlePackageType</key>
      <string>APPL</string>
      <key>CFBundleShortVersionString</key>
      <string>${renderer.version}</string>
      <key>CFBundleVersion</key>
      <string>${renderer.version}</string>
      <key>LSApplicationCategoryType</key>
      <string>public.app-category.developer-tools</string>
      <key>LSArchitecturePriority</key>
      <array>
        <string>arm64</string>
      </array>
      <key>LSMinimumSystemVersion</key>
      <string>12.0</string>
      <!-- LaunchServices-only env for the bundle (does not apply to direct
           CLI exec — $out/bin/hermes-desktop wraps the binary for that).
           HERMES_DESKTOP_HERMES points the desktop's resolver step 4 at the
           fully-wrapped nix hermes: venv with all deps, skills, plugins,
           runtime PATH (ripgrep/git/ffmpeg/etc). -->
      <key>LSEnvironment</key>
      <dict>
        <key>HERMES_DESKTOP_HERMES</key>
        <string>${lib.getExe hermesAgent}</string>
        <key>ELECTRON_IS_DEV</key>
        <string>0</string>
      </dict>
      <!-- Usage strings required by the hardened-runtime audio-input
           entitlement (voice feature) — mirrors electron-builder's
           extendInfo in apps/desktop/package.json. -->
      <key>NSAudioCaptureUsageDescription</key>
      <string>Hermes uses audio capture for voice conversations.</string>
      <key>NSMicrophoneUsageDescription</key>
      <string>Hermes uses the microphone for voice input and voice conversations.</string>
      <key>NSHighResolutionCapable</key>
      <true/>
      <key>NSHumanReadableCopyright</key>
      <string>Copyright Nous Research</string>
      <key>NSPrincipalClass</key>
      <string>AtomApplication</string>
      <key>NSSupportsAutomaticGraphicsSwitching</key>
      <true/>
    </dict>
    </plist>
  '';
in

# Electron wrapper: nixpkgs' electron binary pointed at the renderer dir.
# On Darwin: creates a proper .app bundle under $out/Applications/Hermes.app/
# with the renamed Electron Mach-O binary as CFBundleExecutable (kept a real
# binary — not a wrapper script — so the bundle can be codesigned and
# notarized) and env delivered via Info.plist LSEnvironment.  A thin wrapper
# at $out/bin/hermes-desktop covers direct CLI exec (`nix run`), where
# LSEnvironment does not apply.
# On Linux: flat $out/share/hermes-desktop/ layout (unchanged).
stdenv.mkDerivation {
  pname = "hermes-desktop";
  inherit (renderer) version;

  dontUnpack = true;
  dontBuild = true;

  nativeBuildInputs = [
    makeWrapper
    python3
  ];

  installPhase = if stdenv.hostPlatform.isDarwin then ''
    runHook preInstall

    # Create the Applications directory first
    mkdir -p $out/Applications

    # Copy the entire nixpkgs Electron.app structure to get Frameworks and helper apps
    cp -r ${electron}/Applications/Electron.app $out/Applications/Hermes.app
    chmod -R u+w $out/Applications/Hermes.app

    # Rename the main binary from Electron to Hermes
    mv $out/Applications/Hermes.app/Contents/MacOS/Electron \
       $out/Applications/Hermes.app/Contents/MacOS/Hermes

    # Rename helper apps in Frameworks
    for helper in $out/Applications/Hermes.app/Contents/Frameworks/Electron\ Helper*.app; do
      if [ -d "$helper" ]; then
        newname=$(basename "$helper" | sed 's/Electron/Hermes/g')
        mv "$helper" "$(dirname "$helper")/$newname"
        # Rename the binary inside the helper
        for bin in "$(dirname "$helper")/$newname/Contents/MacOS/"*; do
          if [ -f "$bin" ]; then
            newbin=$(basename "$bin" | sed 's/Electron/Hermes/g')
            mv "$bin" "$(dirname "$bin")/$newbin"
          fi
        done
      fi
    done

    # Each helper's own Info.plist still says "Electron Helper (...)" /
    # com.github.Electron.helper internally — rebrand that too, not just the
    # path. (Temp file instead of `sed -i`: BSD vs GNU sed disagree on it.)
    for plist in "$out/Applications/Hermes.app/Contents/Frameworks/"Hermes\ Helper*.app/Contents/Info.plist; do
      sed \
        -e 's/Electron Helper/Hermes Helper/g' \
        -e 's/com\.github\.Electron\.helper/com.nousresearch.hermes.helper/g' \
        "$plist" > "$plist.new"
      mv "$plist.new" "$plist"
    done

    # Update Info.plist with our custom values
    cp ${infoPlist} $out/Applications/Hermes.app/Contents/Info.plist

    # Copy the app icon to Resources
    cp ${../apps/desktop/assets/icon.icns} $out/Applications/Hermes.app/Contents/Resources/

    # Put our renderer files in Resources/app/ (Electron expects app here)
    mkdir -p $out/Applications/Hermes.app/Contents/Resources/app
    cp -r ${renderer}/* $out/Applications/Hermes.app/Contents/Resources/app/

    # The runtime reads install-stamp.json via process.resourcesPath (=
    # Contents/Resources in a real .app bundle) or APP_ROOT/build/ — copying
    # the renderer into Resources/app/ would hide it from both lookups, so
    # install the stamp at Contents/Resources separately.
    cp ${renderer}/install-stamp.json \
      $out/Applications/Hermes.app/Contents/Resources/install-stamp.json

    # CLI entry point for `nix run` / profile installs.  The bundle itself
    # gets its env from Info.plist LSEnvironment, which only applies to
    # LaunchServices launches — direct exec of the Mach-O binary needs this
    # wrapper.  It lives outside the .app on purpose: codesign seals
    # Contents/ only, and CFBundleExecutable must stay the Mach-O binary
    # above for signing/notarization to work.
    mkdir -p $out/bin
    makeWrapper $out/Applications/Hermes.app/Contents/MacOS/Hermes \
      $out/bin/hermes-desktop \
      --add-flags "$out/Applications/Hermes.app/Contents/Resources/app" \
      --set HERMES_DESKTOP_HERMES "${lib.getExe hermesAgent}" \
      --set ELECTRON_IS_DEV 0

    # Renaming/editing the bundle above leaves every Mach-O in it unsigned
    # (nixpkgs' electron binaries only carry a lightweight adhoc,
    # linker-signed signature that never covered Info.plist/Resources
    # anyway). Ad-hoc (`-s -`, no cert/Apple account needed) matches
    # nixpkgs' own convention for this exact situation (see e.g. the
    # `opencode` and `lmstudio` derivations, and stdenv's own fixup-phase
    # sign() in pkgs/os-specific/darwin/by-name/si/signingUtils). Sign
    # leaf-first — each helper, then the main binary, then the bundle
    # itself — rather than `--deep`, which nixpkgs avoids for the same
    # reason Apple's docs discourage it.
    for helper in "$out/Applications/Hermes.app/Contents/Frameworks/"Hermes\ Helper*.app; do
      codesign --force --sign - "$helper"
    done
    codesign --force --sign - "$out/Applications/Hermes.app/Contents/MacOS/Hermes"
    codesign --force --sign - "$out/Applications/Hermes.app"

    runHook postInstall
  '' else ''
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

    # XDG launcher entry
    mkdir -p $out/share/applications $out/share/icons/hicolor/1024x1024/apps
    install -m 0644 ${../apps/desktop/assets/icon.png} \
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
