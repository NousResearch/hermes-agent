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
  npm = hermesNpmLib.mkNpmPassthru { folder = "apps/desktop"; attr = "desktop"; pname = "hermes-desktop"; };

  packageJson = builtins.fromJSON (builtins.readFile (npm.src + "/apps/desktop/package.json"));
  version = packageJson.version;

  # Build the renderer (dist/ + electron/ + package.json).
  renderer = pkgs.buildNpmPackage (npm // {
    pname = "hermes-desktop-renderer";
    inherit version;

    doCheck = false;
    # The workspace lockfile resolves all peer deps
    # correctly so --legacy-peer-deps is not needed.
    # --ignore-scripts comes from mkNpmPassthru (shared).
    makeCacheWritable = true;

    buildPhase = ''
      runHook preBuild

      # write-build-stamp.cjs replacement.  Packaged Electron reads this
      # at first-launch to pin the install.ps1 git ref; informational in
      # nix builds (the backend comes from the derivation directly).
      mkdir -p apps/desktop/build
      echo '{"schemaVersion":1,"commit":"nix","branch":"nix","dirty":false,"source":"nix"}' > apps/desktop/build/install-stamp.json

      # Build from apps/desktop/ so vite.config.ts resolves correctly.
      # The workspace root's node_modules/ is accessible as ../../node_modules/.
      cd apps/desktop

      # vite handles TS transpilation via esbuild — no type-checking.
      # We skip `tsc -b` to avoid type errors in test files that don't
      # ship in the bundle (real upstream peer-dep version mismatches
      # in @testing-library/react v16 — not blocking the build).
      # Call vite directly from root node_modules to avoid npx resolving
      # through unpatched workspace symlinks.
      node ../../node_modules/vite/bin/vite.js build --outDir dist

      # Return to source root so installPhase paths are correct.
      cd ../..

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      # vite writes to apps/desktop/dist/ (we cd'd there in buildPhase).
      # apps/desktop/build was created before the cd.  electron/ is source.
      cp -r apps/desktop/dist $out/
      cp -r apps/desktop/electron $out/
      cp -r apps/desktop/build $out/
      cp apps/desktop/package.json $out/
      runHook postInstall
    '';
  });

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
      <string>${version}</string>
      <key>CFBundleVersion</key>
      <string>${version}</string>
      <key>LSApplicationCategoryType</key>
      <string>public.app-category.developer-tools</string>
      <key>LSArchitecturePriority</key>
      <array>
        <string>arm64</string>
      </array>
      <key>LSMinimumSystemVersion</key>
      <string>12.0</string>
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
# with Contents/MacOS/Hermes wrapped binary and symlink at $out/bin/hermes-desktop.
# On Linux: flat $out/share/hermes-desktop/ layout (unchanged).
stdenv.mkDerivation {
  pname = "hermes-desktop";
  inherit version;

  dontUnpack = true;
  dontBuild = true;

  nativeBuildInputs = [ makeWrapper ];

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

    # Update Info.plist with our custom values
    cp ${infoPlist} $out/Applications/Hermes.app/Contents/Info.plist

    # Copy the app icon to Resources
    cp ${npm.src}/apps/desktop/assets/icon.icns $out/Applications/Hermes.app/Contents/Resources/

    # Put our renderer files in Resources/app/ (Electron expects app here)
    mkdir -p $out/Applications/Hermes.app/Contents/Resources/app
    cp -r ${renderer}/* $out/Applications/Hermes.app/Contents/Resources/app/

    # Wrap the renamed binary with environment variables
    # First rename the original binary so the wrapper can call it
    mv $out/Applications/Hermes.app/Contents/MacOS/Hermes \
       $out/Applications/Hermes.app/Contents/MacOS/Hermes.real

    # HERMES_DESKTOP_HERMES tells the desktop's resolver step 4 to use our
    # fully-wrapped nix hermes — venv with all deps, skills, plugins, and
    # runtime PATH (ripgrep/git/ffmpeg/etc).
    makeWrapper $out/Applications/Hermes.app/Contents/MacOS/Hermes.real \
      $out/Applications/Hermes.app/Contents/MacOS/Hermes \
      --add-flags "$out/Applications/Hermes.app/Contents/Resources/app" \
      --set HERMES_DESKTOP_HERMES "${lib.getExe hermesAgent}" \
      --set ELECTRON_IS_DEV 0

    # Symlink bin/hermes-desktop to the wrapped binary inside the bundle
    mkdir -p $out/bin
    ln -s ../Applications/Hermes.app/Contents/MacOS/Hermes $out/bin/hermes-desktop

    runHook postInstall
  '' else ''
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
