{
  lib,
  stdenv,
  fetchurl,
  unzip,
  autoPatchelfHook,
  patchelf,
}:
# The thin translation layer: pm/lock.json -> one fetched, unpacked
# derivation per package for this nix system. Nothing here knows versions,
# urls, or hashes — the lockfile is the complete machine interface, the
# same one `pm install` and `pm bundle` consume.
#
# Consumers pick what they need:   (callPackage ./pm-packages.nix { }).ripgrep
# Packages with no artifact for this system are simply absent.
let
  lock = builtins.fromJSON (builtins.readFile ../pm/lock.json);
  mirror = builtins.fromJSON (builtins.readFile ../pm/artifact-mirror.json);

  target =
    let
      arch = if stdenv.hostPlatform.isAarch64 then "arm64" else "x64";
      os =
        if stdenv.hostPlatform.isDarwin then "darwin"
        else if stdenv.hostPlatform.isLinux then "linux"
        else "win32";
    in
    "${os}-${arch}";

  # A target pins one artifact or a list of them (a runtime split across
  # archives that must land in one directory). Normalize to a list.
  artifactsFor =
    pin:
    let
      found = pin.artifacts.${target} or pin.artifacts.any or null;
    in
    if found == null then null else if builtins.isList found then found else [ found ];

  # The archive suffixes pm/store.py::extract unpacks. A pin may also list
  # provenance sidecars (checksums.txt, .sig/.pem/.asc) that pm verifies at
  # install time; the fixed-output hash already pins the archive here, and
  # stdenv's unpackPhase has no unpacker for them, so they are left out.
  isArchive = artifact:
    lib.any (suffix: lib.hasSuffix suffix artifact.url) [ ".tar.gz" ".tgz" ".tar.xz" ".txz" ".tar.bz2" ".zip" ];

  # python-build-standalone finds its own libpython through an $ORIGIN/../lib
  # RPATH. autoPatchelf rewrites that into a literal path and leaves the
  # interpreter unable to load its own stdlib, so for it the ELF interpreter
  # and RPATH are set explicitly: $ORIGIN stays, and Nix's glibc is appended.
  originRpath = name: name == "python";

  derive = name: pin: artifacts:
    stdenv.mkDerivation {
      pname = name;
      version = pin.version;

      srcs = map (artifact: fetchurl {
        urls = [ artifact.url "${mirror.origin}/${mirror.prefix}${artifact.sha256}" ];
        sha256 = artifact.sha256;
      }) (builtins.filter isArchive artifacts);

      # pm's store publishes the unpacked tree; mirror that shape. Several
      # archives unpack over one another into the same root, exactly as
      # pm merges them into one store entry.
      sourceRoot = ".";
      nativeBuildInputs =
        lib.optionals (originRpath name) [ patchelf ]
        ++ lib.optional (!originRpath name) autoPatchelfHook
        ++ lib.optional (lib.any (a: lib.hasSuffix ".zip" a.url) artifacts) unzip;
      # The upstream tarballs are glibc-linked (interpreter
      # /lib64/ld-linux-x86-64.so.2, which only exists on a host with nix-ld).
      # Their whole dependency set is glibc + libstdc++/libgcc/libatomic, so
      # autoPatchelfHook rewrites the interpreter and RPATH at build time and
      # every consumer runs them on a plain NixOS host. ripgrep ships musl-
      # static and is left untouched.
      buildInputs = [ stdenv.cc.cc.lib ];
      # Keep the RPATH written below byte for byte; the fixup shrinker cannot
      # tell that $ORIGIN/../lib is load-bearing.
      dontPatchELF = originRpath name;
      postFixup = lib.optionalString (originRpath name) ''
        for exe in $out/bin/*; do
          if [ -f "$exe" ] && patchelf --print-interpreter "$exe" >/dev/null 2>&1; then
            patchelf \
              --set-interpreter ${stdenv.cc.bintools.dynamicLinker} \
              --set-rpath '$ORIGIN/../lib:${stdenv.cc.libc}/lib' \
              "$exe"
          fi
        done
      '';
      dontBuild = true;
      dontConfigure = true;

      installPhase = ''
        mkdir -p $out
        cp -r . $out/

        # Archives wrap their tree in one versioned directory
        # (`uv-x86_64-unknown-linux-gnu/`, `node-v26.12.1-linux-x64/`, `package/`).
        # pm's store strips it; mirror that so `$out/bin` is the entry point
        # instead of `$out/<archive-name>/bin`. `env-vars` is Nix's own file
        # in $out by the time installPhase runs, so it never counts as content.
        shopt -s dotglob nullglob
        wrapped=() stray=()
        for entry in "$out"/*; do
          [ "$(basename "$entry")" = env-vars ] && continue
          if [ -d "$entry" ]; then wrapped+=("$entry"); else stray+=("$entry"); fi
        done
        if [ ''${#wrapped[@]} -eq 1 ] && [ ''${#stray[@]} -eq 0 ]; then
          mv "''${wrapped[0]}"/* "$out"/
          rmdir "''${wrapped[0]}"
        fi
        shopt -u dotglob nullglob

        # Entry points also disagree: node and ffmpeg carry `bin/`, uv and
        # ripgrep put their executables at the root. PATH composition
        # (mkShell, lib.makeBinPath) only reads `bin`, so link root-level
        # executables into it.
        mkdir -p $out/bin
        for exe in "$out"/*; do
          [ -f "$exe" ] && [ -x "$exe" ] || continue
          ln -sf "../$(basename "$exe")" "$out/bin/$(basename "$exe")"
        done
      '';
    };
in
lib.filterAttrs (_: v: v != null) (
  lib.mapAttrs (
    name: pin:
    let
      artifacts = artifactsFor pin;
    in
    if artifacts == null then null else derive name pin artifacts
  ) lock.packages
)
