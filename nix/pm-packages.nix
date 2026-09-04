{
  lib,
  stdenv,
  fetchurl,
  unzip,
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

  derive = name: pin: artifacts:
    stdenv.mkDerivation {
      pname = name;
      version = pin.version;

      srcs = map (artifact: fetchurl {
        url = artifact.url;
        sha256 = artifact.sha256;
      }) artifacts;

      # pm's store publishes the unpacked tree; mirror that shape. Several
      # archives unpack over one another into the same root, exactly as
      # pm merges them into one store entry.
      sourceRoot = ".";
      nativeBuildInputs =
        lib.optional (lib.any (a: lib.hasSuffix ".zip" a.url) artifacts) unzip;
      dontBuild = true;
      dontConfigure = true;

      installPhase = ''
        mkdir -p $out
        cp -r . $out/
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
