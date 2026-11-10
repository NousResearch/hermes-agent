# Run the shared generator offline; generated assets are not tracked in git.
{ lib, runCommand, venv }:
let
  src = lib.fileset.toSource {
    root = ./..;
    fileset = lib.fileset.unions [
      ../scripts/generate_icons.py
      (lib.fileset.fileFilter (file: file.hasExt "svg") ../assets)
      # The DMG volume icon is hand-made raster artwork, not rendered from the svgs.
      ../assets/dmg-volume.png
    ];
  };
in
runCommand "hermes-icons" { nativeBuildInputs = [ venv ]; } ''
  python ${src}/scripts/generate_icons.py --source ${src} --out $out
  python ${src}/scripts/generate_icons.py --source ${src} --out $out --check
''
