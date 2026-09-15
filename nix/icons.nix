# Run the shared generator offline; generated assets are not tracked in git.
{ lib, runCommand, iconBuildVenv }:
let
  src = lib.fileset.toSource {
    root = ./..;
    fileset = lib.fileset.unions [
      ../scripts/generate_icons.py
      (lib.fileset.fileFilter (file: file.hasExt "svg") ../assets)
    ];
  };
in
runCommand "hermes-icons" { nativeBuildInputs = [ iconBuildVenv ]; } ''
  python ${src}/scripts/generate_icons.py --source ${src} --out $out
  python ${src}/scripts/generate_icons.py --source ${src} --out $out --check
''
