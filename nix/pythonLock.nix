# Nix uses the Python family pinned by PM, not a second version constant.
{
  lib,
  pkgs,
}:
let
  pythonFamily =
    lockfile:
    let
      version = (builtins.fromJSON (builtins.readFile lockfile)).packages.python.version;
      parts = builtins.match "([0-9]+)\\.([0-9]+)(\\..*)?" version;
    in
    if parts == null then
      throw "packages.python.version '${version}' is not a Python <major>.<minor> version — cannot derive the Nix interpreter from it"
    else
      builtins.elemAt parts 0 + "." + builtins.elemAt parts 1;

  # Select pkgs.python3NN by family ("3.14" -> pkgs.python314). A missing
  # lookup throws; there is no default interpreter to fall back to.
  selectPython =
    family: packageSet:
    let
      name = "python" + lib.replaceStrings [ "." ] [ "" ] family;
      interp = packageSet.${name} or null;
    in
    if interp == null then
      throw "package set does not provide ${name}, but pm/lock.json pins Python ${family} — update the nixpkgs input; Nix will not silently substitute another Python"
    else
      interp;

  lockfile = ../pm/lock.json;
  family = pythonFamily lockfile;
in
{
  inherit pythonFamily selectPython;

  inherit family;

  interpreter = selectPython family pkgs;

  # The interpreter the lock itself installs. uv2nix cannot build a venv
  # against it (a tarball interpreter has no `pkgs`), so the venv keeps
  # `interpreter` while a shell that must match a real runtime takes this.
  # Derived here, not passed in: every caller of this file reads `interpreter`
  # and nothing else should have to know about the pinned build.
  pinned = (pkgs.callPackage ./pm-packages.nix { }).python;
}
