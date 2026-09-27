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
}
