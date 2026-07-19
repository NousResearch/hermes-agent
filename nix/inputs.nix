# nix/inputs.nix — Stable-Nix access to the non-nixpkgs build inputs.
#
# The flake pins uv2nix / pyproject-nix / build-system-pkgs / npm-lockfile-fix
# in flake.lock. This file reads that same lock and fetches each input by its
# locked rev + narHash. Flake and non-flake consumers therefore share ONE source
# of truth — the lock — with no duplicated revisions to keep in sync.
#
# Node names are resolved through the ROOT node's `inputs` map, never by
# indexing `lock.nodes` with the logical name: when an input appears more than
# once in the graph, Nix suffixes the node keys (`pyproject-nix_2`, `uv2nix_2`,
# …) and the bare name refers to a *transitive* pin, not the flake's own input.
#
# Each value is shaped to match the corresponding flake input, so downstream
# files (python.nix, lib.nix, …) consume them identically whether wired from the
# flake or from here:
#   uv2nix.lib.workspace ...                 (flake `lib` output)
#   pyproject-nix.build.{hacks,packages} ... (flake `build` output)
#   pyproject-build-systems.overlays.default (flake `overlays.default` output)
#   npm-lockfile-fix                         (a derivation, as `getExe` needs)
#
# Stable-Nix usage:
#   let inputs = import ./nix/inputs.nix { inherit pkgs; }; in ...
{
  pkgs ? import <nixpkgs> { },
}:
let
  inherit (pkgs) lib;

  lock = builtins.fromJSON (builtins.readFile ../flake.lock);

  # Logical input name -> node key, via the root node. See header: indexing
  # `lock.nodes` directly by the logical name can resolve a transitive pin.
  rootInputs = lock.nodes.${lock.root}.inputs;

  # Fetch a locked github input by rev, verified against its narHash. Using the
  # lock's narHash keeps the fetch pure and identical to what the flake fetched.
  fetchLocked =
    name:
    let
      node = lock.nodes.${rootInputs.${name}}.locked;
    in
    assert node.type == "github";
    builtins.fetchTarball {
      url = "https://github.com/${node.owner}/${node.repo}/archive/${node.rev}.tar.gz";
      sha256 = node.narHash;
    };

  # pyproject.nix default.nix: { lib } -> { lib, build, packages }
  pyproject-nix = import (fetchLocked "pyproject-nix") { inherit lib; };

  # uv2nix default.nix: { lib, pyproject-nix } -> { lib }
  uv2nix = import (fetchLocked "uv2nix") { inherit lib pyproject-nix; };

  # build-system-pkgs default.nix: { lib, uv2nix, pyproject-nix }
  #   -> { sdist, wheel, default }. The flake exposes this overlay as
  #   `overlays.default`, so reshape to match for drop-in consumption.
  pyproject-build-systems = {
    overlays.default =
      (import (fetchLocked "pyproject-build-systems") {
        inherit lib uv2nix pyproject-nix;
      }).default;
  };

  # npm-lockfile-fix ships no classic entrypoint (flake-only), but it is a plain
  # setuptools console app, so build it directly from the locked source. lib.nix
  # calls `lib.getExe` on this, so it must be a derivation with a mainProgram —
  # never null.
  npm-lockfile-fix = pkgs.python3Packages.buildPythonApplication {
    pname = "npm-lockfile-fix";
    version = "0.1.0";
    src = fetchLocked "npm-lockfile-fix";
    format = "setuptools";
    propagatedBuildInputs = [ pkgs.python3Packages.requests ];
    doCheck = false;
    meta.mainProgram = "npm-lockfile-fix";
  };
in
{
  inherit
    pyproject-nix
    uv2nix
    pyproject-build-systems
    npm-lockfile-fix
    ;
}
