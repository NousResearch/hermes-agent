# nix/overlay.nix — Nixpkgs overlay exposing pkgs.hermes-agent.
#
# A plain overlay: `final: prev: { … }`. Use it directly, no call needed —
#
#   nixpkgs.overlays = [ (import ./nix/overlay.nix) ];
#
# All build inputs come from nix/inputs.nix, which reads the same flake.lock the
# flake uses — including npm-lockfile-fix, which hermes-agent.nix needs to build
# the bundled TUI/web assets it always installs. A stable (non-flake) consumer
# therefore needs nothing but this repo. The flake re-exports this overlay
# verbatim.
#
# `rev` is the one flake-only concern: it embeds the locked flake revision for
# the update check, and a stable consumer has no flake revision.
final: _prev:
let
  inherit (import ./inputs.nix { pkgs = final; })
    uv2nix
    pyproject-nix
    pyproject-build-systems
    npm-lockfile-fix
    ;
in
{
  hermes-agent = final.callPackage ./hermes-agent.nix {
    inherit
      uv2nix
      pyproject-nix
      pyproject-build-systems
      npm-lockfile-fix
      ;
    rev = null;
  };
}
