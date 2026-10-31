# nix/mkHermesAgent.nix — the one place the package's input wiring lives
#
# `nix/packages.nix` builds through this, and so does any downstream flake
# (`flake.lib.mkHermesAgent`), so the fork and its consumers cannot drift
# into two different wiring sets.
#
# Flake-level, not perSystem: `inputs'` only exists inside perSystem, so each
# flake input is resolved for the caller's system here.
#
# NO IFD. `src` is a source tree — the flake's own source by default, or a
# caller's `builtins.path` of an already-patched checkout — so nothing below
# reads or imports from a derivation. A `patches` argument would force a
# build to learn the version, which is the wall nix-config already hit.
{ inputs }:
{
  pkgs,
  # Source tree to build. A flake set (the default) carries provenance;
  # a plain path carries none and gets the neutral values.
  src ? inputs.self,
  # The optional groups this variant ships; the user's additions go on top.
  defaultDependencyGroups ? [ ],
  extraDependencyGroups ? [ ],
  extraPythonPackages ? [ ],
  # Commits since the tagged revision, for the "+N.g<sha>" display version.
  distance ? 0,
}:
let
  # Provenance for install-stamp.json. `inputs.self` exposes these as
  # top-level attributes (sourceInfo is only a store path, not a record); a
  # plain path exposes none, so every field falls back to its
  # hermes-agent.nix default rather than inventing a sentinel that could be a
  # real value. A flake URL's requested ref is not exposed as self.ref once
  # Nix resolves it to an immutable source, so `branch` is normally null for
  # remote flakes.
  source = if builtins.isAttrs src then src else { };
  dirtyRevision = source.dirtyRev or null;
  rev =
    source.rev or (if dirtyRevision != null then builtins.substring 0 40 dirtyRevision else null);
  rawRef = source.ref or null;
  branch = if rawRef != null then builtins.replaceStrings [ "refs/heads/" ] [ "" ] rawRef else null;
  dirty = dirtyRevision != null;
  lastModified = source.lastModified or null;

  # pm/lock.json translated into one derivation per tool. callPackage adds
  # .override/.overrideDerivation to the returned SET, so keep only the real
  # derivations.
  pmPackages = pkgs.lib.filterAttrs (_: pkgs.lib.isDerivation) (
    pkgs.callPackage "${src}/nix/pm-packages.nix" { }
  );
in
pkgs.callPackage "${src}/nix/hermes-agent.nix" {
  inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
  npm-lockfile-fix = inputs.npm-lockfile-fix.packages.${pkgs.stdenv.hostPlatform.system}.default;
  inherit pmPackages;

  # Passed explicitly so `override (prev: ...)` sees them: an argument default
  # never lands in `prev`, so a function-form override that appends to
  # `prev.extraDependencyGroups` would hit a missing attribute.
  inherit
    defaultDependencyGroups
    extraDependencyGroups
    extraPythonPackages
    distance
    rev
    branch
    dirty
    lastModified
    ;

  # `version` is deliberately not passed: nix/hermes-agent.nix derives it
  # from `__release_date__` (patch/nix-version-from-release-date), and a
  # second copy of that logic here would be a second version constant.
}
