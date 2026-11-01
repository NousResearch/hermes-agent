# nix/exports.nix — the flake.lib surface
#
# Consumers build the package without touching input plumbing:
#   hermes = inputs.hermes.lib.mkHermesAgent { pkgs = nixpkgs.legacyPackages.${system}; };
#   hermes.override { extraPythonPackages = [ pkgs.python3Packages.requests ]; }
# `pkgs.callPackage`'s makeOverridable travels through, so .override keeps
# working on whatever this returns.
#
# `defaultDependencyGroups` is the set of optional groups the shipped package
# carries. It lives here, next to the builder, so the flake's own
# `packages.<system>.default` and `overlays.default` resolve the same list
# against their own `pkgs` instead of repeating it.
{ inputs, ... }:
{
  flake.lib.mkHermesAgent = import ./mkHermesAgent.nix { inherit inputs; };

  flake.lib.defaultDependencyGroups =
    pkgs:
    [
      "anthropic"
      "azure-identity"
      "bedrock"
      "daytona"
      "dingtalk"
      "edge-tts"
      "exa"
      "fal"
      "feishu"
      "firecrawl"
      "messaging"
      "modal"
      "parallel-web"
      "tts-premium"
      "vercel"
      "voice"
    ]
    # matrix is Linux-only (oqs/liboqs lacks aarch64-darwin wheels).
    ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ "matrix" ];
}
