# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }:
{
  perSystem =
    { pkgs, inputs', ... }:
    let
      hermesAgent = pkgs.callPackage ./hermes-agent.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
        npm-lockfile-fix = inputs'.npm-lockfile-fix.packages.default;
        # Only embed clean revs — dirtyRev doesn't represent any upstream
        # commit, so comparing it would always claim "update available".
        rev = inputs.self.rev or null;
      };
    in
    {
      packages = {
        default = hermesAgent;
        tui = hermesAgent.hermesTui;
        web = hermesAgent.hermesWeb;

        fix-lockfiles = hermesAgent.hermesNpmLib.mkFixLockfiles {
          packages = [ hermesAgent.hermesTui hermesAgent.hermesWeb ];
        };
      };
    };
}
