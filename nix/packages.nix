# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }:
{
  perSystem =
    { pkgs, lib, inputs', ... }:
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

        # +33 MB over default — ships discord.py, python-telegram-bot, slack-sdk
        # so `nix profile install .#messaging` users get them out-of-the-box
        # without hitting the read-only /nix/store lazy-install path.
        messaging = hermesAgent.override {
          extraDependencyGroups = [ "messaging" ];
        };

        # +704 MB over default — all optional integrations pre-built.
        # matrix is excluded on Darwin (oqs/liboqs lacks aarch64-darwin wheels).
        full = hermesAgent.override {
          extraDependencyGroups = [
            "messaging" "dingtalk" "feishu" "anthropic" "bedrock" "azure-identity"
            "edge-tts" "tts-premium" "voice" "exa" "firecrawl" "parallel-web" "fal"
            "honcho" "hindsight" "modal" "daytona" "vercel"
          ] ++ lib.optionals pkgs.stdenv.isLinux [ "matrix" ];
        };

        tui = hermesAgent.hermesTui;
        web = hermesAgent.hermesWeb;

        fix-lockfiles = hermesAgent.hermesNpmLib.mkFixLockfiles {
          packages = [ hermesAgent.hermesTui hermesAgent.hermesWeb ];
        };
      };
    };
}
