# nix/packages.nix — Hermes Agent package set in package-normal-form.
#
# Stable Nix (no flakes):
#   let pkgs = import <nixpkgs> { }; in (import ./nix/packages.nix { inherit pkgs; }).default
# Every build input defaults to nix/inputs.nix, which reads the same flake.lock
# the flake uses — so a stable consumer needs nothing but this repo.
#
# The flake passes its own locked inputs explicitly, so existing flake users get
# byte-identical derivations.
{
  pkgs,
  uv2nix ? null,
  pyproject-nix ? null,
  pyproject-build-systems ? null,
  npm-lockfile-fix ? null,
  rev ? null,
}:
let
  inherit (pkgs) lib;
  stable = import ./inputs.nix { inherit pkgs; };

  minimal = pkgs.callPackage ./hermes-agent.nix {
    uv2nix = if uv2nix != null then uv2nix else stable.uv2nix;
    pyproject-nix = if pyproject-nix != null then pyproject-nix else stable.pyproject-nix;
    pyproject-build-systems =
      if pyproject-build-systems != null then pyproject-build-systems else stable.pyproject-build-systems;
    npm-lockfile-fix =
      if npm-lockfile-fix != null then npm-lockfile-fix else stable.npm-lockfile-fix;
    inherit rev;
  };

  # All platform-portable optional integrations pre-built.
  # matrix is Linux-only (oqs/liboqs lacks aarch64-darwin wheels).
  full = minimal.override {
    extraDependencyGroups = [
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
      "hindsight"
      "honcho"
      "messaging"
      "modal"
      "parallel-web"
      "tts-premium"
      "voice"
    ] ++ lib.optionals pkgs.stdenv.isLinux [ "matrix" ];
  };
in
{
  # default ships the fat agent (all platform-portable integrations); `minimal`
  # is the lean core. Matches upstream `change(nix): ship fat hermes agent by default`.
  default = full;

  inherit minimal;

  # Ships discord.py + python-telegram-bot + slack-sdk so a plain
  # `nix profile install .#messaging` connects to Discord/Telegram/Slack
  # on first run — lazy-install can't write to the read-only /nix/store.
  messaging = minimal.override {
    extraDependencyGroups = [ "messaging" ];
  };

  tui = full.hermesTui;
  web = full.hermesWeb;
  desktop = full.hermesDesktop;
}
