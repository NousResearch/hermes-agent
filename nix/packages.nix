# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }:
{
  perSystem =
    {
      pkgs,
      lib,
      ...
    }:
    let
      sandbox = pkgs.callPackage ./sandbox.nix { };

      # pm/lock.json translated into derivations — one per tool package
      # with an artifact for this system. The lockfile carries resolved
      # urls + hashes, so nix consumes it as pure data. callPackage adds
      # .override/.overrideDerivation to the returned SET; keep only the
      # real derivations.
      pmPackages = lib.filterAttrs (_: lib.isDerivation) (
        pkgs.callPackage ./pm-packages.nix { }
      );

      # Every bit of input wiring — flake inputs, provenance of the source,
      # pm/lock.json pins — lives in the builder, so this flake and any
      # consumer of flake.lib.mkHermesAgent build the same derivation.
      minimal = inputs.self.lib.mkHermesAgent { inherit pkgs; };

      # All platform-portable optional integrations pre-built. These are the
      # variant's own groups, not the user's extras, so an
      # `.override { extraDependencyGroups = ...; }` adds to them. The list
      # lives in `nix/exports.nix`; `overlays.default` resolves the same one
      # against the consumer's pkgs, so neither variant can ship fewer groups
      # than the other.
      full = minimal.override {
        defaultDependencyGroups = inputs.self.lib.defaultDependencyGroups pkgs;
      };
    in
    {
      packages = {
        node-gyp =
          (pkgs.callPackage ./lib.nix {
            inherit (pkgs) npm-lockfile-fix;
            inherit pmPackages;
          }).node-gyp;
        default = full;

        inherit sandbox;

        inherit minimal;

        # Ships discord.py + python-telegram-bot + slack-sdk so a plain
        # `nix profile install .#messaging` connects to Discord/Telegram/Slack
        # on first run — lazy-install can't write to the read-only /nix/store.
        messaging = minimal.override {
          defaultDependencyGroups = [ "messaging" ];
        };

        tui = full.hermesTui;
        web = full.hermesWeb;
        webapp = full.hermesWebapp;
        desktop = full.hermesDesktop;

        update-npm-lockfile = full.hermesNpmLib.updateNpmLockfile;
      }
      # Every pm lockfile tool as its own installable derivation:
      # `nix build .#pm-ripgrep`, `nix build .#pm-gh`, ...
      // lib.mapAttrs' (name: drv: lib.nameValuePair "pm-${name}" drv) pmPackages;
    };
}
