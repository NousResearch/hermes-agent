{
  description = "Hermes Agent - AI agent framework by Nous Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    npm-lockfile-fix = {
      url = "github:jeslie0/npm-lockfile-fix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      # checks/ and devShells are flake-specific; the stable-Nix-consumable
      # building blocks live in self-contained files re-exported below.
      imports = [
        ./nix/checks.nix
        ./nix/devShell.nix
      ];

      flake = {
        # Stable Nix:  nixpkgs.overlays = [ (import ./nix/overlay.nix) ];
        # A plain `final: prev:` overlay — re-exported verbatim. It self-sources
        # its build inputs from nix/inputs.nix (the same flake.lock), so flake
        # and non-flake consumers get the same pinned derivations.
        overlays.default = import ./nix/overlay.nix;

        # Stable Nix:  imports = [ ./nix/module.nix ]; with the overlay applied
        # so pkgs.hermes-agent (module.nix's default package) resolves.
        #
        # For flake users we instead pin the package to the flake's own package
        # set — byte-identical to the pre-refactor behavior. This avoids forcing
        # nixpkgs.overlays onto the consumer's nixpkgs (which would conflict with
        # an externally-set nixpkgs.pkgs), so importing nixosModules.default
        # behaves exactly as before.
        nixosModules.default =
          { pkgs, lib, ... }:
          {
            imports = [ ./nix/module.nix ];
            services.hermes-agent.package =
              lib.mkDefault inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default;
          };
      };

      perSystem =
        { pkgs, system, ... }:
        {
          packages = import ./nix/packages.nix {
            inherit pkgs;
            inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
            npm-lockfile-fix = inputs.npm-lockfile-fix.packages.${system}.default;
            rev = inputs.self.rev or null;
          };
        };
    };
}
