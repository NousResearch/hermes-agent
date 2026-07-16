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
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
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
        # Flake consumers get a pure alias of this flake's locked package
        # (same idea as #65237) so pkgs.hermes-agent == .#default.
        # Stable (non-flake) Nix still uses nix/overlay.nix, which rebuilds
        # against the consumer's nixpkgs with build inputs from flake.lock:
        #   nixpkgs.overlays = [ (import ./nix/overlay.nix) ];
        overlays.default =
          final: _: {
            hermes-agent = inputs.self.packages.${final.stdenv.hostPlatform.system}.default;
          };

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
