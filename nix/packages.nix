# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }: {
  perSystem = { pkgs, system, ... }:
    let
      hermesVenv = pkgs.callPackage ./python.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
      };

      runtimeDeps = with pkgs; [
        nodejs_20 ripgrep git openssh ffmpeg
      ];

      runtimePath = pkgs.lib.makeBinPath runtimeDeps;

      # Submodules fetched declaratively (git deps break Nix sandbox)
      mini-swe-agent-src = pkgs.fetchFromGitHub {
        owner = "SWE-agent";
        repo = "mini-swe-agent";
        rev = "07aa6a738556e44b30d7b5c3bbd5063dac871d25";
        hash = "sha256-7+8dvi49iQMO4bXK5VYcem1+Tub5vMCrrZeNcEojAUQ=";
      };

      tinker-atropos-src = pkgs.fetchFromGitHub {
        owner = "nousresearch";
        repo = "tinker-atropos";
        rev = "65f084ee8054a5d02aeac76e24ed60388511c82b";
        hash = "sha256-tD1VyUfMin+KnkQD+eyEibeJNe6d4dgB1b6wFe+3gKs=";
      };
    in {
      packages.default = pkgs.stdenv.mkDerivation {
        pname = "hermes-agent";
        version = "0.1.0";

        dontUnpack = true;
        dontBuild = true;
        nativeBuildInputs = [ pkgs.makeWrapper ];

        installPhase = ''
          runHook preInstall

          # Place submodule sources for runtime import
          mkdir -p $out/share/hermes-agent
          cp -r ${mini-swe-agent-src}/src/minisweagent $out/share/hermes-agent/minisweagent
          cp -r ${tinker-atropos-src} $out/share/hermes-agent/tinker-atropos

          # Wrap entry points from the uv2nix venv
          mkdir -p $out/bin
          makeWrapper ${hermesVenv}/bin/hermes $out/bin/hermes \
            --prefix PATH : "${runtimePath}" \
            --prefix PYTHONPATH : $out/share/hermes-agent

          makeWrapper ${hermesVenv}/bin/hermes-agent $out/bin/hermes-agent \
            --prefix PATH : "${runtimePath}" \
            --prefix PYTHONPATH : $out/share/hermes-agent

          runHook postInstall
        '';

        meta = with pkgs.lib; {
          description = "AI agent with advanced tool-calling capabilities";
          homepage = "https://github.com/NousResearch/hermes-agent";
          mainProgram = "hermes";
          license = licenses.mit;
          platforms = platforms.unix;
        };
      };
    };
}
