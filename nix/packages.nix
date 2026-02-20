# nix/packages.nix — Source-bundle package for hermes-agent
{ inputs, ... }: {
  perSystem = { pkgs, system, ... }:
    let
      python = pkgs.python311;

      runtimeDeps = with pkgs; [
        python uv nodejs_20 ripgrep git openssh
      ];

      # Submodules fetched declaratively (not in uv workspace — their git deps break sandbox)
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

        src = pkgs.lib.cleanSourceWith {
          src = inputs.self;
          filter = path: type:
            let
              baseName = baseNameOf path;
              relPath = pkgs.lib.removePrefix (toString inputs.self + "/") (toString path);
            in
            baseName != ".git" && baseName != ".venv" && baseName != "venv"
            && baseName != "node_modules" && baseName != "__pycache__"
            && baseName != ".mypy_cache" && baseName != ".pytest_cache"
            && baseName != "result" && baseName != ".claude" && baseName != ".direnv"
            && relPath != "mini-swe-agent" && relPath != "tinker-atropos"
            && !(pkgs.lib.hasPrefix "mini-swe-agent/" relPath)
            && !(pkgs.lib.hasPrefix "tinker-atropos/" relPath);
        };

        nativeBuildInputs = [ pkgs.makeWrapper ];
        dontBuild = true;
        dontConfigure = true;

        installPhase = ''
          runHook preInstall

          mkdir -p $out/share/hermes-agent
          cp -r . $out/share/hermes-agent/

          # Place declaratively-fetched submodules
          cp -r ${mini-swe-agent-src} $out/share/hermes-agent/mini-swe-agent
          cp -r ${tinker-atropos-src} $out/share/hermes-agent/tinker-atropos

          # Create wrapper script that sets PATH for runtime deps
          mkdir -p $out/bin
          makeWrapper ${python}/bin/python3 $out/bin/hermes-agent-python \
            --prefix PATH : ${pkgs.lib.makeBinPath runtimeDeps}

          runHook postInstall
        '';

        meta = with pkgs.lib; {
          description = "AI agent with advanced tool-calling capabilities";
          homepage = "https://github.com/NousResearch/hermes-agent";
          license = licenses.mit;
          platforms = platforms.unix;
        };
      };
    };
}
