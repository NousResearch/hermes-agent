{
  description = "Hermes Agent - AI agent framework by Nous Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python311;

        runtimeDeps = with pkgs; [
          python
          uv
          nodejs_20
          ripgrep
          git
          openssh
        ];

        # Submodule sources (fetched declaratively)
        # NOTE: Placeholder hashes — on first `nix build`, Nix will report
        # the correct SRI hash. Replace the hash values with what Nix prints.
        mini-swe-agent-src = pkgs.fetchFromGitHub {
          owner = "SWE-agent";
          repo = "mini-swe-agent";
          rev = "07aa6a738556e44b30d7b5c3bbd5063dac871d25";
          hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        };

        tinker-atropos-src = pkgs.fetchFromGitHub {
          owner = "nousresearch";
          repo = "tinker-atropos";
          rev = "65f084ee8054a5d02aeac76e24ed60388511c82b";
          hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        };

        hermes-agent = pkgs.stdenv.mkDerivation {
          pname = "hermes-agent";
          version = "0.1.0";

          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let
                baseName = baseNameOf path;
                relPath = pkgs.lib.removePrefix (toString ./. + "/") (toString path);
              in
              # Exclude development / transient artifacts
              baseName != ".git"
              && baseName != ".venv"
              && baseName != "venv"
              && baseName != "node_modules"
              && baseName != "__pycache__"
              && baseName != ".mypy_cache"
              && baseName != ".pytest_cache"
              && baseName != "result"
              && baseName != ".claude"
              # Exclude submodule dirs (we fetch them declaratively)
              && relPath != "mini-swe-agent"
              && relPath != "tinker-atropos"
              && !(pkgs.lib.hasPrefix "mini-swe-agent/" relPath)
              && !(pkgs.lib.hasPrefix "tinker-atropos/" relPath);
          };

          nativeBuildInputs = [ pkgs.makeWrapper ];

          # No build phase — this is a source bundle, not a compiled derivation
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

      in {
        packages.default = hermes-agent;

        devShells.default = pkgs.mkShell {
          packages = runtimeDeps;

          shellHook = ''
            echo "Hermes Agent dev shell"

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating Python 3.11 venv..."
              uv venv .venv --python ${python}/bin/python3
            fi

            # Activate venv
            source .venv/bin/activate

            # Install Python deps if not already present
            if ! python -c "import openai" 2>/dev/null; then
              echo "Installing Python dependencies..."
              uv pip install -e ".[all]"
              if [ -d mini-swe-agent ]; then
                uv pip install -e ./mini-swe-agent
              fi
              if [ -d tinker-atropos ]; then
                uv pip install -e ./tinker-atropos
              fi
            fi

            # Install npm deps if needed
            if [ -f package.json ] && [ ! -d node_modules ]; then
              echo "Installing npm dependencies..."
              npm install
            fi

            echo "Ready. Run 'hermes' to start."
          '';
        };
      }
    )) // {
      # NixOS module (system-independent)
      nixosModules.default = import ./nix/module.nix self;
    };
}
