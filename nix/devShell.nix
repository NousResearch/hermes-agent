# nix/devShell.nix — Fast dev shell with stamp-file optimization
{ inputs, ... }: {
  perSystem = { pkgs, ... }:
    let
      python = pkgs.python311;
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python uv nodejs_20 ripgrep git openssh
        ];

        shellHook = ''
          echo "Hermes Agent dev shell"

          # Composite stamp: changes when nix python or uv change
          STAMP_VALUE="${python}:${pkgs.uv}"
          STAMP_FILE=".venv/.nix-stamp"

          # Create venv if missing
          if [ ! -d .venv ]; then
            echo "Creating Python 3.11 venv..."
            uv venv .venv --python ${python}/bin/python3
          fi

          source .venv/bin/activate

          # Only install if stamp is stale or missing
          if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$STAMP_VALUE" ]; then
            echo "Installing Python dependencies..."
            uv pip install -e ".[all]"
            if [ -d mini-swe-agent ]; then
              uv pip install -e ./mini-swe-agent 2>/dev/null || true
            fi
            if [ -d tinker-atropos ]; then
              uv pip install -e ./tinker-atropos 2>/dev/null || true
            fi

            # Install npm deps
            if [ -f package.json ] && [ ! -d node_modules ]; then
              echo "Installing npm dependencies..."
              npm install
            fi

            echo "$STAMP_VALUE" > "$STAMP_FILE"
          fi

          echo "Ready. Run 'hermes' to start."
        '';
      };
    };
}
