# nix/nixosModules.nix — NixOS service module for hermes-agent
{ inputs, ... }: {
  flake.nixosModules.default = { config, lib, pkgs, ... }:
    let
      cfg = config.services.hermes-agent;
      pkg = cfg.package;

      # Shared across gateway + cron services
      commonBinPath = lib.makeBinPath (with pkgs; [
        python311 nodejs_20 ripgrep git openssh
      ]);

      commonEnvironment = [
        "HOME=${cfg.stateDir}"
        "HERMES_HOME=${cfg.stateDir}/.hermes"
        "PATH=${commonBinPath}:${cfg.stateDir}/app/.venv/bin"
      ];

      commonHardening = {
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ReadWritePaths = [ cfg.stateDir ];
      };

      mkServiceConfig = extraEnv: extraAttrs:
        {
          Type = "simple";
          User = cfg.user;
          Group = cfg.group;
          WorkingDirectory = "${cfg.stateDir}/app";
          Restart = "on-failure";
          RestartSec = 10;
          Environment = commonEnvironment ++ extraEnv;
        }
        // lib.optionalAttrs (cfg.environmentFile != null) {
          EnvironmentFile = cfg.environmentFile;
        }
        // commonHardening
        // extraAttrs;

    in {
      options.services.hermes-agent = {
        enable = lib.mkEnableOption "Hermes Agent AI assistant";

        package = lib.mkOption {
          type = lib.types.package;
          default = inputs.self.packages.${pkgs.system}.default;
          defaultText = lib.literalExpression "hermes-agent.packages.\${pkgs.system}.default";
          description = "The hermes-agent source bundle package.";
        };

        user = lib.mkOption {
          type = lib.types.str;
          default = "hermes";
          description = "User account under which hermes-agent runs.";
        };

        group = lib.mkOption {
          type = lib.types.str;
          default = "hermes";
          description = "Group under which hermes-agent runs.";
        };

        stateDir = lib.mkOption {
          type = lib.types.path;
          default = "/var/lib/hermes";
          description = "Directory for hermes-agent state (source copy, venv, config).";
        };

        environmentFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = "Path to an environment file containing API keys.";
        };

        gateway = {
          enable = lib.mkEnableOption "Hermes Agent messaging gateway (Telegram/Discord/WhatsApp)";
        };

        cron = {
          enable = lib.mkEnableOption "Hermes Agent cron scheduler";
        };
      };

      config = lib.mkIf cfg.enable {
        users.users.${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = cfg.stateDir;
          createHome = true;
          description = "Hermes Agent service user";
        };

        users.groups.${cfg.group} = {};

        # Setup service: copies source, creates venv, installs deps
        systemd.services.hermes-agent-setup = {
          description = "Hermes Agent environment setup";
          wantedBy = [ "multi-user.target" ];
          wants = [ "network-online.target" ];
          after = [ "network-online.target" ];

          serviceConfig = {
            Type = "oneshot";
            RemainAfterExit = true;
            TimeoutStartSec = 300;
            User = cfg.user;
            Group = cfg.group;
            WorkingDirectory = cfg.stateDir;
            StateDirectory = "hermes";
            PrivateNetwork = false;
          };

          path = with pkgs; [ python311 uv nodejs_20 git openssh coreutils rsync ];

          script = ''
            set -euo pipefail

            APP_DIR="${cfg.stateDir}/app"
            VENV_DIR="$APP_DIR/.venv"
            HERMES_HOME="${cfg.stateDir}/.hermes"
            STAMP_FILE="$APP_DIR/.nix-pkg-stamp"
            PKG_PATH="${pkg}"

            # Create hermes home structure
            mkdir -p "$HERMES_HOME"/{sessions,cron/output,logs}

            # Create default config.yaml if missing
            if [ ! -f "$HERMES_HOME/config.yaml" ]; then
              cat > "$HERMES_HOME/config.yaml" << 'YAML'
_config_version: 1
model:
  default: "anthropic/claude-opus-4.6"
terminal:
  env_type: "local"
YAML
            fi

            # Only re-copy source if package changed
            if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$PKG_PATH" ]; then
              echo "Package unchanged, skipping source copy."
            else
              echo "Copying source tree from Nix store..."
              mkdir -p "$APP_DIR"

              # rsync source, exclude venv and node_modules to preserve them
              rsync -a --delete \
                --exclude='.venv' \
                --exclude='node_modules' \
                "$PKG_PATH/share/hermes-agent/" "$APP_DIR/"

              # Make writable (Nix store copies are read-only)
              chmod -R u+w "$APP_DIR"

              echo "$PKG_PATH" > "$STAMP_FILE"

              # Force reinstall when source changes
              rm -f "$VENV_DIR/.deps-installed"
            fi

            # Create venv if missing
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv..."
              uv venv "$VENV_DIR" --python python3.11
            fi

            # Install Python deps if needed
            if [ ! -f "$VENV_DIR/.deps-installed" ]; then
              echo "Installing Python dependencies..."
              VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$APP_DIR[all]"

              if [ -d "$APP_DIR/mini-swe-agent" ]; then
                VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$APP_DIR/mini-swe-agent"
              fi

              if [ -d "$APP_DIR/tinker-atropos" ]; then
                VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$APP_DIR/tinker-atropos"
              fi

              touch "$VENV_DIR/.deps-installed"
            fi

            # Install npm deps if needed
            if [ -f "$APP_DIR/package.json" ] && [ ! -d "$APP_DIR/node_modules" ]; then
              echo "Installing npm dependencies..."
              cd "$APP_DIR"
              npm install --production
            fi

            echo "Setup complete."
          '';
        };

        # Gateway service
        systemd.services.hermes-agent-gateway = lib.mkIf cfg.gateway.enable {
          description = "Hermes Agent messaging gateway";
          wantedBy = [ "multi-user.target" ];
          requires = [ "hermes-agent-setup.service" ];
          after = [ "hermes-agent-setup.service" "network-online.target" ];
          wants = [ "network-online.target" ];
          serviceConfig = mkServiceConfig
            [ "MESSAGING_CWD=${cfg.stateDir}" ]
            { ExecStart = "${cfg.stateDir}/app/.venv/bin/hermes gateway run"; };
        };

        # Cron service
        systemd.services.hermes-agent-cron = lib.mkIf cfg.cron.enable {
          description = "Hermes Agent cron scheduler";
          wantedBy = [ "multi-user.target" ];
          requires = [ "hermes-agent-setup.service" ];
          after = [ "hermes-agent-setup.service" "network-online.target" ];
          wants = [ "network-online.target" ];
          serviceConfig = mkServiceConfig
            []
            { ExecStart = "${cfg.stateDir}/app/.venv/bin/hermes cron daemon"; };
        };
      };
    };
}
