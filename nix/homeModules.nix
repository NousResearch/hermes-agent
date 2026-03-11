# nix/homeModules.nix — Home-manager module for hermes-agent
{ inputs, ... }: {
  flake.homeManagerModules.default = { config, lib, pkgs, ... }:
    let
      cfg = config.services.hermes-agent;
    in {
      options.services.hermes-agent = {
        enable = lib.mkEnableOption "Hermes Agent AI assistant";

        package = lib.mkOption {
          type = lib.types.package;
          default = inputs.self.packages.${pkgs.system}.default;
          defaultText = lib.literalExpression "hermes-agent.packages.\${pkgs.system}.default";
          description = "The hermes-agent package.";
        };

        hermesHome = lib.mkOption {
          type = lib.types.str;
          default = "${config.home.homeDirectory}/.hermes";
          description = "Hermes config/sessions/memories directory.";
        };

        environmentFile = lib.mkOption {
          type = lib.types.str;
          default = "${config.home.homeDirectory}/.hermes/.env";
          description = "Path to environment file containing API keys.";
        };

        messagingCwd = lib.mkOption {
          type = lib.types.str;
          default = config.home.homeDirectory;
          description = "Working directory for gateway messaging.";
        };

        gateway = {
          enable = lib.mkEnableOption "Hermes Agent messaging gateway";
        };

        addToPATH = lib.mkOption {
          type = lib.types.bool;
          default = true;
          description = "Add hermes CLI to PATH via home.packages.";
        };
      };

      config = lib.mkIf cfg.enable {
        home.packages = lib.mkIf cfg.addToPATH [ cfg.package ];

        # Lightweight activation: just ensure directory structure exists
        home.activation.hermesAgentSetup = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
          mkdir -p "${cfg.hermesHome}"/{sessions,cron/output,logs,memories,skills}

          if [ ! -f "${cfg.hermesHome}/config.yaml" ]; then
            cat > "${cfg.hermesHome}/config.yaml" << 'YAML'
_config_version: 1
model:
  default: "anthropic/claude-opus-4.6"
terminal:
  env_type: "local"
YAML
          fi
        '';

        # Gateway systemd service
        systemd.user.services.hermes-agent-gateway = lib.mkIf cfg.gateway.enable {
          Unit = {
            Description = "Hermes Agent messaging gateway";
            After = [ "network-online.target" ];
            Wants = [ "network-online.target" ];
            StartLimitIntervalSec = 300;
            StartLimitBurst = 10;
          };

          Service = {
            Type = "simple";
            ExecStart = "${cfg.package}/bin/hermes gateway run --replace";
            Restart = "on-failure";
            RestartSec = 15;
            Environment = [
              "HOME=${config.home.homeDirectory}"
              "HERMES_HOME=${cfg.hermesHome}"
              "MESSAGING_CWD=${cfg.messagingCwd}"
            ];
            EnvironmentFile = cfg.environmentFile;
          };

          Install = {
            WantedBy = [ "default.target" ];
          };
        };
      };
    };
}
