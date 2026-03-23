# nix/nixosModules.nix — NixOS module for hermes-agent
#
# Two modes:
#   container.enable = false (default) → native systemd service
#   container.enable = true            → OCI container (persistent writable layer)
#
# Container mode: hermes runs from /nix/store bind-mounted read-only into a
# plain Ubuntu container. The writable layer (apt/pip/npm installs) persists
# across restarts and agent updates. Only image/volume/options changes trigger
# container recreation. Environment variables are written to $HERMES_HOME/.env
# and read by hermes at startup — no container recreation needed for env changes.
#
# Usage:
#   services.hermes-agent = {
#     enable = true;
#     settings.model = "anthropic/claude-sonnet-4";
#     environmentFiles = [ config.sops.secrets."hermes/env".path ];
#   };
#
{ inputs, ... }: {
  flake.nixosModules.default = { config, lib, pkgs, ... }:

  let
    cfg = config.services.hermes-agent;
    hermes-agent = inputs.self.packages.${pkgs.system}.default;

    # Deep-merge config type (from 0xrsydn/nix-hermes-agent)
    deepConfigType = lib.types.mkOptionType {
      name = "hermes-config-attrs";
      description = "Hermes YAML config (attrset), merged deeply via lib.recursiveUpdate.";
      check = builtins.isAttrs;
      merge = _loc: defs: lib.foldl' lib.recursiveUpdate { } (map (d: d.value) defs);
    };

    # Generate config.yaml from Nix attrset (YAML is a superset of JSON)
    configJson = builtins.toJSON cfg.settings;
    generatedConfigFile = pkgs.writeText "hermes-config.yaml" configJson;
    configFile = if cfg.configFile != null then cfg.configFile else generatedConfigFile;

    # Generate .env from non-secret environment attrset
    envFileContent = lib.concatStringsSep "\n" (
      lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment
    );
    # Build documents derivation (from 0xrsydn)
    documentDerivation = pkgs.runCommand "hermes-documents" { } (
      ''
        mkdir -p $out
      '' + lib.concatStringsSep "\n" (
        lib.mapAttrsToList (name: value:
          if builtins.isPath value || lib.isStorePath value
          then "cp ${value} $out/${name}"
          else "cat > $out/${name} <<'HERMES_DOC_EOF'\n${value}\nHERMES_DOC_EOF"
        ) cfg.documents
      )
    );

    containerName = "hermes-agent";

    # ── Container mode helpers ──────────────────────────────────────────
    containerBin = if cfg.container.backend == "docker"
      then "${pkgs.docker}/bin/docker"
      else "${pkgs.podman}/bin/podman";

    # Identity hash — only recreate container when structural config changes.
    # Environment variables are handled via $HERMES_HOME/.env (read by
    # load_hermes_dotenv at Python startup), so they don't need container recreation.
    containerIdentity = builtins.hashString "sha256" (builtins.toJSON {
      image = cfg.container.image;
      extraVolumes = cfg.container.extraVolumes;
      extraOptions = cfg.container.extraOptions;
    });

    identityFile = "${cfg.stateDir}/.container-identity";

  in {
    options.services.hermes-agent = with lib; {
      enable = mkEnableOption "Hermes Agent gateway service";

      # ── Package ──────────────────────────────────────────────────────────
      package = mkOption {
        type = types.package;
        default = hermes-agent;
        description = "The hermes-agent package to use.";
      };

      # ── Service identity ─────────────────────────────────────────────────
      user = mkOption {
        type = types.str;
        default = "hermes";
        description = "System user running the gateway.";
      };

      group = mkOption {
        type = types.str;
        default = "hermes";
        description = "System group running the gateway.";
      };

      createUser = mkOption {
        type = types.bool;
        default = true;
        description = "Create the user/group automatically.";
      };

      # ── Directories ──────────────────────────────────────────────────────
      stateDir = mkOption {
        type = types.str;
        default = "/var/lib/hermes";
        description = "State directory. Contains .hermes/ subdir (HERMES_HOME).";
      };

      workingDirectory = mkOption {
        type = types.str;
        default = "${cfg.stateDir}/workspace";
        defaultText = literalExpression ''"''${cfg.stateDir}/workspace"'';
        description = "Working directory for the agent (MESSAGING_CWD).";
      };

      # ── Declarative config ───────────────────────────────────────────────
      configFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          Path to an existing config.yaml. If set, takes precedence over
          the declarative `settings` option.
        '';
      };

      settings = mkOption {
        type = deepConfigType;
        default = { };
        description = ''
          Declarative Hermes config (attrset). Deep-merged across module
          definitions and rendered as config.yaml.
        '';
        example = literalExpression ''
          {
            model = "anthropic/claude-sonnet-4";
            terminal.backend = "local";
            compression = { enabled = true; threshold = 0.85; };
            toolsets = [ "all" ];
          }
        '';
      };

      # ── Secrets / environment ────────────────────────────────────────────
      environmentFiles = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = ''
          Paths to environment files containing secrets (API keys, tokens).
          Contents are merged into $HERMES_HOME/.env at activation time.
          Hermes reads this file on every startup via load_hermes_dotenv().
        '';
      };

      environment = mkOption {
        type = types.attrsOf types.str;
        default = { };
        description = ''
          Non-secret environment variables. Merged into $HERMES_HOME/.env
          at activation time. Do NOT put secrets here — use environmentFiles.
        '';
      };

      authFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          Path to an auth.json seed file (OAuth credentials).
          Only copied on first deploy — existing auth.json is preserved.
        '';
      };

      authFileForceOverwrite = mkOption {
        type = types.bool;
        default = false;
        description = "Always overwrite auth.json from authFile on activation.";
      };

      # ── Documents ────────────────────────────────────────────────────────
      documents = mkOption {
        type = types.attrsOf (types.either types.str types.path);
        default = { };
        description = ''
          Workspace files (SOUL.md, USER.md, etc.). Keys are filenames,
          values are inline strings or paths. Installed into workingDirectory.
        '';
        example = literalExpression ''
          {
            "SOUL.md" = "You are a helpful AI assistant.";
            "USER.md" = ./documents/USER.md;
          }
        '';
      };

      # ── MCP Servers ──────────────────────────────────────────────────────
      mcpServers = mkOption {
        type = types.attrsOf (types.submodule {
          options = {
            command = mkOption { type = types.str; description = "MCP server command."; };
            args = mkOption { type = types.listOf types.str; default = [ ]; };
            env = mkOption { type = types.attrsOf types.str; default = { }; };
            timeout = mkOption { type = types.nullOr types.int; default = null; };
          };
        });
        default = { };
        description = "MCP server configurations (merged into settings.mcp_servers).";
      };

      # ── Service behavior ─────────────────────────────────────────────────
      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra command-line arguments for `hermes gateway`.";
      };

      extraPackages = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = "Extra packages available on PATH.";
      };

      restart = mkOption {
        type = types.str;
        default = "always";
        description = "systemd Restart= policy.";
      };

      restartSec = mkOption {
        type = types.int;
        default = 5;
        description = "systemd RestartSec= value.";
      };

      addToSystemPackages = mkOption {
        type = types.bool;
        default = false;
        description = "Add hermes CLI to environment.systemPackages.";
      };

      # ── OCI Container (opt-in) ──────────────────────────────────────────
      container = {
        enable = mkEnableOption "OCI container mode (Ubuntu base, full self-modification support)";

        backend = mkOption {
          type = types.enum [ "docker" "podman" ];
          default = "docker";
          description = "Container runtime.";
        };

        extraVolumes = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "Extra volume mounts (host:container:mode format).";
          example = [ "/home/user/projects:/projects:rw" ];
        };

        extraOptions = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "Extra arguments passed to docker/podman run.";
        };

        image = mkOption {
          type = types.str;
          default = "ubuntu:24.04";
          description = "OCI container image. The container pulls this at runtime via Docker/Podman.";
        };
      };
    };

    config = lib.mkIf cfg.enable (lib.mkMerge [

      # ── Merge MCP servers into settings ────────────────────────────────
      (lib.mkIf (cfg.mcpServers != { }) {
        services.hermes-agent.settings.mcp_servers = lib.mapAttrs (_name: srv:
          { inherit (srv) command args; }
          // lib.optionalAttrs (srv.env != { }) { inherit (srv) env; }
          // lib.optionalAttrs (srv.timeout != null) { inherit (srv) timeout; }
        ) cfg.mcpServers;
      })

      # ── User / group ──────────────────────────────────────────────────
      (lib.mkIf cfg.createUser {
        users.groups.${cfg.group} = { };
        users.users.${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = cfg.stateDir;
          createHome = true;
          shell = pkgs.bashInteractive;
        };
      })

      # ── Host CLI ──────────────────────────────────────────────────────
      (lib.mkIf cfg.addToSystemPackages {
        environment.systemPackages = [ cfg.package ];
        environment.variables.HERMES_HOME = "${cfg.stateDir}/.hermes";
      })

      # ── Directories ───────────────────────────────────────────────────
      {
        systemd.tmpfiles.rules = [
          "d ${cfg.stateDir}                0755 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes        0755 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.workingDirectory}         0750 ${cfg.user} ${cfg.group} - -"
        ];
      }

      # ── Activation: link config + auth + documents ────────────────────
      {
        system.activationScripts."hermes-agent-setup" = lib.stringAfter [ "users" ] ''
          # Ensure directories exist (activation runs before tmpfiles)
          mkdir -p ${cfg.stateDir}/.hermes
          mkdir -p ${cfg.workingDirectory}
          chown ${cfg.user}:${cfg.group} ${cfg.stateDir} ${cfg.stateDir}/.hermes ${cfg.workingDirectory}

          # Link config file
          install -o ${cfg.user} -g ${cfg.group} -m 0644 -D ${configFile} ${cfg.stateDir}/.hermes/config.yaml

          # Managed mode marker (so interactive shells also detect NixOS management)
          touch ${cfg.stateDir}/.hermes/.managed
          chown ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/.managed

          # Seed auth file if provided
          ${lib.optionalString (cfg.authFile != null) ''
            ${if cfg.authFileForceOverwrite then ''
              install -o ${cfg.user} -g ${cfg.group} -m 0600 ${cfg.authFile} ${cfg.stateDir}/.hermes/auth.json
            '' else ''
              if [ ! -f ${cfg.stateDir}/.hermes/auth.json ]; then
                install -o ${cfg.user} -g ${cfg.group} -m 0600 ${cfg.authFile} ${cfg.stateDir}/.hermes/auth.json
              fi
            ''}
          ''}

          # Seed .env from Nix-declared environment + environmentFiles.
          # Hermes reads $HERMES_HOME/.env at startup via load_hermes_dotenv(),
          # so this is the single source of truth for both native and container mode.
          ${lib.optionalString (cfg.environment != {} || cfg.environmentFiles != []) ''
            ENV_FILE="${cfg.stateDir}/.hermes/.env"
            install -o ${cfg.user} -g ${cfg.group} -m 0600 /dev/null "$ENV_FILE"
            cat > "$ENV_FILE" <<'HERMES_NIX_ENV_EOF'
${envFileContent}
HERMES_NIX_ENV_EOF
            ${lib.concatStringsSep "\n" (map (f: ''
              if [ -f "${f}" ]; then
                echo "" >> "$ENV_FILE"
                cat "${f}" >> "$ENV_FILE"
              fi
            '') cfg.environmentFiles)}
          ''}

          # Link documents into workspace
          ${lib.concatStringsSep "\n" (lib.mapAttrsToList (name: _value: ''
            install -o ${cfg.user} -g ${cfg.group} -m 0644 ${documentDerivation}/${name} ${cfg.workingDirectory}/${name}
          '') cfg.documents)}
        '';
      }

      # ══════════════════════════════════════════════════════════════════
      # MODE A: Native systemd service (default)
      # ══════════════════════════════════════════════════════════════════
      (lib.mkIf (!cfg.container.enable) {
        systemd.services.hermes-agent = {
          description = "Hermes Agent Gateway";
          wantedBy = [ "multi-user.target" ];
          after = [ "network-online.target" ];
          wants = [ "network-online.target" ];

          environment = {
            HOME = cfg.stateDir;
            HERMES_HOME = "${cfg.stateDir}/.hermes";
            HERMES_MANAGED = "true";
            MESSAGING_CWD = cfg.workingDirectory;
          };

          serviceConfig = {
            User = cfg.user;
            Group = cfg.group;
            WorkingDirectory = cfg.workingDirectory;

            # cfg.environment and cfg.environmentFiles are written to
            # $HERMES_HOME/.env by the activation script. load_hermes_dotenv()
            # reads them at Python startup — no systemd EnvironmentFile needed.

            ExecStart = lib.concatStringsSep " " ([
              "${cfg.package}/bin/hermes"
              "gateway"
            ] ++ cfg.extraArgs);

            Restart = cfg.restart;
            RestartSec = cfg.restartSec;

            # Hardening
            NoNewPrivileges = true;
            ProtectSystem = "strict";
            ProtectHome = false;
            ReadWritePaths = [ cfg.stateDir ];
            PrivateTmp = true;
          };

          path = [
            cfg.package
            pkgs.bash
            pkgs.coreutils
            pkgs.git
          ] ++ cfg.extraPackages;
        };
      })

      # ══════════════════════════════════════════════════════════════════
      # MODE B: OCI container (persistent writable layer)
      # ══════════════════════════════════════════════════════════════════
      (lib.mkIf cfg.container.enable {
        # Ensure the container runtime is available
        virtualisation.docker.enable = lib.mkDefault (cfg.container.backend == "docker");

        systemd.services.hermes-agent = {
          description = "Hermes Agent Gateway (container)";
          wantedBy = [ "multi-user.target" ];
          after = [ "network-online.target" ]
            ++ lib.optional (cfg.container.backend == "docker") "docker.service";
          wants = [ "network-online.target" ];
          requires = lib.optional (cfg.container.backend == "docker") "docker.service";

          preStart = ''
            # Update symlink to current hermes package
            ln -sfn ${cfg.package} ${cfg.stateDir}/current-package

            # GC root so nix-collect-garbage doesn't remove the running package
            ${pkgs.nix}/bin/nix-store --add-root ${cfg.stateDir}/.gc-root --indirect -r ${cfg.package} 2>/dev/null || true

            # Check if container needs (re)creation
            NEED_CREATE=false
            if ! ${containerBin} inspect ${containerName} &>/dev/null; then
              NEED_CREATE=true
            elif [ ! -f ${identityFile} ] || [ "$(cat ${identityFile})" != "${containerIdentity}" ]; then
              echo "Container config changed, recreating..."
              ${containerBin} rm -f ${containerName} || true
              NEED_CREATE=true
            fi

            if [ "$NEED_CREATE" = "true" ]; then
              # Resolve numeric UID/GID for container --user flag
              HERMES_UID=$(${pkgs.coreutils}/bin/id -u ${cfg.user})
              HERMES_GID=$(${pkgs.coreutils}/bin/id -g ${cfg.user})

              echo "Creating container..."
              ${containerBin} create \
                --name ${containerName} \
                --user "$HERMES_UID:$HERMES_GID" \
                --network=host \
                --volume /nix/store:/nix/store:ro \
                --volume ${cfg.stateDir}:/data \
                ${lib.concatStringsSep " " (map (v: "--volume ${v}") cfg.container.extraVolumes)} \
                --env HERMES_HOME=/data/.hermes \
                --env HERMES_MANAGED=true \
                --env HOME=/home/hermes \
                ${lib.concatStringsSep " " cfg.container.extraOptions} \
                ${cfg.container.image} \
                /data/current-package/bin/hermes gateway run --replace ${lib.concatStringsSep " " cfg.extraArgs}

              echo "${containerIdentity}" > ${identityFile}
            fi
          '';

          script = ''
            exec ${containerBin} start -a ${containerName}
          '';

          preStop = ''
            ${containerBin} stop -t 10 ${containerName} || true
          '';

          serviceConfig = {
            Type = "simple";
            Restart = cfg.restart;
            RestartSec = cfg.restartSec;
            TimeoutStopSec = 30;
          };
        };
      })
    ]);
  };
}
