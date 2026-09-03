# nix/nixosModules.nix — the NixOS module for hermes-agent
#
# This module shares its options, its renderers for config.yaml, .env and
# documents, and its state setup with the Home Manager module
# (nix/homeManagerModules.nix). The shared code is in nix/moduleCommon.nix.
# This file holds only the parts that need root: the service user, a system
# state directory, the system PATH, and container mode.
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
# Tool resolution: the hermes wrapper uses --suffix PATH for nix store tools,
# so apt/uv-installed versions take priority. The container entrypoint provisions
# extensible tools on first boot: nodejs/npm via apt, uv via curl, and a Python
# 3.11 venv (bootstrapped entirely by uv) at ~/.venv with pip seeded. Agents get
# writable tool prefixes for npm i -g, pip install, uv tool install, etc.
#
# Usage:
#   services.hermes-agent = {
#     enable = true;
#     settings.model.default = "anthropic/claude-sonnet-4";
#     environmentFiles = [ config.sops.secrets."hermes/env".path ];
#   };
#
{ inputs, ... }:
{
  flake.nixosModules.default =
    {
      config,
      lib,
      options,
      pkgs,
      ...
    }:

    let
      cfg = config.services.hermes-agent;
      common = import ./moduleCommon.nix { inherit lib; };

      effectivePackage = common.effectivePackage cfg;
      hermes-agent = inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default;

      hermesHome = "${cfg.stateDir}/.hermes";

      # In container mode, the agent uses the mount path in the container.
      effectiveWorkDir = if cfg.container.enable then containerWorkDir else cfg.workingDirectory;

      # config.yaml mode: group-writable (0660) when interactive users share this
      # HERMES_HOME via addToSystemPackages, so they can save settings through the
      # CLI/TUI without hitting EACCES; otherwise group-read-only (0640). Secrets
      # (.env) stay 0640 regardless.
      configYamlMode = if cfg.addToSystemPackages then "0660" else "0640";

      containerName = "hermes-agent";
      containerDataDir = "/data"; # stateDir mount point inside container
      containerHomeDir = "/home/hermes";

      # ── Container mode helpers ──────────────────────────────────────────
      containerBin =
        if cfg.container.backend == "docker" then
          "${pkgs.docker}/bin/docker"
        else
          "${pkgs.podman}/bin/podman";

      # Runs as root inside the container on every start. Provisions the
      # hermes user + sudo on first boot (writable layer persists), then
      # drops privileges. Supports arbitrary base images (Debian, Alpine, etc).
      containerEntrypoint = pkgs.writeShellScript "hermes-container-entrypoint" ''
        set -eu

        HERMES_UID="''${HERMES_UID:?HERMES_UID must be set}"
        HERMES_GID="''${HERMES_GID:?HERMES_GID must be set}"

        # ── Group: ensure a group with GID=$HERMES_GID exists ──
        # Check by GID (not name) to avoid collisions with pre-existing groups
        # (e.g. GID 100 = "users" on Ubuntu)
        EXISTING_GROUP=$(getent group "$HERMES_GID" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "$EXISTING_GROUP" ]; then
          GROUP_NAME="$EXISTING_GROUP"
        else
          GROUP_NAME="hermes"
          if command -v groupadd >/dev/null 2>&1; then
            groupadd -g "$HERMES_GID" "$GROUP_NAME"
          elif command -v addgroup >/dev/null 2>&1; then
            addgroup -g "$HERMES_GID" "$GROUP_NAME" 2>/dev/null || true
          fi
        fi

        # ── User: ensure a user with UID=$HERMES_UID exists ──
        PASSWD_ENTRY=$(getent passwd "$HERMES_UID" 2>/dev/null || true)
        if [ -n "$PASSWD_ENTRY" ]; then
          TARGET_USER=$(echo "$PASSWD_ENTRY" | cut -d: -f1)
          TARGET_HOME=$(echo "$PASSWD_ENTRY" | cut -d: -f6)
        else
          TARGET_USER="hermes"
          TARGET_HOME="/home/hermes"
          if command -v useradd >/dev/null 2>&1; then
            useradd -u "$HERMES_UID" -g "$HERMES_GID" -m -d "$TARGET_HOME" -s /bin/bash "$TARGET_USER"
          elif command -v adduser >/dev/null 2>&1; then
            adduser -u "$HERMES_UID" -D -h "$TARGET_HOME" -s /bin/sh -G "$GROUP_NAME" "$TARGET_USER" 2>/dev/null || true
          fi
        fi
        mkdir -p "$TARGET_HOME"
        chown "$HERMES_UID:$HERMES_GID" "$TARGET_HOME"
        chmod 0750 "$TARGET_HOME"

        # Ensure HERMES_HOME is owned by the target user.
        # Use find instead of chown -R: chown strips the setgid bit (kernel
        # behavior), destroying the 2770 permissions the NixOS activation
        # script sets for group access by hostUsers.  Only touch files with
        # wrong ownership so correctly-owned dirs keep their permission bits.
        if [ -n "''${HERMES_HOME:-}" ] && [ -d "$HERMES_HOME" ]; then
          find "$HERMES_HOME" \! -user "$HERMES_UID" -exec chown "$HERMES_UID:$HERMES_GID" {} +
        fi

        # ── Provision apt packages (first boot only, cached in writable layer) ──
        # sudo: agent self-modification
        # nodejs/npm: writable node so npm i -g works (nix store copies are read-only)
        #   Node 22 via NodeSource — Ubuntu 24.04 ships Node 18 which is EOL.
        # curl: needed for uv installer + NodeSource setup
        if [ ! -f /var/lib/hermes-tools-provisioned ] && command -v apt-get >/dev/null 2>&1; then
          echo "First boot: provisioning agent tools..."
          apt-get update -qq
          apt-get install -y -qq sudo curl ca-certificates gnupg
          mkdir -p /etc/apt/keyrings
          curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
            | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
          echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
            > /etc/apt/sources.list.d/nodesource.list
          apt-get update -qq
          apt-get install -y -qq nodejs
          touch /var/lib/hermes-tools-provisioned
        fi

        if command -v sudo >/dev/null 2>&1 && [ ! -f /etc/sudoers.d/hermes ]; then
          mkdir -p /etc/sudoers.d
          echo "$TARGET_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/hermes
          chmod 0440 /etc/sudoers.d/hermes
        fi

        # uv (Python manager) — not in Ubuntu repos, retry-safe outside the sentinel
        if ! command -v uv >/dev/null 2>&1 && [ ! -x "$TARGET_HOME/.local/bin/uv" ] && command -v curl >/dev/null 2>&1; then
          su -s /bin/sh "$TARGET_USER" -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || true
        fi

        # Python 3.12 venv — gives the agent a writable Python with pip.
        # --seed includes pip/setuptools so bare `pip install` works.
        _UV_BIN="$TARGET_HOME/.local/bin/uv"
        if [ ! -d "$TARGET_HOME/.venv" ] && [ -x "$_UV_BIN" ]; then
          su -s /bin/sh "$TARGET_USER" -c "
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            uv python install 3.12
            uv venv --python 3.12 --seed \"\$HOME/.venv\"
          " || true
        fi

        # Put the agent venv first on PATH so python/pip resolve to writable copies
        if [ -d "$TARGET_HOME/.venv/bin" ]; then
          export PATH="$TARGET_HOME/.venv/bin:$PATH"
        fi

        if command -v setpriv >/dev/null 2>&1; then
          exec setpriv --reuid="$HERMES_UID" --regid="$HERMES_GID" --init-groups "$@"
        elif command -v su >/dev/null 2>&1; then
          exec su -s /bin/sh "$TARGET_USER" -c 'exec "$0" "$@"' -- "$@"
        else
          echo "WARNING: no privilege-drop tool (setpriv/su), running as root" >&2
          exec "$@"
        fi
      '';

      # Identity hash — only recreate container when structural config changes.
      # Package and entrypoint use stable symlinks (current-package, current-entrypoint)
      # so they can update without recreation. Env vars go through $HERMES_HOME/.env.
      containerIdentity = builtins.hashString "sha256" (
        builtins.toJSON {
          schema = 4; # bump when identity inputs change (4: Node 18→22 via NodeSource)
          image = cfg.container.image;
          extraVolumes = cfg.container.extraVolumes;
          extraOptions = cfg.container.extraOptions;
        }
      );

      identityFile = "${cfg.stateDir}/.container-identity";

      # The CLI on the host reads this file, in get_container_exec_info. The
      # file tells the CLI to run in the container and not on the host.
      containerModeFile = pkgs.writeText "hermes-container-mode" ''
        # Written by the NixOS activation script. Do not edit manually.
        backend=${cfg.container.backend}
        container_name=${containerName}
        exec_user=${cfg.user}
        hermes_bin=${containerDataDir}/current-package/bin/hermes
      '';

      # Default: /var/lib/hermes/workspace → /data/workspace.
      # Custom paths outside stateDir pass through unchanged (user must add extraVolumes).
      containerWorkDir =
        if lib.hasPrefix "${cfg.stateDir}/" cfg.workingDirectory then
          "${containerDataDir}/${lib.removePrefix "${cfg.stateDir}/" cfg.workingDirectory}"
        else
          cfg.workingDirectory;

      # The hardening and the environment that the gateway unit and the
      # backend unit share.
      commonServiceConfig = {
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.workingDirectory;

        Restart = cfg.restart;
        RestartSec = cfg.restartSec;

        # Shared-state: files created by the service should be group-writable
        # so interactive users in the hermes group can read/write them.
        UMask = "0007";

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = false;
        ReadWritePaths = [
          cfg.stateDir
          cfg.workingDirectory
        ];
        PrivateTmp = true;
      };

      commonUnitEnvironment = {
        HOME = cfg.stateDir;
      }
      // common.processEnvironment { inherit hermesHome; };

      unitPath = common.processPath { inherit pkgs cfg; };

    in
    {
      options.services.hermes-agent =
        common.sharedOptions {
          defaultPackage = hermes-agent;
          defaultPackageText = lib.literalExpression "hermes-agent.packages.\${system}.default";
          defaultWorkingDirectory = "${cfg.stateDir}/workspace";
          defaultWorkingDirectoryText = lib.literalExpression ''"''${cfg.stateDir}/workspace"'';
        }
        // (
          with lib;
          {
            # ── Service identity ───────────────────────────────────────────
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

            # ── Directories ────────────────────────────────────────────────
            stateDir = mkOption {
              type = types.str;
              default = "/var/lib/hermes";
              description = "State directory. Contains .hermes/ subdir (HERMES_HOME).";
            };

            addToSystemPackages = mkOption {
              type = types.bool;
              default = false;
              description = ''
                Add the hermes CLI to environment.systemPackages and export
                HERMES_HOME system-wide (via environment.variables) so interactive
                shells share state with the gateway service.
              '';
            };

            # ── OCI Container (opt-in) ────────────────────────────────────
            container = {
              enable = mkEnableOption "OCI container mode (Ubuntu base, full self-modification support)";

              backend = mkOption {
                type = types.enum [
                  "docker"
                  "podman"
                ];
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

              hostUsers = mkOption {
                type = types.listOf types.str;
                default = [ ];
                description = ''
                  Interactive users who get a ~/.hermes symlink to the service
                  stateDir. These users are automatically added to the hermes group.
                '';
                example = [ "sidbin" ];
              };
            };
          }
        );

      config = lib.mkIf cfg.enable (
        lib.mkMerge [

          # ── Merge MCP servers into settings ────────────────────────────────
          (lib.mkIf (cfg.mcpServers != { }) {
            services.hermes-agent.settings.mcp_servers = common.mcpServersToConfig cfg.mcpServers;
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
          # Add the hermes CLI to system PATH and export HERMES_HOME system-wide
          # so interactive shells share state (sessions, skills, cron) with the
          # gateway service instead of creating a separate ~/.hermes/.
          (lib.mkIf cfg.addToSystemPackages {
            environment.systemPackages = [ effectivePackage ];
            environment.variables.HERMES_HOME = hermesHome;
          })

          # ── Host user group membership ─────────────────────────────────────
          (lib.mkIf (cfg.container.enable && cfg.container.hostUsers != [ ]) {
            users.users = lib.genAttrs cfg.container.hostUsers (_user: {
              extraGroups = [ cfg.group ];
            });
          })

          # ── Assertions ─────────────────────────────────────────────────────
          {
            assertions =
              common.pluginNameAssertions {
                inherit cfg;
                optionPath = "services.hermes-agent";
              }
              ++ common.workspaceFilesAssertions {
                inherit cfg;
                opt = options.services.hermes-agent.workingDirectory;
                optionPath = "services.hermes-agent";
              }
              ++ common.backendBindAssertions {
                inherit cfg;
                optionPath = "services.hermes-agent";
              }
              ++ [
                {
                  # Container mode runs one command in one container. A second
                  # process needs its own container and its own ports. This
                  # module does not do that.
                  assertion = !(cfg.container.enable && cfg.backend.mode != "none");
                  message = "services.hermes-agent: backend.mode is not supported together with container.enable — the container runs the gateway only.";
                }
              ];
          }

          # ── Per-user profile for extraPackages ───────────────────────────
          # Wire extraPackages into the hermes user's per-user profile so the
          # login-shell snapshot (which rebuilds PATH from NixOS profiles) sees
          # them.  The systemd service PATH also includes them for direct access.
          (lib.mkIf (cfg.extraPackages != [ ]) {
            # listOf options are merged by the NixOS module system — this appends to
            # any packages the operator assigned to this user externally (e.g. when
            # createUser = false and the user definition lives elsewhere in the config).
            users.users.${cfg.user}.packages = cfg.extraPackages;
          })

          # ── Warnings ──────────────────────────────────────────────────────
          (lib.mkIf
            (cfg.container.enable && !cfg.addToSystemPackages && cfg.container.hostUsers != [ ])
            {
              warnings = [
                ''
                  services.hermes-agent: container.enable is true and container.hostUsers
                  is set, but addToSystemPackages is false. Without a host-installed hermes
                  binary, container routing will not work for interactive users.
                  Set addToSystemPackages = true or ensure hermes is on PATH.
                ''
              ];
            }
          )

          # ── Directories ───────────────────────────────────────────────────
          {
            systemd.tmpfiles.rules = [
              "d ${cfg.stateDir}                2770 ${cfg.user} ${cfg.group} - -"
              "d ${hermesHome}                  2770 ${cfg.user} ${cfg.group} - -"
              "d ${cfg.stateDir}/home           0750 ${cfg.user} ${cfg.group} - -"
              "d ${cfg.workingDirectory}        2770 ${cfg.user} ${cfg.group} - -"
            ]
            ++ map (d: "d ${hermesHome}/${d} 2770 ${cfg.user} ${cfg.group} - -") common.stateSubdirs;
          }

          # ── Activation: link config + auth + documents ────────────────────
          {
            system.activationScripts."hermes-agent-setup" =
              lib.stringAfter
                (
                  [ "users" ] ++ lib.optional (config.system.activationScripts ? setupSecrets) "setupSecrets"
                )
                ''
                  # Ensure directories exist (activation runs before tmpfiles)
                  mkdir -p ${hermesHome}
                  mkdir -p ${cfg.stateDir}/home
                  mkdir -p ${cfg.workingDirectory}
                  chown ${cfg.user}:${cfg.group} ${cfg.stateDir} ${hermesHome} ${cfg.stateDir}/home ${cfg.workingDirectory}
                  chmod 2770 ${cfg.stateDir} ${hermesHome} ${cfg.workingDirectory}
                  chmod 0750 ${cfg.stateDir}/home

                  # Create subdirs, set setgid + group-writable, migrate existing files.
                  # Nix-managed .env/.managed stay 0640/0644; config.yaml uses
                  # configYamlMode (0660 under addToSystemPackages, else 0640).
                  find ${hermesHome} -maxdepth 1 \
                    \( -name "*.db" -o -name "*.db-wal" -o -name "*.db-shm" -o -name "SOUL.md" \) \
                    -exec chmod g+rw {} + 2>/dev/null || true
                  for _subdir in ${lib.concatStringsSep " " common.stateSubdirs}; do
                    mkdir -p "${hermesHome}/$_subdir"
                    chown ${cfg.user}:${cfg.group} "${hermesHome}/$_subdir"
                    chmod 2770 "${hermesHome}/$_subdir"
                    find "${hermesHome}/$_subdir" -type f \
                      -exec chmod g+rw {} + 2>/dev/null || true
                  done

                  ${common.mkStateScript {
                    inherit pkgs cfg hermesHome;
                    workingDirectory = cfg.workingDirectory;
                    configWorkingDirectory = effectiveWorkDir;
                    owner = "${cfg.user}:${cfg.group}";
                    stateDirs = common.stateSubdirs;
                    modes = {
                      config = configYamlMode;
                      env = "0640";
                      managed = "0644";
                      auth = "0600";
                      document = "0640";
                    };
                  }}

                  chown -h ${cfg.user}:${cfg.group} ${hermesHome}/plugins/nix-managed-* 2>/dev/null || true

                  # Container mode metadata — tells the host CLI to exec into the
                  # container instead of running locally. Removed when container mode
                  # is disabled so the host CLI falls back to native execution.
                  ${
                    if cfg.container.enable then
                      ''
                        install -o ${cfg.user} -g ${cfg.group} -m 0644 ${containerModeFile} ${hermesHome}/.container-mode
                      ''
                    else
                      ''
                        rm -f ${hermesHome}/.container-mode

                        # Remove symlink bridge for hostUsers
                        ${lib.concatStringsSep "\n" (
                          map (
                            user:
                            let
                              userHome = config.users.users.${user}.home;
                              symlinkPath = "${userHome}/.hermes";
                            in
                            ''
                              if [ -L "${symlinkPath}" ] && [ "$(readlink "${symlinkPath}")" = "${hermesHome}" ]; then
                                rm -f "${symlinkPath}"
                                echo "hermes-agent: removed symlink ${symlinkPath}"
                              fi
                            ''
                          ) cfg.container.hostUsers
                        )}
                      ''
                  }

                  # ── Symlink bridge for interactive users ───────────────────────
                  # Create ~/.hermes -> stateDir/.hermes for each hostUser so the
                  # host CLI shares state with the container service.
                  # Only runs when container mode is enabled.
                  ${lib.optionalString cfg.container.enable (
                    lib.concatStringsSep "\n" (
                      map (
                        user:
                        let
                          userHome = config.users.users.${user}.home;
                          symlinkPath = "${userHome}/.hermes";
                        in
                        ''
                          if [ -d "${symlinkPath}" ] && [ ! -L "${symlinkPath}" ]; then
                            # Real directory — back it up, then create symlink.
                            # (ln -sfn cannot atomically replace a directory.)
                            _backup="${symlinkPath}.bak.$(date +%s)"
                            echo "hermes-agent: backing up existing ${symlinkPath} to $_backup"
                            mv "${symlinkPath}" "$_backup"
                          fi
                          # For everything else (existing symlink, doesn't exist, etc.)
                          # ln -sfn handles it: replaces symlinks, creates new ones.
                          ln -sfn "${hermesHome}" "${symlinkPath}"
                          chown -h ${user}:${cfg.group} "${symlinkPath}"
                        ''
                      ) cfg.container.hostUsers
                    )
                  )}
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

              # cfg.environment and cfg.environmentFiles are written to
              # $HERMES_HOME/.env by the activation script. load_hermes_dotenv()
              # reads them at Python startup — no systemd EnvironmentFile needed.
              environment = commonUnitEnvironment;

              serviceConfig = commonServiceConfig // {
                ExecStart = lib.escapeShellArgs (common.gatewayArgv cfg);
              };

              path = unitPath;
            };
          })

          # ── The backend: hermes serve or hermes dashboard ─────────────────
          # This is a different process from the gateway. Both use one
          # HERMES_HOME.
          (lib.mkIf (!cfg.container.enable && cfg.backend.mode != "none") {
            systemd.services.hermes-backend = {
              description = common.backendDescription cfg;
              wantedBy = [ "multi-user.target" ];
              after = [ "network-online.target" ];
              wants = [ "network-online.target" ];

              environment = commonUnitEnvironment;

              serviceConfig = commonServiceConfig // {
                ExecStart = lib.escapeShellArgs (common.backendArgv { inherit pkgs cfg; });
              };

              path = unitPath;
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
              after = [
                "network-online.target"
              ]
              ++ lib.optional (cfg.container.backend == "docker") "docker.service";
              wants = [ "network-online.target" ];
              requires = lib.optional (cfg.container.backend == "docker") "docker.service";

              preStart = ''
                # Stable symlinks — container references these, not store paths directly
                ln -sfn ${effectivePackage} ${cfg.stateDir}/current-package
                ln -sfn ${containerEntrypoint} ${cfg.stateDir}/current-entrypoint

                # GC roots so nix-collect-garbage doesn't remove store paths in use
                ${pkgs.nix}/bin/nix-store --add-root ${cfg.stateDir}/.gc-root --indirect -r ${effectivePackage} 2>/dev/null || true
                ${pkgs.nix}/bin/nix-store --add-root ${cfg.stateDir}/.gc-root-entrypoint --indirect -r ${containerEntrypoint} 2>/dev/null || true

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
                  # Resolve numeric UID/GID — passed to entrypoint for in-container user setup
                  HERMES_UID=$(${pkgs.coreutils}/bin/id -u ${cfg.user})
                  HERMES_GID=$(${pkgs.coreutils}/bin/id -g ${cfg.user})

                  echo "Creating container..."
                  ${containerBin} create \
                    --name ${containerName} \
                    --network=host \
                    --entrypoint ${containerDataDir}/current-entrypoint \
                    --volume /nix/store:/nix/store:ro \
                    --volume ${cfg.stateDir}:${containerDataDir} \
                    --volume ${cfg.stateDir}/home:${containerHomeDir} \
                    ${lib.concatStringsSep " " (map (v: "--volume ${v}") cfg.container.extraVolumes)} \
                    --env HERMES_UID="$HERMES_UID" \
                    --env HERMES_GID="$HERMES_GID" \
                    --env HERMES_HOME=${containerDataDir}/.hermes \
                    --env HERMES_MANAGED=true \
                    --env HOME=${containerHomeDir} \
                    ${lib.concatStringsSep " " cfg.container.extraOptions} \
                    ${cfg.container.image} \
                    ${containerDataDir}/current-package/bin/hermes gateway run --replace ${lib.concatStringsSep " " cfg.extraArgs}

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
        ]
      );
    };
}
