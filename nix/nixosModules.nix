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
# Tool resolution: the hermes wrapper uses --suffix PATH for nix store tools,
# so apt/uv-installed versions take priority. The container entrypoint provisions
# extensible tools on first boot: nodejs/npm via apt, uv via curl, and a Python
# 3.11 venv (bootstrapped entirely by uv) at ~/.venv with pip seeded. Agents get
# writable tool prefixes for npm i -g, pip install, uv tool install, etc.
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
    effectivePackage =
      if cfg.extraPythonPackages == [ ] && cfg.extraDependencyGroups == [ ]
      then cfg.package
      else cfg.package.override { inherit (cfg) extraPythonPackages extraDependencyGroups; };
    hermes-agent = inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default;

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

    configMergeScript = pkgs.callPackage ./configMergeScript.nix { };

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
    containerDataDir = "/data";     # stateDir mount point inside container
    containerHomeDir = "/home/hermes";

    # ── Container mode helpers ──────────────────────────────────────────
    containerBin = if cfg.container.backend == "docker"
      then "${pkgs.docker}/bin/docker"
      else "${pkgs.podman}/bin/podman";

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

      # Ensure HERMES_HOME is owned by the target user
      if [ -n "''${HERMES_HOME:-}" ] && [ -d "$HERMES_HOME" ]; then
        chown -R "$HERMES_UID:$HERMES_GID" "$HERMES_HOME"
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
    containerIdentity = builtins.hashString "sha256" (builtins.toJSON {
      schema = 4; # bump when identity inputs change (4: Node 18→22 via NodeSource)
      image = cfg.container.image;
      extraVolumes = cfg.container.extraVolumes;
      extraOptions = cfg.container.extraOptions;
    });

    identityFile = "${cfg.stateDir}/.container-identity";

    # Default: /var/lib/hermes/workspace → /data/workspace.
    # Custom paths outside stateDir pass through unchanged (user must add extraVolumes).
    containerWorkDir =
      if lib.hasPrefix "${cfg.stateDir}/" cfg.workingDirectory
      then "${containerDataDir}/${lib.removePrefix "${cfg.stateDir}/" cfg.workingDirectory}"
      else cfg.workingDirectory;

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
            # Stdio transport
            command = mkOption {
              type = types.nullOr types.str;
              default = null;
              description = "MCP server command (stdio transport).";
            };
            args = mkOption {
              type = types.listOf types.str;
              default = [ ];
              description = "Command-line arguments (stdio transport).";
            };
            env = mkOption {
              type = types.attrsOf types.str;
              default = { };
              description = "Environment variables for the server process (stdio transport).";
            };

            # HTTP/StreamableHTTP transport
            url = mkOption {
              type = types.nullOr types.str;
              default = null;
              description = "MCP server endpoint URL (HTTP/StreamableHTTP transport).";
            };
            headers = mkOption {
              type = types.attrsOf types.str;
              default = { };
              description = "HTTP headers, e.g. for authentication (HTTP transport).";
            };

            # Authentication
            auth = mkOption {
              type = types.nullOr (types.enum [ "oauth" ]);
              default = null;
              description = ''
                Authentication method. Set to "oauth" for OAuth 2.1 PKCE flow
                (remote MCP servers). Tokens are stored in $HERMES_HOME/mcp-tokens/.
              '';
            };

            # Enable/disable
            enabled = mkOption {
              type = types.bool;
              default = true;
              description = "Enable or disable this MCP server.";
            };

            # Common options
            timeout = mkOption {
              type = types.nullOr types.int;
              default = null;
              description = "Tool call timeout in seconds (default: 120).";
            };
            connect_timeout = mkOption {
              type = types.nullOr types.int;
              default = null;
              description = "Initial connection timeout in seconds (default: 60).";
            };

            # Tool filtering
            tools = mkOption {
              type = types.nullOr (types.submodule {
                options = {
                  include = mkOption {
                    type = types.listOf types.str;
                    default = [ ];
                    description = "Tool allowlist — only these tools are registered.";
                  };
                  exclude = mkOption {
                    type = types.listOf types.str;
                    default = [ ];
                    description = "Tool blocklist — these tools are hidden.";
                  };
                };
              });
              default = null;
              description = "Filter which tools are exposed by this server.";
            };

            # Sampling (server-initiated LLM requests)
            sampling = mkOption {
              type = types.nullOr (types.submodule {
                options = {
                  enabled = mkOption { type = types.bool; default = true; description = "Enable sampling."; };
                  model = mkOption { type = types.nullOr types.str; default = null; description = "Override model for sampling requests."; };
                  max_tokens_cap = mkOption { type = types.nullOr types.int; default = null; description = "Max tokens per request."; };
                  timeout = mkOption { type = types.nullOr types.int; default = null; description = "LLM call timeout in seconds."; };
                  max_rpm = mkOption { type = types.nullOr types.int; default = null; description = "Max requests per minute."; };
                  max_tool_rounds = mkOption { type = types.nullOr types.int; default = null; description = "Max tool-use rounds per sampling request."; };
                  allowed_models = mkOption { type = types.listOf types.str; default = [ ]; description = "Models the server is allowed to request."; };
                  log_level = mkOption {
                    type = types.nullOr (types.enum [ "debug" "info" "warning" ]);
                    default = null;
                    description = "Audit log level for sampling requests.";
                  };
                };
              });
              default = null;
              description = "Sampling configuration for server-initiated LLM requests.";
            };
          };
        });
        default = { };
        description = ''
          MCP server configurations (merged into settings.mcp_servers).
          Each server uses either stdio (command/args) or HTTP (url) transport.
        '';
        example = literalExpression ''
          {
            filesystem = {
              command = "npx";
              args = [ "-y" "@modelcontextprotocol/server-filesystem" "/home/user" ];
            };
            remote-api = {
              url = "http://my-server:8080/v0/mcp";
              headers = { Authorization = "Bearer ..."; };
            };
            remote-oauth = {
              url = "https://mcp.example.com/mcp";
              auth = "oauth";
            };
          }
        '';
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
        description = ''
          Extra packages available to the agent — terminal commands, skills,
          cron jobs, and the service process all see them.

          Implemented via the hermes user's per-user profile
          (`/etc/profiles/per-user/${cfg.user}/bin`), which NixOS includes
          in PATH for login shells.  The packages are also added to the
          systemd service PATH for direct process access.
        '';
      };

      extraPlugins = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = ''
          Directory-based plugin packages to symlink into the hermes plugins
          directory. Each package should contain a plugin.yaml and __init__.py
          at its root. Hermes discovers these automatically on startup.
        '';
        example = literalExpression ''
          [
            (pkgs.fetchFromGitHub {
              owner = "stephenschoettler";
              repo = "hermes-lcm";
              name = "hermes-lcm";
              rev = "v0.7.0";
              hash = "sha256-...";
            })
          ]
        '';
      };

      extraPythonPackages = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = ''
          Python packages to add to PYTHONPATH for entry-point plugin discovery.
          These are pip-packaged plugins that register via the
          hermes_agent.plugins entry-point group. Each package must be built
          with the same Python interpreter as hermes (python312).
        '';
        example = literalExpression ''
          [
            (pkgs.python312Packages.buildPythonPackage {
              pname = "rtk-hermes";
              version = "1.0.0";
              src = pkgs.fetchFromGitHub {
                owner = "ogallotti";
                repo = "rtk-hermes";
                rev = "main";
                hash = "sha256-...";
              };
            })
          ]
        '';
      };

      extraDependencyGroups = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = ''
          Additional pyproject.toml optional-dependency groups to include in
          the sealed Python venv. These are resolved by uv alongside core
          dependencies — no PYTHONPATH patching or collision risk.

          Use this for optional extras already declared in hermes-agent's
          pyproject.toml (e.g. "hindsight", "honcho", "voice").
          Use extraPythonPackages for external packages not in pyproject.toml.
        '';
        example = [ "hindsight" ];
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
        description = ''
          Add the hermes CLI to environment.systemPackages and export
          HERMES_HOME system-wide (via environment.variables) so interactive
          shells share state with the gateway service.
        '';
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
    };

    config = lib.mkIf cfg.enable (lib.mkMerge [

      # ── Merge MCP servers into settings ────────────────────────────────
      (lib.mkIf (cfg.mcpServers != { }) {
        services.hermes-agent.settings.mcp_servers = lib.mapAttrs (_name: srv:
          # Stdio transport
          lib.optionalAttrs (srv.command != null) { inherit (srv) command args; }
          // lib.optionalAttrs (srv.env != { }) { inherit (srv) env; }
          # HTTP transport
          // lib.optionalAttrs (srv.url != null) { inherit (srv) url; }
          // lib.optionalAttrs (srv.headers != { }) { inherit (srv) headers; }
          # Auth
          // lib.optionalAttrs (srv.auth != null) { inherit (srv) auth; }
          # Enable/disable
          // { inherit (srv) enabled; }
          # Common options
          // lib.optionalAttrs (srv.timeout != null) { inherit (srv) timeout; }
          // lib.optionalAttrs (srv.connect_timeout != null) { inherit (srv) connect_timeout; }
          # Tool filtering
          // lib.optionalAttrs (srv.tools != null) {
            tools = lib.filterAttrs (_: v: v != [ ]) {
              inherit (srv.tools) include exclude;
            };
          }
          # Sampling
          // lib.optionalAttrs (srv.sampling != null) {
            sampling = lib.filterAttrs (_: v: v != null && v != [ ]) {
              inherit (srv.sampling) enabled model max_tokens_cap timeout max_rpm
                max_tool_rounds allowed_models log_level;
            };
          }
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
      # Add the hermes CLI to system PATH and export HERMES_HOME system-wide
      # so interactive shells share state (sessions, skills, cron) with the
      # gateway service instead of creating a separate ~/.hermes/.
      (lib.mkIf cfg.addToSystemPackages {
        environment.systemPackages = [ effectivePackage ];
        environment.variables.HERMES_HOME = "${cfg.stateDir}/.hermes";
      })

      # ── Host user group membership ─────────────────────────────────────
      (lib.mkIf (cfg.container.enable && cfg.container.hostUsers != []) {
        users.users = lib.genAttrs cfg.container.hostUsers (user: {
          extraGroups = [ cfg.group ];
        });
      })

      # ── Assertions ─────────────────────────────────────────────────────
      {
        assertions = let
          names = map lib.getName cfg.extraPlugins;
        in [{
          assertion = (lib.length names) == (lib.length (lib.unique names));
          message = "services.hermes-agent.extraPlugins: duplicate plugin names detected: ${toString names}. If using fetchFromGitHub, set name = \"plugin-name\" to disambiguate.";
        }];
      }

      # ── Assertions ─────────────────────────────────────────────────────
      {
        assertions = let
          names = map lib.getName cfg.extraPlugins;
        in [{
          assertion = (lib.length names) == (lib.length (lib.unique names));
          message = "services.hermes-agent.extraPlugins: duplicate plugin names detected: ${toString names}. If using fetchFromGitHub, set name = \"plugin-name\" to disambiguate.";
        }];
      }

      # ── Warnings ──────────────────────────────────────────────────────
      # ── Per-user profile for extraPackages ───────────────────────────
      # Wire extraPackages into the hermes user's per-user profile so the
      # login-shell snapshot (which rebuilds PATH from NixOS profiles) sees
      # them.  The systemd service PATH also includes them for direct access.
      (lib.mkIf (cfg.extraPackages != []) {
        # listOf options are merged by the NixOS module system — this appends to
        # any packages the operator assigned to this user externally (e.g. when
        # createUser = false and the user definition lives elsewhere in the config).
        users.users.${cfg.user}.packages = cfg.extraPackages;
      })

      (lib.mkIf (cfg.container.enable && !cfg.addToSystemPackages && cfg.container.hostUsers != []) {
        warnings = [
          ''
            services.hermes-agent: container.enable is true and container.hostUsers
            is set, but addToSystemPackages is false. Without a host-installed hermes
            binary, container routing will not work for interactive users.
            Set addToSystemPackages = true or ensure hermes is on PATH.
          ''
        ];
      })

      # ── Directories ───────────────────────────────────────────────────
      {
        systemd.tmpfiles.rules = [
          "d ${cfg.stateDir}                2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes        2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes/cron   2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes/sessions 2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes/logs   2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes/memories 2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/.hermes/plugins 2770 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.stateDir}/home           0750 ${cfg.user} ${cfg.group} - -"
          "d ${cfg.workingDirectory}         2770 ${cfg.user} ${cfg.group} - -"
        ];
      }

      # ── Activation: link config + auth + documents ────────────────────
      {
        system.activationScripts."hermes-agent-setup" = lib.stringAfter ([ "users" ] ++ lib.optional (config.system.activationScripts ? setupSecrets) "setupSecrets") ''
          # Ensure directories exist (activation runs before tmpfiles)
          mkdir -p ${cfg.stateDir}/.hermes
          mkdir -p ${cfg.stateDir}/home
          mkdir -p ${cfg.workingDirectory}
          chown ${cfg.user}:${cfg.group} ${cfg.stateDir} ${cfg.stateDir}/.hermes ${cfg.stateDir}/home ${cfg.workingDirectory}
          chmod 2770 ${cfg.stateDir} ${cfg.stateDir}/.hermes ${cfg.workingDirectory}
          chmod 0750 ${cfg.stateDir}/home

          # Create subdirs, set setgid + group-writable, migrate existing files.
          # Nix-managed files (config.yaml, .env, .managed) stay 0640/0644.
          find ${cfg.stateDir}/.hermes -maxdepth 1 \
            \( -name "*.db" -o -name "*.db-wal" -o -name "*.db-shm" -o -name "SOUL.md" \) \
            -exec chmod g+rw {} + 2>/dev/null || true
          for _subdir in cron sessions logs memories plugins; do
            mkdir -p "${cfg.stateDir}/.hermes/$_subdir"
            chown ${cfg.user}:${cfg.group} "${cfg.stateDir}/.hermes/$_subdir"
            chmod 2770 "${cfg.stateDir}/.hermes/$_subdir"
            find "${cfg.stateDir}/.hermes/$_subdir" -type f \
              -exec chmod g+rw {} + 2>/dev/null || true
          done

          # Merge Nix settings into existing config.yaml.
          # Preserves user-added keys (skills, streaming, etc.); Nix keys win.
          # If configFile is user-provided (not generated), overwrite instead of merge.
          ${if cfg.configFile != null then ''
            install -o ${cfg.user} -g ${cfg.group} -m 0640 -D ${configFile} ${cfg.stateDir}/.hermes/config.yaml
          '' else ''
            ${configMergeScript} ${generatedConfigFile} ${cfg.stateDir}/.hermes/config.yaml
            chown ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/config.yaml
            chmod 0640 ${cfg.stateDir}/.hermes/config.yaml
          ''}

          # Managed mode marker (so interactive shells also detect NixOS management)
          touch ${cfg.stateDir}/.hermes/.managed
          chown ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/.managed
          chmod 0644 ${cfg.stateDir}/.hermes/.managed

          # Container mode metadata — tells the host CLI to exec into the
          # container instead of running locally. Removed when container mode
          # is disabled so the host CLI falls back to native execution.
          ${if cfg.container.enable then ''
            cat > ${cfg.stateDir}/.hermes/.container-mode <<'HERMES_CONTAINER_MODE_EOF'
    # Written by NixOS activation script. Do not edit manually.
    backend=${cfg.container.backend}
    container_name=${containerName}
    exec_user=${cfg.user}
    hermes_bin=${containerDataDir}/current-package/bin/hermes
    HERMES_CONTAINER_MODE_EOF
            chown ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/.container-mode
            chmod 0644 ${cfg.stateDir}/.hermes/.container-mode
          '' else ''
            rm -f ${cfg.stateDir}/.hermes/.container-mode

            # Remove symlink bridge for hostUsers
            ${lib.concatStringsSep "\n" (map (user:
              let
                userHome = config.users.users.${user}.home;
                symlinkPath = "${userHome}/.hermes";
              in ''
                if [ -L "${symlinkPath}" ] && [ "$(readlink "${symlinkPath}")" = "${cfg.stateDir}/.hermes" ]; then
                  rm -f "${symlinkPath}"
                  echo "hermes-agent: removed symlink ${symlinkPath}"
                fi
              '') cfg.container.hostUsers)}
          ''}

          # ── Symlink bridge for interactive users ───────────────────────
          # Create ~/.hermes -> stateDir/.hermes for each hostUser so the
          # host CLI shares state with the container service.
          # Only runs when container mode is enabled.
          ${lib.optionalString cfg.container.enable
            (lib.concatStringsSep "\n" (map (user:
              let
                userHome = config.users.users.${user}.home;
                symlinkPath = "${userHome}/.hermes";
                target = "${cfg.stateDir}/.hermes";
              in ''
                if [ -d "${symlinkPath}" ] && [ ! -L "${symlinkPath}" ]; then
                  # Real directory — back it up, then create symlink.
                  # (ln -sfn cannot atomically replace a directory.)
                  _backup="${symlinkPath}.bak.$(date +%s)"
                  echo "hermes-agent: backing up existing ${symlinkPath} to $_backup"
                  mv "${symlinkPath}" "$_backup"
                fi
                # For everything else (existing symlink, doesn't exist, etc.)
                # ln -sfn handles it: replaces symlinks, creates new ones.
                ln -sfn "${target}" "${symlinkPath}"
                chown -h ${user}:${cfg.group} "${symlinkPath}"
              '') cfg.container.hostUsers))}

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
            install -o ${cfg.user} -g ${cfg.group} -m 0640 /dev/null "$ENV_FILE"
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
            install -o ${cfg.user} -g ${cfg.group} -m 0640 ${documentDerivation}/${name} ${cfg.workingDirectory}/${name}
          '') cfg.documents)}

        # ── Declarative plugins ─────────────────────────────────────────
        # Remove stale managed symlinks (plugins removed from config)
        find ${cfg.stateDir}/.hermes/plugins -maxdepth 1 -type l -name 'nix-managed-*' -delete 2>/dev/null || true

        ${lib.concatStringsSep "\n" (map (plugin:
          let
            name = lib.getName plugin;
          in ''
            if [ ! -f "${plugin}/plugin.yaml" ]; then
              echo "ERROR: extraPlugins entry '${plugin}' has no plugin.yaml" >&2
              exit 1
            fi
            ln -sfn ${plugin} ${cfg.stateDir}/.hermes/plugins/nix-managed-${name}
            chown -h ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/plugins/nix-managed-${name}
          '') cfg.extraPlugins)}
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
              "${effectivePackage}/bin/hermes"
              "gateway"
            ] ++ cfg.extraArgs);

            Restart = cfg.restart;
            RestartSec = cfg.restartSec;

            # Shared-state: files created by the gateway should be group-writable
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

          path = [
            effectivePackage
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
                --env MESSAGING_CWD=${containerWorkDir} \
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
    ]);
  };
}
