# nix/homeManagerModules.nix — Home Manager module for hermes-agent
#
# Runs hermes-agent as a per-user service under home-manager.
#   Linux:   systemd.user.services
#   macOS:   launchd.agents
# Container mode is not supported (requires system-level docker/podman).
#
{ inputs, ... }:

{
  flake.homeManagerModules.default =
    {
      config,
      lib,
      pkgs,
      ...
    }:

    let
      cfg = config.programs.hermes-agent;
      hermes-agent = inputs.self.packages.${pkgs.system}.default;

      # Deep-merge config type (same as nixosModules.nix)
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
      envFileContent = lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment);

      # Build documents derivation
      documentDerivation = pkgs.runCommand "hermes-documents" { } (
        ''
          mkdir -p $out
        ''
        + lib.concatStringsSep "\n" (
          lib.mapAttrsToList (
            name: value:
            if builtins.isPath value || lib.isStorePath value then
              "cp ${value} $out/${name}"
            else
              "cat > $out/${name} <<'HERMES_DOC_EOF'\n${value}\nHERMES_DOC_EOF"
          ) cfg.documents
        )
      );

    in
    {
      options.programs.hermes-agent = {
        enable = lib.mkEnableOption "Hermes Agent gateway service";

        # ── Package ──────────────────────────────────────────────────────────
        package = lib.mkOption {
          type = lib.types.package;
          default = hermes-agent;
          description = "The hermes-agent package to use.";
        };

        # ── Directories ──────────────────────────────────────────────────────
        stateDir = lib.mkOption {
          type = lib.types.str;
          default = "${config.home.homeDirectory}/.local/share/hermes";
          defaultText = lib.literalExpression ''"''${config.home.homeDirectory}/.local/share/hermes"'';
          description = "State directory. Contains .hermes/ subdir (HERMES_HOME).";
        };

        workingDirectory = lib.mkOption {
          type = lib.types.str;
          default = "${cfg.stateDir}/workspace";
          defaultText = lib.literalExpression ''"''${cfg.stateDir}/workspace"'';
          description = "Working directory for the agent (MESSAGING_CWD).";
        };

        # ── Declarative config ───────────────────────────────────────────────
        configFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = ''
            Path to an existing config.yaml. If set, takes precedence over
            the declarative `settings` option.
          '';
        };

        settings = lib.mkOption {
          type = deepConfigType;
          default = { };
          description = ''
            Declarative Hermes config (attrset). Deep-merged across module
            definitions and rendered as config.yaml.
          '';
          example = lib.literalExpression ''
            {
              model = "anthropic/claude-sonnet-4";
              terminal.backend = "local";
              compression = { enabled = true; threshold = 0.85; };
              toolsets = [ "all" ];
            }
          '';
        };

        # ── Secrets / environment ────────────────────────────────────────────
        environmentFiles = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [ ];
          description = ''
            Paths to environment files containing secrets (API keys, tokens).
            Contents are merged into $HERMES_HOME/.env at activation time.
            Hermes reads this file on every startup via load_hermes_dotenv().
          '';
        };

        environment = lib.mkOption {
          type = lib.types.attrsOf lib.types.str;
          default = { };
          description = ''
            Non-secret environment variables. Merged into $HERMES_HOME/.env
            at activation time. Do NOT put secrets here — use environmentFiles.
          '';
        };

        authFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = ''
            Path to an auth.json seed file (OAuth credentials).
            Only copied on first deploy — existing auth.json is preserved.
          '';
        };

        authFileForceOverwrite = lib.mkOption {
          type = lib.types.bool;
          default = false;
          description = "Always overwrite auth.json from authFile on activation.";
        };

        # ── Documents ────────────────────────────────────────────────────────
        documents = lib.mkOption {
          type = lib.types.attrsOf (lib.types.either lib.types.str lib.types.path);
          default = { };
          description = ''
            Workspace files (SOUL.md, USER.md, etc.). Keys are filenames,
            values are inline strings or paths. Installed into workingDirectory.
          '';
          example = lib.literalExpression ''
            {
              "SOUL.md" = "You are a helpful AI assistant.";
              "USER.md" = ./documents/USER.md;
            }
          '';
        };

        # ── MCP Servers ──────────────────────────────────────────────────────
        mcpServers = lib.mkOption {
          type = lib.types.attrsOf (
            lib.types.submodule {
              options = {
                # Stdio transport
                command = lib.mkOption {
                  type = lib.types.nullOr lib.types.str;
                  default = null;
                  description = "MCP server command (stdio transport).";
                };
                args = lib.mkOption {
                  type = lib.types.listOf lib.types.str;
                  default = [ ];
                  description = "Command-line arguments (stdio transport).";
                };
                env = lib.mkOption {
                  type = lib.types.attrsOf lib.types.str;
                  default = { };
                  description = "Environment variables for the server process (stdio transport).";
                };

                # HTTP/StreamableHTTP transport
                url = lib.mkOption {
                  type = lib.types.nullOr lib.types.str;
                  default = null;
                  description = "MCP server endpoint URL (HTTP/StreamableHTTP transport).";
                };
                headers = lib.mkOption {
                  type = lib.types.attrsOf lib.types.str;
                  default = { };
                  description = "HTTP headers, e.g. for authentication (HTTP transport).";
                };

                # Authentication
                auth = lib.mkOption {
                  type = lib.types.nullOr (lib.types.enum [ "oauth" ]);
                  default = null;
                  description = ''
                    Authentication method. Set to "oauth" for OAuth 2.1 PKCE flow
                    (remote MCP servers). Tokens are stored in $HERMES_HOME/mcp-tokens/.
                  '';
                };

                # Enable/disable
                enabled = lib.mkOption {
                  type = lib.types.bool;
                  default = true;
                  description = "Enable or disable this MCP server.";
                };

                # Common options
                timeout = lib.mkOption {
                  type = lib.types.nullOr lib.types.int;
                  default = null;
                  description = "Tool call timeout in seconds (default: 120).";
                };
                connect_timeout = lib.mkOption {
                  type = lib.types.nullOr lib.types.int;
                  default = null;
                  description = "Initial connection timeout in seconds (default: 60).";
                };

                # Tool filtering
                tools = lib.mkOption {
                  type = lib.types.nullOr (
                    lib.types.submodule {
                      options = {
                        include = lib.mkOption {
                          type = lib.types.listOf lib.types.str;
                          default = [ ];
                          description = "Tool allowlist — only these tools are registered.";
                        };
                        exclude = lib.mkOption {
                          type = lib.types.listOf lib.types.str;
                          default = [ ];
                          description = "Tool blocklist — these tools are hidden.";
                        };
                      };
                    }
                  );
                  default = null;
                  description = "Filter which tools are exposed by this server.";
                };

                # Sampling (server-initiated LLM requests)
                sampling = lib.mkOption {
                  type = lib.types.nullOr (
                    lib.types.submodule {
                      options = {
                        enabled = lib.mkOption {
                          type = lib.types.bool;
                          default = true;
                          description = "Enable sampling.";
                        };
                        model = lib.mkOption {
                          type = lib.types.nullOr lib.types.str;
                          default = null;
                          description = "Override model for sampling requests.";
                        };
                        max_tokens_cap = lib.mkOption {
                          type = lib.types.nullOr lib.types.int;
                          default = null;
                          description = "Max tokens per request.";
                        };
                        timeout = lib.mkOption {
                          type = lib.types.nullOr lib.types.int;
                          default = null;
                          description = "LLM call timeout in seconds.";
                        };
                        max_rpm = lib.mkOption {
                          type = lib.types.nullOr lib.types.int;
                          default = null;
                          description = "Max requests per minute.";
                        };
                        max_tool_rounds = lib.mkOption {
                          type = lib.types.nullOr lib.types.int;
                          default = null;
                          description = "Max tool-use rounds per sampling request.";
                        };
                        allowed_models = lib.mkOption {
                          type = lib.types.listOf lib.types.str;
                          default = [ ];
                          description = "Models the server is allowed to request.";
                        };
                        log_level = lib.mkOption {
                          type = lib.types.nullOr (
                            lib.types.enum [
                              "debug"
                              "info"
                              "warning"
                            ]
                          );
                          default = null;
                          description = "Audit log level for sampling requests.";
                        };
                      };
                    }
                  );
                  default = null;
                  description = "Sampling configuration for server-initiated LLM requests.";
                };
              };
            }
          );
          default = { };
          description = ''
            MCP server configurations (merged into settings.mcp_servers).
            Each server uses either stdio (command/args) or HTTP (url) transport.
          '';
          example = lib.literalExpression ''
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
        extraArgs = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [ ];
          description = "Extra command-line arguments for `hermes gateway`.";
        };

        extraPackages = lib.mkOption {
          type = lib.types.listOf lib.types.package;
          default = [ ];
          description = "Extra packages available on PATH.";
        };

        restart = lib.mkOption {
          type = lib.types.str;
          default = "always";
          description = ''
            Restart policy. Maps to systemd Restart= (Linux) or
            launchd KeepAlive (macOS).
          '';
        };

        restartSec = lib.mkOption {
          type = lib.types.int;
          default = 5;
          description = ''
            Seconds between restart attempts. Maps to systemd RestartSec=
            (Linux) or launchd ThrottleInterval (macOS, minimum 10s).
          '';
        };
      };

      config = lib.mkIf cfg.enable (
        lib.mkMerge [

          # ── Merge MCP servers into settings ────────────────────────────────
          (lib.mkIf (cfg.mcpServers != { }) {
            programs.hermes-agent.settings.mcp_servers = lib.mapAttrs (
              _name: srv:
              # Stdio transport
              lib.optionalAttrs (srv.command != null) { inherit (srv) command args; }
              // lib.optionalAttrs (srv.env != { }) { inherit (srv) env; }
              # HTTP transport
              // lib.optionalAttrs (srv.url != null) { inherit (srv) url; }
              // lib.optionalAttrs (srv.headers != { }) { inherit (srv) headers; }
              # Auth
              // lib.optionalAttrs (srv.auth != null) { inherit (srv) auth; }
              # Enable/disable
              // {
                inherit (srv) enabled;
              }
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
                  inherit (srv.sampling)
                    enabled
                    model
                    max_tokens_cap
                    timeout
                    max_rpm
                    max_tool_rounds
                    allowed_models
                    log_level
                    ;
                };
              }
            ) cfg.mcpServers;
          })

          # ── Packages & environment ─────────────────────────────────────────
          {
            home.packages = [ cfg.package ] ++ cfg.extraPackages;
            home.sessionVariables.HERMES_HOME = "${cfg.stateDir}/.hermes";
          }

          # ── Activation: directories, config, auth, env, documents ─────────
          {
            home.activation.hermesAgentSetup = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
              # Ensure directories exist
              mkdir -p ${cfg.stateDir}/.hermes
              mkdir -p ${cfg.stateDir}/.hermes/cron
              mkdir -p ${cfg.stateDir}/.hermes/sessions
              mkdir -p ${cfg.stateDir}/.hermes/logs
              mkdir -p ${cfg.stateDir}/.hermes/memories
              mkdir -p ${cfg.workingDirectory}

              # Merge Nix settings into existing config.yaml.
              # Preserves user-added keys; Nix keys win.
              ${
                if cfg.configFile != null then
                  ''
                    install -m 0640 -D ${configFile} ${cfg.stateDir}/.hermes/config.yaml
                  ''
                else
                  ''
                    ${configMergeScript} ${generatedConfigFile} ${cfg.stateDir}/.hermes/config.yaml
                    chmod 0640 ${cfg.stateDir}/.hermes/config.yaml
                  ''
              }

              # Managed mode marker
              touch ${cfg.stateDir}/.hermes/.managed

              # Seed auth file if provided
              ${lib.optionalString (cfg.authFile != null) (
                if cfg.authFileForceOverwrite then
                  ''
                    install -m 0600 ${cfg.authFile} ${cfg.stateDir}/.hermes/auth.json
                  ''
                else
                  ''
                    if [ ! -f ${cfg.stateDir}/.hermes/auth.json ]; then
                      install -m 0600 ${cfg.authFile} ${cfg.stateDir}/.hermes/auth.json
                    fi
                  ''
              )}

              # Seed .env from non-secret environment only.
              # Secret environmentFiles are appended at service start via ExecStartPre,
              # after agenix/sops have decrypted them.
              ${lib.optionalString (cfg.environment != { }) ''
                ENV_FILE="${cfg.stateDir}/.hermes/.env"
                install -m 0640 /dev/null "$ENV_FILE"
                cat > "$ENV_FILE" <<'HERMES_NIX_ENV_EOF'
                ${envFileContent}
                HERMES_NIX_ENV_EOF
              ''}

              # Link documents into workspace
              ${lib.concatStringsSep "\n" (
                lib.mapAttrsToList (name: _value: ''
                  install -m 0640 ${documentDerivation}/${name} ${cfg.workingDirectory}/${name}
                '') cfg.documents
              )}
            '';
          }

          # ── Linux: systemd user service ────────────────────────────────────
          (lib.mkIf pkgs.stdenv.hostPlatform.isLinux {
            systemd.user.services.hermes-agent = {
              Unit = {
                Description = "Hermes Agent Gateway";
                After = [ "network-online.target" ] ++ lib.optional (cfg.environmentFiles != [ ]) "agenix.service";
                Wants = [ "network-online.target" ] ++ lib.optional (cfg.environmentFiles != [ ]) "agenix.service";
              };

              Service =
                let
                  servicePath = lib.makeBinPath (
                    [
                      cfg.package
                      pkgs.bash
                      pkgs.coreutils
                      pkgs.git
                    ]
                    ++ cfg.extraPackages
                  );

                  # Script to append secret environmentFiles to .env at service
                  # start time. Runs after agenix/sops have decrypted secrets,
                  # unlike the activation script which runs before systemd.
                  envSeedScript = pkgs.writeShellScript "hermes-seed-envfiles" ''
                    ENV_FILE="${cfg.stateDir}/.hermes/.env"
                    ${lib.concatStringsSep "\n" (
                      map (f: ''
                        if [ -f "${f}" ]; then
                          echo "" >> "$ENV_FILE"
                          cat "${f}" >> "$ENV_FILE"
                        fi
                      '') cfg.environmentFiles
                    )}
                  '';
                in
                lib.mkMerge [
                  {
                    Environment = [
                      "HOME=${config.home.homeDirectory}"
                      "HERMES_HOME=${cfg.stateDir}/.hermes"
                      "HERMES_MANAGED=true"
                      "MESSAGING_CWD=${cfg.workingDirectory}"
                      ("PATH=" + servicePath + "\${PATH:+:$PATH}")
                    ];

                    ExecStart = lib.concatStringsSep " " (
                      [
                        "${cfg.package}/bin/hermes"
                        "gateway"
                      ]
                      ++ cfg.extraArgs
                    );

                    Restart = cfg.restart;
                    RestartSec = cfg.restartSec;

                    WorkingDirectory = cfg.workingDirectory;
                  }

                  (lib.mkIf (cfg.environmentFiles != [ ]) {
                    ExecStartPre = "${envSeedScript}";
                  })
                ];

              Install.WantedBy = [ "default.target" ];
            };
          })

          # ── Darwin: launchd agent ──────────────────────────────────────────
          (lib.mkIf pkgs.stdenv.hostPlatform.isDarwin {
            launchd.agents.hermes-agent = {
              enable = true;
              config =
                let
                  # launchd has no ExecStartPre, so we wrap env seeding +
                  # gateway into a single script.
                  gatewayWrapper = pkgs.writeShellScript "hermes-gateway-wrapper" ''
                    # Append secret environmentFiles to .env (agenix decrypts
                    # via its own launchd agent with RunAtLoad, same as us —
                    # the script gracefully skips missing files).
                    ${lib.optionalString (cfg.environmentFiles != [ ]) ''
                      ENV_FILE="${cfg.stateDir}/.hermes/.env"
                      ${lib.concatStringsSep "\n" (
                        map (f: ''
                          if [ -f "${f}" ]; then
                            echo "" >> "$ENV_FILE"
                            cat "${f}" >> "$ENV_FILE"
                          fi
                        '') cfg.environmentFiles
                      )}
                    ''}

                    exec ${
                      lib.concatStringsSep " " (
                        [
                          "${cfg.package}/bin/hermes"
                          "gateway"
                        ]
                        ++ cfg.extraArgs
                      )
                    }
                  '';

                  # Map systemd restart policy to launchd KeepAlive.
                  keepAlive =
                    if cfg.restart == "always" then
                      true
                    else if cfg.restart == "on-failure" then
                      { SuccessfulExit = false; }
                    else
                      false;
                in
                {
                  ProgramArguments = [ "${gatewayWrapper}" ];
                  RunAtLoad = true;
                  KeepAlive = keepAlive;
                  ThrottleInterval = cfg.restartSec;
                  WorkingDirectory = cfg.workingDirectory;
                  EnvironmentVariables = {
                    HOME = config.home.homeDirectory;
                    HERMES_HOME = "${cfg.stateDir}/.hermes";
                    HERMES_MANAGED = "true";
                    MESSAGING_CWD = cfg.workingDirectory;
                    PATH = lib.makeBinPath (
                      [
                        cfg.package
                        pkgs.bash
                        pkgs.coreutils
                        pkgs.git
                      ]
                      ++ cfg.extraPackages
                    );
                  };
                  StandardOutPath = "${cfg.stateDir}/.hermes/logs/launchd-stdout.log";
                  StandardErrorPath = "${cfg.stateDir}/.hermes/logs/launchd-stderr.log";
                  ProcessType = "Background";
                };
            };
          })
        ]
      );
    };
}
