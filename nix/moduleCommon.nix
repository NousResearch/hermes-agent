# nix/moduleCommon.nix — the code that the NixOS and Home Manager modules share
#
# `services.hermes-agent` is the same option set on both modules. Both modules
# get their options, their renderers for config.yaml, .env and documents, and
# their state setup from this file. A NixOS example works on Home Manager
# without a change. An option added here appears on both modules at once.
#
# Each module keeps only the parts that belong to its own scope:
#
#   nixosModules.nix        the service user and group, stateDir,
#                           addToSystemPackages, container mode, tmpfiles,
#                           system.activationScripts, system systemd units
#   homeManagerModules.nix  hermesHome, installPackage, home.activation,
#                           systemd.user.services, launchd.agents
#
# The split is by scope, not by feature. Code that needs root or a system
# identity stays in the NixOS module. All other code is here.
{ lib }:

let
  inherit (lib)
    literalExpression
    mkOption
    types
    ;

  # ── Configuration type ──────────────────────────────────────────────────
  # More than one module can set `settings = { ... }`. recursiveUpdate joins
  # all of the definitions. Without it, only the last definition applies.
  deepConfigType = types.mkOptionType {
    name = "hermes-config-attrs";
    description = "Hermes YAML config (attrset), merged deeply via lib.recursiveUpdate.";
    check = builtins.isAttrs;
    merge = _loc: defs: lib.foldl' lib.recursiveUpdate { } (map (d: d.value) defs);
  };

  # ── MCP server submodule ────────────────────────────────────────────────
  mcpServerType = types.submodule {
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
        type = types.nullOr (
          types.submodule {
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
          }
        );
        default = null;
        description = "Filter which tools are exposed by this server.";
      };

      # Sampling (server-initiated LLM requests)
      sampling = mkOption {
        type = types.nullOr (
          types.submodule {
            options = {
              enabled = mkOption {
                type = types.bool;
                default = true;
                description = "Enable sampling.";
              };
              model = mkOption {
                type = types.nullOr types.str;
                default = null;
                description = "Override model for sampling requests.";
              };
              max_tokens_cap = mkOption {
                type = types.nullOr types.int;
                default = null;
                description = "Max tokens per request.";
              };
              timeout = mkOption {
                type = types.nullOr types.int;
                default = null;
                description = "LLM call timeout in seconds.";
              };
              max_rpm = mkOption {
                type = types.nullOr types.int;
                default = null;
                description = "Max requests per minute.";
              };
              max_tool_rounds = mkOption {
                type = types.nullOr types.int;
                default = null;
                description = "Max tool-use rounds per sampling request.";
              };
              allowed_models = mkOption {
                type = types.listOf types.str;
                default = [ ];
                description = "Models the server is allowed to request.";
              };
              log_level = mkOption {
                type = types.nullOr (
                  types.enum [
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
  };

  # Convert the mcpServers submodules into the shape that config.yaml uses.
  mcpServersToConfig =
    servers:
    lib.mapAttrs (
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
    ) servers;

  documentsType = types.attrsOf (types.either types.str types.path);

  # ── The options that both modules share ─────────────────────────────────
  # `defaultPackage` and `defaultWorkingDirectory` are different on each
  # module, so the caller gives them. All other options are the same.
  sharedOptions =
    {
      defaultPackage,
      defaultPackageText,
      defaultWorkingDirectory,
      defaultWorkingDirectoryText,
    }:
    {
      enable = lib.mkEnableOption "Hermes Agent";

      # ── Package ────────────────────────────────────────────────────────
      package = mkOption {
        type = types.package;
        default = defaultPackage;
        defaultText = defaultPackageText;
        description = "The hermes-agent package to use.";
      };

      workingDirectory = mkOption {
        type = types.str;
        default = defaultWorkingDirectory;
        defaultText = defaultWorkingDirectoryText;
        description = ''
          The working directory for the agent. The module also writes this
          path to config.yaml as `terminal.cwd`. The terminal and file tools
          of the agent use that value.
        '';
      };

      # ── Declarative config ─────────────────────────────────────────────
      configFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          The path to an existing config.yaml. If you set this option, it
          replaces the `settings` option. The module installs the file
          without a change and overwrites all runtime edits on each
          activation.
        '';
      };

      settings = mkOption {
        type = deepConfigType;
        default = { };
        description = ''
          The Hermes configuration, as an attribute set. The module joins the
          definitions from all modules and writes the result to config.yaml.

          The merge into the config.yaml on disk is also a deep merge. These
          keys replace the keys on disk. The module keeps all other keys,
          which includes the keys that `hermes config set` and the settings
          panes of the TUI and the desktop app write at runtime.
        '';
        example = literalExpression ''
          {
            model.default = "anthropic/claude-sonnet-4";
            terminal.backend = "local";
            compression = { enabled = true; threshold = 0.85; };
          }
        '';
      };

      # ── Secrets / environment ──────────────────────────────────────────
      environmentFiles = mkOption {
        # The type is `str` and not `path` for a reason. A Nix path literal
        # copies the secret into the Nix store, which all users can read. Use
        # a runtime path from sops-nix or agenix instead, for example
        # `config.sops.secrets."x".path`.
        type = types.listOf types.str;
        default = [ ];
        description = ''
          The paths to environment files that contain secrets, for example
          API keys and tokens. Activation adds the contents of these files to
          $HERMES_HOME/.env. Hermes reads that file at each start, with
          load_hermes_dotenv().

          Each activation writes .env again from the start. Thus a secret
          file cannot go into .env two times.
        '';
        example = literalExpression ''[ config.sops.secrets."hermes/env".path ]'';
      };

      environment = mkOption {
        type = types.attrsOf types.str;
        default = { };
        description = ''
          Environment variables that are not secret. Activation writes them
          to $HERMES_HOME/.env.

          CAUTION: Do not put secrets in this option. All users can read the
          Nix store. Use environmentFiles for secrets.
        '';
      };

      authFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          The path to a file that gives the first contents of auth.json, the
          OAuth credentials. The module copies the file only when auth.json
          does not exist. Thus a token that Hermes refreshes at runtime stays
          after an activation.
        '';
      };

      authFileForceOverwrite = mkOption {
        type = types.bool;
        default = false;
        description = "Always overwrite auth.json from authFile on activation.";
      };

      # ── Documents ──────────────────────────────────────────────────────
      documents = mkOption {
        type = documentsType;
        default = { };
        description = ''
          Workspace files. The module installs them into workingDirectory.
          Each key is a path relative to that directory, and the module makes
          the necessary subdirectories. Each value is a string or a path.

          Use this option for the project context that the agent reads from
          its working directory, for example AGENTS.md, notes and checklists.
          Hermes reads SOUL.md and memories/ from HERMES_HOME, so put those
          files in `hermesHomeFiles`.

          If you set this option, you must also set `workingDirectory`. The
          default of that option is different on each module. Thus an unset
          default puts these files in a directory that you did not select.
        '';
        example = literalExpression ''
          {
            "AGENTS.md" = ./AGENTS.md;
            "notes/oncall.md" = "Page #infra before restarting anything.";
          }
        '';
      };

      hermesHomeFiles = mkOption {
        type = documentsType;
        default = { };
        description = ''
          Files that the module installs into HERMES_HOME. Each key is a path
          relative to that directory, and the module makes the necessary
          subdirectories. Each value is a string or a path.

          Hermes reads SOUL.md and the memory files from HERMES_HOME and not
          from the working directory. Declare those files here, or Hermes
          does not load them.
        '';
        example = literalExpression ''
          {
            "SOUL.md" = "You are a helpful AI assistant.";
            "memories/USER.md" = ./USER.md;
          }
        '';
      };

      # ── MCP Servers ────────────────────────────────────────────────────
      mcpServers = mkOption {
        type = types.attrsOf mcpServerType;
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

      # ── Packages / plugins ─────────────────────────────────────────────
      extraPackages = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = "More packages on the PATH of the agent. The agent can run these tools.";
      };

      extraPlugins = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = ''
          Directory-based plugin packages to symlink into the hermes plugins
          directory. Each package must contain a plugin.yaml and __init__.py
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

      # ── Service behaviour ──────────────────────────────────────────────
      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra command-line arguments for `hermes gateway`.";
      };

      restart = mkOption {
        type = types.str;
        default = "always";
        description = "The systemd Restart= policy. Darwin does not use this option.";
      };

      restartSec = mkOption {
        type = types.int;
        default = 5;
        description = "The systemd RestartSec= value. Darwin does not use this option.";
      };

      # ── The backend: `hermes serve` or `hermes dashboard` ──────────────
      # `hermes serve` and `hermes dashboard` are the same entry point,
      # hermes_cli.main:cmd_dashboard, with one flag of difference. serve runs
      # without a user interface. dashboard also serves the web application.
      # Both give the /api/ws and /api/pty sockets that Hermes Desktop
      # connects to. They are one process, and you can run only one of them.
      # Thus this option is an enum and not two booleans.
      #
      # The backend does not run the messaging gateway. web_server.py only
      # controls an external gateway, with `hermes gateway restart`. It does
      # not contain a gateway.
      backend = {
        mode = mkOption {
          type = types.enum [
            "none"
            "serve"
            "dashboard"
          ];
          default = "none";
          description = ''
            The backend process to run with the messaging gateway.

            - "none"      — no backend
            - "serve"     — the backend without a user interface. It gives
                            the /api/ws and /api/pty sockets that Hermes
                            Desktop connects to.
            - "dashboard" — all that "serve" gives, and the browser admin
                            panel on the same port

            "dashboard" contains all of "serve".
          '';
        };

        host = mkOption {
          type = types.str;
          default = "127.0.0.1";
          description = ''
            The address that the backend binds to.

            An address other than loopback starts the authentication gate of
            the dashboard. You must then configure credentials, or a client
            cannot connect. The server also refuses each request with a Host
            header that is different from the address that the server bound
            to. This is a defence against DNS rebinding. Bind to the name or
            the address that your clients use.
          '';
        };

        port = mkOption {
          type = types.port;
          default = 9119;
          description = "The port for the backend.";
        };

        extraArgs = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "More command-line arguments for the backend command.";
        };
      };
    };

  # ── Package resolution ──────────────────────────────────────────────────
  effectivePackage =
    cfg:
    if cfg.extraPythonPackages == [ ] && cfg.extraDependencyGroups == [ ] then
      cfg.package
    else
      cfg.package.override { inherit (cfg) extraPythonPackages extraDependencyGroups; };

  # ── The rendered config.yaml ────────────────────────────────────────────
  # YAML contains JSON, so the output of toJSON is a correct config.yaml.
  # terminal.cwd replaces the old MESSAGING_CWD environment variable. The
  # order of the recursiveUpdate lets an explicit settings.terminal.cwd
  # replace the default value.
  mkConfigFiles =
    {
      pkgs,
      cfg,
      workingDirectory,
    }:
    let
      generated = pkgs.writeText "hermes-config.yaml" (
        builtins.toJSON (lib.recursiveUpdate { terminal.cwd = workingDirectory; } cfg.settings)
      );
    in
    {
      inherit generated;
      effective = if cfg.configFile != null then cfg.configFile else generated;
      mergeScript = pkgs.callPackage ./configMergeScript.nix { };
    };

  # ── Documents ───────────────────────────────────────────────────────────
  # A key can contain subdirectories. The tree has the same shape, so the
  # install loop can copy each entry with `install -D`.
  mkDocumentTree =
    { pkgs, documents }:
    pkgs.runCommand "hermes-documents" { } (
      ''
        mkdir -p $out
      ''
      + lib.concatStringsSep "\n" (
        lib.mapAttrsToList (
          name: value:
          let
            dir = builtins.dirOf name;
            mkdir = lib.optionalString (dir != ".") "mkdir -p $out/${dir}";
          in
          if builtins.isPath value || lib.isStorePath value then
            "${mkdir}\ncp ${value} $out/${name}"
          else
            "${mkdir}\ncat > $out/${name} <<'HERMES_DOC_EOF'\n${value}\nHERMES_DOC_EOF"
        ) documents
      )
    );

  # ── How .env is built ───────────────────────────────────────────────────
  # The values that are not secret come from the Nix store. Activation adds
  # the secrets from paths outside the store. This is one command, so it is
  # safe in a dry run. A second activation writes .env again and does not add
  # the same secrets a second time.
  mkEnvScript =
    { pkgs, environment }:
    let
      base = pkgs.writeText "hermes-env-base" (
        lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "${k}=${v}") environment)
        + lib.optionalString (environment != { }) "\n"
      );
    in
    pkgs.writeShellScript "hermes-env-merge" ''
      set -eu

      dest="$1"
      mode="$2"
      shift 2

      install -m "$mode" ${base} "$dest"
      for file in "$@"; do
        if [ -r "$file" ]; then
          printf '\n' >> "$dest"
          cat "$file" >> "$dest"
        else
          echo "hermes-agent: WARNING cannot read environmentFile $file" >&2
        fi
      done
    '';

  # ── State setup ─────────────────────────────────────────────────────────
  # The activation code that both modules run. It makes the directories and
  # installs config.yaml, .env, auth.json, the documents and the plugins. The
  # differences between the two modules are only the install flags for the
  # owner and the file modes. Thus they are arguments, and not a second copy
  # of the script.
  #
  #   run       the command prefix ("" on NixOS, "$DRY_RUN_CMD " on
  #             Home Manager)
  #   owner     "user:group" that owns each file, or null for the user that
  #             runs the activation
  #   modes     the file mode for each kind of file
  mkStateScript =
    {
      pkgs,
      cfg,
      hermesHome,
      workingDirectory,
      # The value to write as terminal.cwd. It is different from
      # workingDirectory only in the container mode of NixOS. There the agent
      # sees the directory at its mount point in the container, but
      # activation writes to the path on the host.
      configWorkingDirectory ? workingDirectory,
      run ? "",
      owner ? null,
      modes,
      stateDirs ? [ ],
      # The module writes this value into the .managed marker. An
      # interactive shell reads the marker, because it does not see the
      # HERMES_MANAGED variable of the service. The value tells the shell
      # which system owns the install and which rebuild command to name.
      managedSystem ? "nixos",
    }:
    let
      installFlags = lib.optionalString (owner != null) (
        let
          parts = lib.splitString ":" owner;
        in
        "-o ${lib.head parts} -g ${lib.last parts}"
      );
      configFiles = mkConfigFiles {
        inherit pkgs cfg;
        workingDirectory = configWorkingDirectory;
      };
      envScript = mkEnvScript {
        inherit pkgs;
        inherit (cfg) environment;
      };
      documentTree = mkDocumentTree {
        inherit pkgs;
        inherit (cfg) documents;
      };
      homeDocumentTree = mkDocumentTree {
        inherit pkgs;
        documents = cfg.hermesHomeFiles;
      };

      inst = "${run}install ${installFlags}";

      installDocuments =
        tree: root: docs:
        lib.concatStringsSep "\n" (
          lib.mapAttrsToList (
            name: _value: "${inst} -m ${modes.document} -D ${tree}/${name} ${root}/${name}"
          ) docs
        );
    in
    ''
      # Directories. The service units and Hermes make most of these
      # directories when they first need them. Activation makes them here so
      # that the first activation sets the correct owner and mode, and does
      # not use the umask.
      ${run}mkdir -p ${
        lib.escapeShellArgs (
          [
            hermesHome
            workingDirectory
          ]
          ++ map (d: "${hermesHome}/${d}") stateDirs
        )
      }

      # config.yaml: merge the Nix settings into the file on disk. Hermes
      # writes this file at runtime. A read-only symlink to the Nix store
      # breaks each save from the application. The Nix keys replace the keys
      # on disk, and the module keeps all other keys.
      ${
        if cfg.configFile != null then
          "${inst} -m ${modes.config} -D ${configFiles.effective} ${hermesHome}/config.yaml"
        else
          ''
            ${run}${configFiles.mergeScript} ${configFiles.generated} ${hermesHome}/config.yaml
            ${run}chmod ${modes.config} ${hermesHome}/config.yaml
          ''
      }

      # The managed-mode marker. It makes an interactive shell also refuse to
      # change the configuration that Nix owns.
      ${inst} -m ${modes.managed} ${pkgs.writeText "hermes-managed" managedSystem} ${hermesHome}/.managed

      ${lib.optionalString (cfg.environment != { } || cfg.environmentFiles != [ ]) ''
        ${run}${envScript} ${hermesHome}/.env ${modes.env} ${lib.escapeShellArgs cfg.environmentFiles}
        ${lib.optionalString (owner != null) "${run}chown ${owner} ${hermesHome}/.env"}
      ''}

      ${lib.optionalString (cfg.authFile != null) (
        if cfg.authFileForceOverwrite then
          "${inst} -m ${modes.auth} ${cfg.authFile} ${hermesHome}/auth.json"
        else
          ''
            if [ ! -e ${hermesHome}/auth.json ]; then
              ${inst} -m ${modes.auth} ${cfg.authFile} ${hermesHome}/auth.json
            fi
          ''
      )}

      ${installDocuments documentTree workingDirectory cfg.documents}
      ${installDocuments homeDocumentTree hermesHome cfg.hermesHomeFiles}

      # Declarative plugins. Activation first deletes the old managed
      # symlinks. Thus a plugin that you remove from the configuration also
      # goes away from the plugins directory.
      ${run}find ${hermesHome}/plugins -maxdepth 1 -type l -name 'nix-managed-*' -delete 2>/dev/null || true
      ${lib.concatMapStringsSep "\n" (plugin: ''
        if [ ! -f ${plugin}/plugin.yaml ]; then
          echo "hermes-agent: ERROR extraPlugins entry '${plugin}' has no plugin.yaml" >&2
          exit 1
        fi
        ${run}ln -sfn ${plugin} ${hermesHome}/plugins/nix-managed-${lib.getName plugin}
      '') cfg.extraPlugins}
    '';

  # ── Process argv ────────────────────────────────────────────────────────
  gatewayArgv =
    cfg:
    [
      "${effectivePackage cfg}/bin/hermes"
      "gateway"
    ]
    ++ cfg.extraArgs;

  backendArgv =
    cfg:
    [
      "${effectivePackage cfg}/bin/hermes"
      cfg.backend.mode
      "--host"
      cfg.backend.host
      "--port"
      (toString cfg.backend.port)
      # CAUTION: A service must not try to open a browser when it starts.
      "--no-open"
    ]
    ++ cfg.backend.extraArgs;

  backendDescription =
    cfg:
    if cfg.backend.mode == "dashboard" then
      "Hermes Agent web dashboard and desktop backend"
    else
      "Hermes Agent backend for Hermes Desktop";

  # The environment that each Hermes process needs, from either module.
  #
  # managedSystem gives the value of HERMES_MANAGED. The CLI reads that
  # variable to refuse a configuration change that it cannot keep, and to
  # name the correct rebuild command. The answer is different on each module,
  # so each module gives its own value.
  processEnvironment =
    {
      hermesHome,
      managedSystem ? "true",
    }:
    {
      HERMES_HOME = hermesHome;
      HERMES_MANAGED = managedSystem;
    };

  processPath =
    { pkgs, cfg }:
    [
      (effectivePackage cfg)
      pkgs.bash
      pkgs.coreutils
      pkgs.git
    ]
    ++ cfg.extraPackages;

  # workingDirectory has a default on both modules, but a bad one. It is the
  # home directory of the user on Home Manager, and ${stateDir}/workspace on
  # NixOS. A user who declares files without a directory therefore gets a
  # place that the user did not select. The place is also different on each
  # module. The modules refuse that combination.
  #
  # The test is on the priority of the option and not on its value. An option
  # that nothing sets keeps the priority of its own default, and each
  # definition from a user is stronger. Thus a directory that has the same
  # text as the default is still a selection, and so is a mkDefault. A
  # comparison of values detects neither.
  workspaceFilesAssertions =
    {
      cfg,
      opt,
      optionPath,
    }:
    let
      untouched = (lib.mkOptionDefault null).priority; # 1500, derived not spelled
    in
    [
      {
        assertion = cfg.documents == { } || opt.highestPrio < untouched;
        message = ''
          ${optionPath}.documents needs an explicit ${optionPath}.workingDirectory.

          The files go into workingDirectory. The default of that option is
          different on each module, so an unset default puts the files in a
          directory that you did not select. Set the directory:

            ${optionPath}.workingDirectory = "/path/you/want";

          To give Hermes an identity and a memory, use
          ${optionPath}.hermesHomeFiles instead. Those files go to
          HERMES_HOME. Hermes reads SOUL.md and memories/ only from there.
        '';
      }
    ];

  # Two plugins with the same name use one nix-managed-<name> symlink. One of
  # the plugins then disappears without a message. Both modules assert
  # against this condition.
  pluginNameAssertions =
    { cfg, optionPath }:
    let
      names = map lib.getName cfg.extraPlugins;
    in
    [
      {
        assertion = (lib.length names) == (lib.length (lib.unique names));
        message = "${optionPath}.extraPlugins: duplicate plugin names detected: ${toString names}. If using fetchFromGitHub, set name = \"plugin-name\" to disambiguate.";
      }
    ];

  # The subdirectories of HERMES_HOME that both modules make.
  stateSubdirs = [
    "cron"
    "sessions"
    "logs"
    "memories"
    "plugins"
  ];
in
{
  inherit
    backendArgv
    backendDescription
    deepConfigType
    effectivePackage
    gatewayArgv
    mcpServerType
    mcpServersToConfig
    mkConfigFiles
    mkDocumentTree
    mkEnvScript
    mkStateScript
    pluginNameAssertions
    processEnvironment
    processPath
    sharedOptions
    stateSubdirs
    workspaceFilesAssertions
    ;
}
