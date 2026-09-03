# nix/homeManagerModules.nix — the Home Manager module for hermes-agent
#
# This module is the user-level equivalent of nixosModules.default. Hermes is
# an agent for one person. The credentials, the memory, the sessions and the
# cron jobs all belong to that person. Thus a user-level module is correct on
# each distribution, and not only on NixOS.
#
# `services.hermes-agent` is the same option set on both modules. All of the
# options except the system-level ones come from nix/moduleCommon.nix, so an
# example from the NixOS documentation works here without a change. Only the
# necessary parts are different:
#
#   removed   user, group, createUser  — Home Manager runs as the user
#   removed   container.*              — it needs root and the Docker socket
#   removed   UMask 0007               — that mode shares state with a UNIX
#                                        group, but this state has one user
#   changed   systemd.services         -> systemd.user.services or
#                                        launchd.agents
#   changed   system.activationScripts -> home.activation
#   changed   addToSystemPackages      -> programs.hermes-agent.enable and
#                                        home.sessionVariables
#   added     programs.hermes-agent    the CLI and the desktop application,
#                                      because Home Manager separates an
#                                      installation from a daemon
#   changed   stateDir (+ "/.hermes")  -> hermesHome, set directly
#
# To use the module:
#   imports = [ hermes-agent.homeManagerModules.default ];
#   programs.hermes-agent = {
#     enable = true;          # the hermes CLI on your PATH
#     desktop.enable = true;  # the Electron application and a launcher
#   };
#   services.hermes-agent = {
#     enable = true;
#     gateway.enable = true;
#     settings.model.default = "anthropic/claude-sonnet-4";
#     environmentFiles = [ config.sops.secrets."hermes/env".path ];
#   };
#
# CAUTION: Enable linger for the account. Without linger, systemd stops the
# user manager at logout, and both units stop with it. Home Manager cannot
# run `loginctl enable-linger`. On NixOS, set
#   users.users.<name>.linger = true;
# On other systems, run `loginctl enable-linger <name>` one time.
{ inputs, ... }:
{
  flake.homeManagerModules.default =
    {
      config,
      lib,
      options,
      pkgs,
      ...
    }:

    let
      cfg = config.services.hermes-agent;
      cfgPrograms = config.programs.hermes-agent;
      common = import ./moduleCommon.nix { inherit lib; };

      effectivePackage = common.effectivePackage cfg;
      hermes-agent = inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default;

      inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

      processEnvironment = common.processEnvironment {
        inherit (cfg) hermesHome;
        # The CLI reads this value and names it when it refuses a
        # configuration change.
        managedSystem = "home-manager";
      };

      profileHome = name: "${cfg.hermesHome}/profiles/${name}";
      profileCfg =
        profile:
        profile
        // {
          inherit (cfg)
            package
            extraDependencyGroups
            extraPackages
            extraPythonPackages
            ;
          settings = lib.recursiveUpdate profile.settings (
            lib.optionalAttrs (profile.mcpServers != { }) {
              mcp_servers = common.mcpServersToConfig profile.mcpServers;
            }
          );
        };
      profileType = lib.types.submodule (
        { name, ... }:
        {
          options = common.profileOptions {
            defaultPackage = hermes-agent;
            defaultPackageText = lib.literalExpression "config.services.hermes-agent.package";
            defaultWorkingDirectory = "${profileHome name}/workspace";
            defaultWorkingDirectoryText = lib.literalExpression ''"config.services.hermes-agent.hermesHome/profiles/${name}/workspace"'';
          };
        }
      );

      profileStateScripts = lib.mapAttrsToList (
        name: profile:
        common.mkStateScript {
          inherit pkgs;
          cfg = profileCfg profile;
          hermesHome = profileHome name;
          inherit (profile) workingDirectory;
          run = "$DRY_RUN_CMD ";
          stateDirs = common.stateSubdirs;
          managedSystem = "home-manager";
          modes = {
            config = "0600";
            env = "0600";
            managed = "0600";
            auth = "0600";
            document = "0600";
          };
        }
      ) cfg.profiles;

      profilePluginAssertions = lib.concatLists (
        lib.mapAttrsToList (
          name: profile:
          common.pluginNameAssertions {
            cfg = profile;
            optionPath = "services.hermes-agent.profiles.${name}";
          }
        ) cfg.profiles
      );

      # ── The desktop launcher ───────────────────────────────────────────
      # A GUI launcher reads no shell profile, so home.sessionVariables does
      # not reach it, and the application would open ~/.hermes while the
      # services use hermesHome. Thus the launcher carries the value itself.
      #
      # HERMES_MANAGED rides along only when the services are enabled. That
      # variable makes the CLI refuse a configuration change and name the
      # rebuild command. A person who enables `programs.` alone has no
      # activation and no managed configuration, so the application must not
      # claim one and refuse an edit that nothing else owns.
      desktopEnvironment = {
        HERMES_HOME = cfg.hermesHome;
      }
      // lib.optionalAttrs cfg.enable {
        inherit (processEnvironment) HERMES_MANAGED;
      }
      // lib.optionalAttrs desktopUsesService {
        HERMES_DESKTOP_REMOTE_URL = "http://${cfg.backend.host}:${toString cfg.backend.port}";
      };

      # The application reaches the backend of the service only when there is
      # a backend to reach AND a shared token to present with. Without the
      # token the desktop resolver throws ("HERMES_DESKTOP_REMOTE_URL is set
      # but HERMES_DESKTOP_REMOTE_TOKEN is not"), so the two variables travel
      # together or not at all.
      desktopUsesService = cfg.enable && cfg.backend.mode != "none" && cfg.backend.sessionTokenFile != null;

      # The token is read at start time and never with `--set`. makeWrapper
      # writes a --set value into the Nix store, which all users can read.
      desktopRun = lib.optional desktopUsesService ''
        if [ -r ${lib.escapeShellArg cfg.backend.sessionTokenFile} ]; then
          HERMES_DESKTOP_REMOTE_TOKEN="$(tr -d '\r\n' < ${lib.escapeShellArg cfg.backend.sessionTokenFile})"
          export HERMES_DESKTOP_REMOTE_TOKEN
        else
          echo "hermes-desktop: cannot read the session token at ${cfg.backend.sessionTokenFile}." >&2
          echo "hermes-desktop: the application starts its own backend instead of the one of the service." >&2
        fi
      '';

      # `override`, and not `overrideAttrs`: the values go into the wrapper
      # that the installPhase writes, and not into a derivation attribute.
      desktopPackage = cfgPrograms.desktop.package.override {
        extraEnv = desktopEnvironment;
        extraRun = desktopRun;
      };

      # The systemd unit that the gateway and the backend both start from.
      mkUnit =
        {
          description,
          argv,
          agentCfg ? cfg,
          hermesHome ? cfg.hermesHome,
        }:
        let
          environment = common.processEnvironment {
            inherit hermesHome;
            managedSystem = "home-manager";
          };
          path = lib.makeBinPath (
            common.processPath {
              inherit pkgs;
              cfg = agentCfg;
            }
          );
        in
        {
          Unit = {
            Description = description;
            # Do not use network-online.target here. That is a system target.
            # A user unit that orders against it has no effect, and systemd
            # gives no message.
            After = [ "default.target" ];
          };
          Install.WantedBy = [ "default.target" ];
          Service = {
            Type = "simple";
            Environment = (lib.mapAttrsToList (k: v: "${k}=${v}") environment) ++ [
              "PATH=${path}"
            ];
            ExecStart = lib.escapeShellArgs argv;
            WorkingDirectory = agentCfg.workingDirectory;
            Restart = agentCfg.restart;
            RestartSec = agentCfg.restartSec;
            # This state has one user. Keep it private. The NixOS module uses
            # 0007 to share the state with a UNIX group.
            UMask = "0077";
            NoNewPrivileges = true;
            PrivateTmp = true;
          };
        };

      mkAgent =
        {
          argv,
          logName,
          agentCfg ? cfg,
          hermesHome ? cfg.hermesHome,
        }:
        let
          environment = common.processEnvironment {
            inherit hermesHome;
            managedSystem = "home-manager";
          };
          path = lib.makeBinPath (
            common.processPath {
              inherit pkgs;
              cfg = agentCfg;
            }
          );
        in
        {
          enable = true;
          config = {
            Label = "org.nix-community.home.${logName}";
            ProgramArguments = argv;
            EnvironmentVariables = environment // {
              PATH = "${path}:/usr/bin:/bin:/usr/sbin:/sbin";
            };
            WorkingDirectory = agentCfg.workingDirectory;
            RunAtLoad = true;
            KeepAlive =
              if agentCfg.restart == "always" then
                true
              else
                {
                  SuccessfulExit = false;
                  Crashed = true;
                };
            ThrottleInterval = agentCfg.restartSec;
            StandardOutPath = "${config.home.homeDirectory}/Library/Logs/${logName}.log";
            StandardErrorPath = "${config.home.homeDirectory}/Library/Logs/${logName}.err.log";
            ProcessType = "Background";
          };
        };

    in
    {
      # ── programs.hermes-agent — the installation ───────────────────────
      # Home Manager separates "install this application for me" from "run
      # this daemon". Hermes needs both, and a person can want one without
      # the other: an application with no gateway, or a headless gateway on
      # a machine with no display.
      #
      # `services.hermes-agent` stays the authority for the state and the
      # configuration. This module reads hermesHome and the backend address
      # from it, and never the reverse.
      options.programs.hermes-agent = {
        enable = lib.mkEnableOption ''
          the Hermes Agent command line application.

          This adds `hermes` to home.packages, and exports HERMES_HOME with
          home.sessionVariables. An interactive shell then uses the same
          state as `services.hermes-agent`
        '';

        package = lib.mkOption {
          type = lib.types.package;
          default = effectivePackage;
          defaultText = lib.literalExpression "config.services.hermes-agent.package";
          description = ''
            The hermes-agent package to install.

            The default follows `services.hermes-agent.package`, and applies
            `extraPythonPackages` and `extraDependencyGroups` from that
            module. Thus the command line and the services are one build,
            and a plugin that the services can load is a plugin that your
            shell can load.
          '';
        };

        desktop = {
          enable = lib.mkEnableOption ''
            the Hermes Desktop application (Electron).

            This adds `hermes-desktop` to home.packages, with an XDG
            launcher entry on Linux. The launcher starts the same Hermes
            runtime that `package` gives, and reads the HERMES_HOME of
            `services.hermes-agent`. Thus the application, the interactive
            shell and the services share one state directory.

            The Electron application carries its own Hermes runtime with
            the usual distribution. This module gives it the Nix package
            instead, with HERMES_DESKTOP_HERMES. It installs no second copy
            of Hermes, and it downloads nothing on the first start
          '';

          package = lib.mkOption {
            type = lib.types.package;
            default = cfgPrograms.package.hermesDesktop;
            defaultText = lib.literalExpression "config.programs.hermes-agent.package.hermesDesktop";
            description = ''
              The hermes-desktop package to use.

              The default follows `package`, and thus also
              `services.hermes-agent.extraPythonPackages` and
              `extraDependencyGroups`, because the desktop application is a
              passthru of the agent package. A package that you set here
              carries its own Hermes runtime, and this module cannot make
              it agree with the services.
            '';
          };
        };
      };

      options.services.hermes-agent =
        common.sharedOptions {
          defaultPackage = hermes-agent;
          defaultPackageText = lib.literalExpression "hermes-agent.packages.\${system}.default";
          defaultWorkingDirectory = config.home.homeDirectory;
          defaultWorkingDirectoryText = lib.literalExpression "config.home.homeDirectory";
        }
        // {
          hermesHome = lib.mkOption {
            type = lib.types.str;
            default = "${config.home.homeDirectory}/.hermes";
            defaultText = lib.literalExpression ''"''${config.home.homeDirectory}/.hermes"'';
            description = ''
              The value of HERMES_HOME. This state directory holds
              config.yaml, .env, auth.json, the sessions, the skills, the
              memory and the cron jobs.

              The NixOS module takes a `stateDir` and adds `/.hermes` to it.
              This module sets HERMES_HOME directly. Thus an existing
              ~/.hermes continues to work, and you can give the directory any
              name.
            '';
            example = "/home/alice/.hermes-work";
          };

          # `installPackage` moved to `programs.hermes-agent.enable`. The
          # option is dead, but it must not be silent: it defaulted to true,
          # so a person who never named it still got the command line, and a
          # quiet removal gives them a machine with no `hermes` and no
          # message. mkOption with an assertion, and not
          # mkRemovedOptionModule, because the message must name the exact
          # replacement for the value they set.
          installPackage = lib.mkOption {
            type = lib.types.nullOr lib.types.bool;
            default = null;
            visible = false;
            description = ''
              Removed. Use `programs.hermes-agent.enable` instead.
            '';
          };

          profiles = lib.mkOption {
            type = lib.types.attrsOf profileType;
            default = { };
            description = ''
              Declaratively managed named Hermes profiles. Each attribute
              creates an independent HERMES_HOME under
              hermesHome/profiles/<name>. Use `hermes -p <name>` to select it.
            '';
          };

          gateway.enable = lib.mkEnableOption "the messaging gateway service (Telegram, Discord, Slack, ...)";
        };

      config = lib.mkMerge [

        # ── programs.hermes-agent — the installation ──────────────────────
        # Outside the `services.enable` guard on purpose. A person can want
        # the command line or the application on a machine that runs no
        # daemon at all.
        (lib.mkIf cfgPrograms.enable {
          home.packages = [ cfgPrograms.package ];
          home.sessionVariables.HERMES_HOME = cfg.hermesHome;
        })

        # A launcher from the desktop menu reads no shell profile, so the
        # HERMES_HOME that `programs.enable` exports does not reach it. Home
        # Manager writes only systemd.user.sessionVariables into
        # environment.d, and this module does not put HERMES_HOME there,
        # because that file applies to each user unit. Thus the launcher
        # carries the value itself. See desktopEnvironment above.
        (lib.mkIf cfgPrograms.desktop.enable {
          home.packages = [ desktopPackage ];
        })

        {
          assertions = [
            {
              # `installPackage` was removed in favour of the programs/services
              # split. It defaulted to true, so a quiet removal leaves a person
              # with no `hermes` on the PATH and no message.
              assertion = cfg.installPackage == null;
              message = common.installPackageRemovedMessage cfg.installPackage;
            }
          ];
        }

        (lib.mkIf cfg.enable (
          lib.mkMerge [

            # ── Merge MCP servers into settings ────────────────────────────
            (lib.mkIf (cfg.mcpServers != { }) {
              services.hermes-agent.settings.mcp_servers = common.mcpServersToConfig cfg.mcpServers;
            })

            {
              assertions =
                common.pluginNameAssertions {
                  inherit cfg;
                  optionPath = "services.hermes-agent";
                }
                ++ profilePluginAssertions
                ++ common.profileNameAssertions {
                  profiles = cfg.profiles;
                  optionPath = "services.hermes-agent.profiles";
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
                    # The interface poll reads `ip`, which iproute2 supplies on
                    # Linux only.
                    assertion = !isDarwin || cfg.backend.waitFor != "interface";
                    message = "services.hermes-agent.backend.waitFor = \"interface\" works on Linux only. Use \"hostname\" on Darwin.";
                  }
                ];
            }

            # The agent runs these tools, so they belong on the PATH of the
            # person as well as in the unit.
            (lib.mkIf cfgPrograms.enable {
              home.packages = cfg.extraPackages;
            })

            # ── Activation: directories, config, secrets, documents ────────
            {
              # The activation runs after writeBoundary, when the home.file
              # symlinks are in place. It also runs after linkGeneration, when
              # Home Manager completes the switch. A secret that the activation
              # entry of sops-nix writes exists at that point.
              home.activation.hermesAgentSetup =
                lib.hm.dag.entryAfter
                  [
                    "writeBoundary"
                    "linkGeneration"
                  ]
                  (
                    common.mkStateScript {
                      inherit pkgs cfg;
                      inherit (cfg) hermesHome workingDirectory;
                      run = "$DRY_RUN_CMD ";
                      stateDirs = common.stateSubdirs;
                      managedSystem = "home-manager";
                      # This state has one user. No group needs access to it.
                      modes = {
                        config = "0600";
                        env = "0600";
                        managed = "0600";
                        auth = "0600";
                        document = "0600";
                      };
                    }
                    + "\n"
                    + lib.concatStringsSep "\n" profileStateScripts
                  );
            }

            # ── Linux: systemd user services ───────────────────────────────
            (lib.mkIf (isLinux && cfg.gateway.enable) {
              systemd.user.services.hermes-agent = mkUnit {
                description = "Hermes Agent Gateway";
                argv = common.gatewayArgv cfg;
              };
            })

            (lib.mkIf isLinux {
              systemd.user.services = lib.mapAttrs' (
                name: profile:
                let
                  agentCfg = profileCfg profile;
                in
                lib.nameValuePair "hermes-agent-${name}" (mkUnit {
                  description = "Hermes Agent Gateway (${name} profile)";
                  argv = common.gatewayArgv agentCfg;
                  inherit agentCfg;
                  hermesHome = profileHome name;
                })
              ) (lib.filterAttrs (_name: profile: profile.gateway.enable) cfg.profiles);
            })

            (lib.mkIf (isLinux && cfg.backend.mode != "none") {
              systemd.user.services.hermes-backend = mkUnit {
                description = common.backendDescription cfg;
                argv = common.backendArgv { inherit pkgs cfg; };
              };
            })

            # ── Darwin: launchd agents ─────────────────────────────────────
            (lib.mkIf (isDarwin && cfg.gateway.enable) {
              launchd.agents.hermes-agent = mkAgent {
                argv = common.gatewayArgv cfg;
                logName = "hermes-agent";
              };
            })

            (lib.mkIf isDarwin {
              launchd.agents = lib.mapAttrs' (
                name: profile:
                let
                  agentCfg = profileCfg profile;
                  logName = "hermes-agent-${name}";
                in
                lib.nameValuePair logName (mkAgent {
                  argv = common.gatewayArgv agentCfg;
                  inherit agentCfg logName;
                  hermesHome = profileHome name;
                })
              ) (lib.filterAttrs (_name: profile: profile.gateway.enable) cfg.profiles);
            })

            (lib.mkIf (isDarwin && cfg.backend.mode != "none") {
              launchd.agents.hermes-backend = mkAgent {
                argv = common.backendArgv { inherit pkgs cfg; };
                logName = "hermes-backend";
              };
            })
          ]
        ))
      ];
    };
}
