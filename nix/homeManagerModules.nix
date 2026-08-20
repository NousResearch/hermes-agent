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
#   changed   addToSystemPackages      -> installPackage and
#                                        home.sessionVariables
#   changed   stateDir (+ "/.hermes")  -> hermesHome, set directly
#
# To use the module:
#   imports = [ hermes-agent.homeManagerModules.default ];
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
      common = import ./moduleCommon.nix { inherit lib; };

      effectivePackage = common.effectivePackage cfg;
      hermes-agent = inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default;

      inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

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

          installPackage = lib.mkOption {
            type = lib.types.bool;
            default = true;
            description = ''
              Add the hermes CLI to home.packages, and export HERMES_HOME
              with home.sessionVariables. Interactive shells then use the
              same state as the services.

              The equivalent NixOS option, `addToSystemPackages`, exports
              HERMES_HOME with environment.variables. That variable applies
              to the full system and replaces the HERMES_HOME of each other
              user. This module exports the variable for one user session
              only, which is the reason to use Home Manager.
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

      config = lib.mkIf cfg.enable (
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
              };
          }

          # ── Packages and interactive-shell environment ─────────────────
          (lib.mkIf cfg.installPackage {
            home.packages = [ effectivePackage ] ++ cfg.extraPackages;
            home.sessionVariables.HERMES_HOME = cfg.hermesHome;
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
              argv = common.backendArgv cfg;
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
              argv = common.backendArgv cfg;
              logName = "hermes-backend";
            };
          })
        ]
      );
    };
}
