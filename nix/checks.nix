# nix/checks.nix — Build-time verification tests
#
# Checks are Linux-only: the full Python venv (via uv2nix) includes
# transitive deps like onnxruntime that lack compatible wheels on
# aarch64-darwin. The package and devShell still work on macOS.
{ inputs, ... }: {
  perSystem = { pkgs, lib, self', ... }:
    let
      rabbit-agent = self'.packages.default;
      rabbitVenv = rabbit-agent.rabbitVenv;

      configMergeScript = pkgs.callPackage ./configMergeScript.nix { };

      # Same lock-derived interpreter the packages use (nix/pythonLock.nix
      # owns the pm/lock.json -> family -> nixpkgs selection).
      pythonLock = pkgs.callPackage ./pythonLock.nix { };

      # ── How the checks evaluate the modules ───────────────────────────
      # The checks evaluate both modules for real. The NixOS module goes
      # through lib.evalModules with the NixOS module list. The Home Manager
      # module goes through the homeManagerConfiguration function of
      # home-manager. The option system rejects a wrong type, an option that
      # does not exist, and a broken activation string. Each of these faults
      # then stops the check, and not the rebuild of a user.
      evalNixosModule =
        settings:
        inputs.nixpkgs.lib.evalModules {
          modules = import "${inputs.nixpkgs}/nixos/modules/module-list.nix" ++ [
            inputs.self.nixosModules.default
            { _module.args.lib = inputs.nixpkgs.lib; }
            { nixpkgs.hostPlatform = pkgs.stdenv.hostPlatform.system; }
            {
              system.stateVersion = "24.11";
              boot.loader.grub.enable = false;
              fileSystems."/" = {
                device = "/dev/null";
                fsType = "ext4";
              };
            }
            { services.rabbit-agent = settings; }
          ];
        };

      evalHomeModule =
        settings:
        inputs.home-manager.lib.homeManagerConfiguration {
          inherit pkgs;
          modules = [
            inputs.self.homeManagerModules.default
            {
              home = {
                username = "rabbit-check";
                homeDirectory = "/home/rabbit-check";
                stateVersion = "24.11";
              };
            }
            { services.rabbit-agent = settings; }
          ];
        };

      # The programs./services. split means a check often needs both halves.
      # This takes each one as its own attribute set.
      evalHomeSplit =
        {
          programs ? { },
          services ? { },
        }:
        inputs.home-manager.lib.homeManagerConfiguration {
          inherit pkgs;
          modules = [
            inputs.self.homeManagerModules.default
            {
              home = {
                username = "rabbit-check";
                homeDirectory = "/home/rabbit-check";
                stateVersion = "24.11";
              };
            }
            {
              programs.rabbit-agent = programs;
              services.rabbit-agent = services;
            }
          ];
        };

      # The option names that each module defines under
      # services.rabbit-agent. The internal names that the module system adds
      # are not in the list.
      moduleOptionNames =
        eval: lib.attrNames (lib.filterAttrs (n: _: !lib.hasPrefix "_" n) eval.options.services.rabbit-agent);

      # These options belong to one module by design. The check does not
      # compare the two lists against each other, because that test only
      # detects a change. The important property is that each shared option
      # is on both modules.
      nixosOnlyOptions = [
        "addToSystemPackages"
        "container"
        "createUser"
        "group"
        "stateDir"
        "user"
      ];
      homeOnlyOptions = [
        "gateway"
        "rabbitHome"
        "installPackage"
      ];

      # Auto-generated config key reference — always in sync with Python
      configKeys = pkgs.runCommand "rabbit-config-keys" {} ''
        set -euo pipefail
        export HOME=$TMPDIR
        ${rabbitVenv}/bin/python3 -c '
import json, sys
from rabbit_cli.config import DEFAULT_CONFIG

def leaf_paths(d, prefix=""):
    paths = []
    for k, v in sorted(d.items()):
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            paths.extend(leaf_paths(v, path))
        else:
            paths.append(path)
    return paths

json.dump(sorted(leaf_paths(DEFAULT_CONFIG)), sys.stdout, indent=2)
' > $out
      '';
    in {
      packages.configKeys = configKeys;

      checks = {
        # Cross-platform evaluation — catches "not supported for interpreter"
        # errors (e.g. sphinx dropping python311) without needing a darwin builder.
        # Evaluation is pure and instant; it doesn't build anything.
        cross-eval = let
          targetSystems = builtins.filter
            (s: inputs.self.packages ? ${s})
            [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
          tryEvalPkg = sys:
            let pkg = inputs.self.packages.${sys}.default;
            in builtins.tryEval (builtins.seq pkg.drvPath true);
          results = map (sys: { inherit sys; result = tryEvalPkg sys; }) targetSystems;
          failures = builtins.filter (r: !r.result.success) results;
          failMsg = lib.concatMapStringsSep "\n" (r: "  - ${r.sys}") failures;
        in pkgs.runCommand "rabbit-cross-eval" { } (
          if failures != [] then
            throw "Package fails to evaluate on:\n${failMsg}"
          else ''
            echo "PASS: package evaluates on all ${toString (builtins.length targetSystems)} platforms"
            mkdir -p $out
            echo "ok" > $out/result
          ''
        );

        # pm/lock.json pins provenance sidecars (checksums.txt, .sig/.pem/.asc)
        # next to these archives. `nix flake check` otherwise only evaluates
        # the pm derivations, so a sidecar leaking into srcs ("do not know
        # how to unpack") stayed green; build the two sidecar-bearing pins.
        pm-packages-unpack = pkgs.runCommand "rabbit-pm-packages-unpack" { } ''
          test -x ${self'.packages.pm-tirith}/tirith
          test -x ${self'.packages.pm-iron-proxy}/iron-proxy
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify the default package builds successfully (cross-platform).
        # On Linux the runtime checks below already depend on the package,
        # but this ensures darwin builders also build it during flake check.
        build-package = pkgs.runCommand "rabbit-build-package" { } ''
          echo "PASS: package built at ${rabbit-agent}"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Inspect the shipped assets: successful JS compilation alone does
        # not prove Vite copied the generated public files into the package.
        frontend-icons = pkgs.runCommand "rabbit-frontend-icons" {
          nativeBuildInputs = [ (pkgs.python3.withPackages (ps: [ ps.pillow ])) ];
        } ''
          python3 - <<'PY'
          from pathlib import Path
          from PIL import Image

          desktop = Path('${self'.packages.desktop}/share')
          dist = desktop / 'rabbit-desktop/dist'
          launcher = desktop / 'icons/hicolor/1024x1024/apps/rabbit.png'
          for path in [launcher, dist / 'apple-touch-icon.png',
                       dist / 'nous-girl.png', dist / 'nous-girl-dark.png',
                       Path('${self'.packages.web}/favicon.ico')]:
              with Image.open(path) as image:
                  image.load()
                  assert image.width > 0 and image.height > 0, path
          with Image.open(launcher) as image, Image.open(dist / 'apple-touch-icon.png') as window_icon:
              assert image.size == window_icon.size
              assert image.convert('RGBA').tobytes() == window_icon.convert('RGBA').tobytes()
          print('PASS: desktop and web ship decodable generated icons; launcher matches window icon')
          PY
          mkdir -p $out
        '';

        # Verify the devShell builds successfully (cross-platform).
        build-devshell = pkgs.runCommand "rabbit-build-devshell" { } ''
          echo "PASS: devShell built at ${self'.devShells.default}"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # ── The Home Manager module ──────────────────────────────────────
        # This check evaluates homeManagerModules.default through the real
        # module system of home-manager. It runs on each platform. The module
        # supports Linux, with systemd user units, and Darwin, with launchd
        # agents. Each host checks its own kind of process.
        home-manager-module =
          let
            enabled = evalHomeSplit {
              programs.enable = true;
              services = {
                enable = true;
                gateway.enable = true;
                backend.mode = "serve";
                settings.model.default = "test/model";
                environment.RABBIT_TEST = "1";
                environmentFiles = [ "/run/secrets/rabbit-env" ];
                rabbitHomeFiles."SOUL.md" = "test soul";
                # documents needs an explicit workingDirectory. The check
                # workspace-files-need-a-directory below asserts that rule.
                workingDirectory = "/home/test-user/workspace";
                documents."AGENTS.md" = "test agents";
                mcpServers.demo = {
                  command = "echo";
                  args = [ "hi" ];
                };
              };
            };
            cfg = enabled.config;

            # The gateway and the backend are two processes with one
            # RABBIT_HOME.
            processes =
              if pkgs.stdenv.hostPlatform.isDarwin then
                lib.mapAttrs (_: agent: {
                  argv = agent.config.ProgramArguments;
                  env = agent.config.EnvironmentVariables;
                }) (lib.filterAttrs (n: _: lib.hasPrefix "rabbit" n) cfg.launchd.agents)
              else
                lib.mapAttrs (_: unit: {
                  argv = [ unit.Service.ExecStart ];
                  env = unit.Service.Environment;
                }) (lib.filterAttrs (n: _: lib.hasPrefix "rabbit" n) cfg.systemd.user.services);

            names = lib.attrNames processes;
            argvOf = name: lib.concatStringsSep " " (lib.flatten (processes.${name}.argv));
            # The systemd Environment is a list of "K=V" strings. The launchd
            # equivalent is an attribute set. Make both into one "K=V K=V"
            # string, so that the assertions below are the same on each host.
            envOf =
              name:
              let
                env = processes.${name}.env;
              in
              lib.concatStringsSep " " (
                if lib.isAttrs env then lib.mapAttrsToList (k: v: "${k}=${toString v}") env else env
              );

            activation = cfg.home.activation.rabbitAgentSetup.data;

            failures =
              lib.optional (names != [
                "rabbit-agent"
                "rabbit-backend"
              ]) "expected rabbit-agent + rabbit-backend processes, got: ${toString names}"
              ++ lib.optional (
                !lib.hasInfix "bin/rabbit gateway" (argvOf "rabbit-agent")
              ) "gateway process does not run `rabbit gateway`: ${argvOf "rabbit-agent"}"
              ++ lib.optional (
                !lib.hasInfix "bin/rabbit serve" (argvOf "rabbit-backend")
              ) "backend process does not run `rabbit serve`: ${argvOf "rabbit-backend"}"
              ++ lib.optional (
                !lib.hasInfix "--no-open" (argvOf "rabbit-backend")
              ) "backend must pass --no-open so a service never opens a browser"
              ++ lib.optional (
                lib.any (n: !lib.hasInfix "/home/rabbit-check/.rabbit" (envOf n)) names
              ) "gateway and backend must share one RABBIT_HOME"
              ++ lib.optional (
                cfg.home.sessionVariables.RABBIT_HOME or null != "/home/rabbit-check/.rabbit"
              ) "programs.rabbit-agent.enable must export RABBIT_HOME for interactive shells"
              ++ lib.optional (
                !lib.hasInfix "rabbit-config-merge" activation
              ) "activation must deep-merge config.yaml, not overwrite it"
              ++ lib.optional (
                !lib.hasInfix "/home/rabbit-check/.rabbit/SOUL.md" activation
              ) "rabbitHomeFiles must install into RABBIT_HOME"
              ++ lib.optional (
                !lib.hasInfix "/home/test-user/workspace/AGENTS.md" activation
              ) "documents must install into workingDirectory"
              # The CLI reads RABBIT_MANAGED to name the rebuild command when
              # it refuses to write the configuration. A Home Manager install
              # has no nixos-rebuild command. Thus it must not report NixOS.
              ++ lib.optional (
                !lib.any (n: lib.hasInfix "RABBIT_MANAGED=home-manager" (envOf n)) names
              ) "processes must report RABBIT_MANAGED=home-manager"
              ++ lib.optional (
                !lib.hasInfix "rabbit-managed" activation
              ) "activation must write a .managed marker naming the managing system";
          in
          pkgs.runCommand "rabbit-home-manager-module" { } (
            if failures != [ ] then
              throw "Home Manager module check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: home-manager module evaluates (${toString (lib.length names)} processes)"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── Workspace files need a chosen directory ──────────────────────
        # `documents` goes into workingDirectory. The default of that option
        # is bad, and it is different on each module, so the modules refuse
        # the two options together.
        #
        # Home Manager reads the assertions while it builds `config`. Thus a
        # refused case throws an error and does not return a list. tryEval
        # makes the error into data again.
        workspace-files-need-a-directory =
          let
            accepts =
              settings:
              let
                cfg = (evalHomeModule ({ enable = true; } // settings)).config;
                probe = builtins.tryEval (lib.all (a: a.assertion) cfg.assertions);
              in
              probe.success && probe.value;

            rejects = settings: !(accepts settings);

            # This directory has the same text as the default. A comparison
            # of values reads it as untouched, but a comparison of priorities
            # sees the definition. This row is the reason that the code tests
            # the priority.
            sameAsDefault = "/home/rabbit-check";

            cases = [
              {
                name = "documents without a directory is refused";
                ok = rejects { documents."AGENTS.md" = "x"; };
              }
              {
                name = "documents with a directory is accepted";
                ok = accepts {
                  documents."AGENTS.md" = "x";
                  workingDirectory = "/srv/workspace";
                };
              }
              {
                name = "a directory equal to the default still counts as chosen";
                ok = accepts {
                  documents."AGENTS.md" = "x";
                  workingDirectory = sameAsDefault;
                };
              }
              {
                name = "mkDefault counts as chosen";
                ok = accepts {
                  documents."AGENTS.md" = "x";
                  workingDirectory = lib.mkDefault "/srv/workspace";
                };
              }
              {
                name = "rabbitHomeFiles needs no directory";
                ok = accepts { rabbitHomeFiles."SOUL.md" = "x"; };
              }
              {
                name = "no files at all is accepted";
                ok = accepts { };
              }
            ];

            failed = lib.filter (c: !c.ok) cases;
          in
          pkgs.runCommand "rabbit-workspace-files-need-a-directory" { } (
            if failed != [ ] then
              throw "workspace-files rule failed:\n${
                lib.concatMapStringsSep "\n" (c: "  - ${c.name}") failed
              }"
            else
              ''
                ${lib.concatMapStringsSep "\n" (c: ''echo "PASS: ${c.name}"'') cases}
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── The desktop application shares one RABBIT_HOME ───────────────
        # `programs.enable` exports RABBIT_HOME with home.sessionVariables,
        # which reaches an interactive shell only. Home Manager writes that
        # file to etc/profile.d, and a launcher from the desktop menu reads
        # no shell profile. Thus the desktop application would open ~/.rabbit
        # while the services use the RABBIT_HOME of the module, and the user
        # would see an empty application with no sessions and no keys.
        #
        # The launcher must therefore carry the value itself. This check
        # reads the real wrapper text of the package that the module
        # installs, and not an option value.
        home-manager-desktop =
          let
            tokenFile = "/run/secrets/rabbit-desktop-token";

            enabled = evalHomeSplit {
              programs = {
                enable = true;
                desktop.enable = true;
              };
              services = {
                enable = true;
                rabbitHome = "/home/rabbit-check/.rabbit-work";
                # An override on purpose. Without one the effective package
                # IS the default package, so a launcher that pinned the plain
                # default would look correct while it shipped a second
                # runtime to anyone who customises theirs.
                extraDependencyGroups = [ "exa" ];
                backend = {
                  mode = "serve";
                  port = 9231;
                  sessionTokenFile = tokenFile;
                };
              };
            };
            cfg = enabled.config;

            desktopPackages = builtins.filter (p: (p.pname or "") == "rabbit-desktop") cfg.home.packages;
            desktop = lib.head desktopPackages;
            wrapper = desktop.installPhase;

            # Read the value that each --set flag gives the launcher. The
            # quotes are not part of the test: escapeShellArg adds them only
            # when the value needs them, and a path with no special character
            # arrives bare.
            setValue =
              name:
              let
                m = builtins.match ".*--set ${name} ['\"]?([^'\"\n ]*)['\"]?.*" wrapper;
              in
              if m == null then null else lib.head m;

            # The agent package that the module installs, and the runtime
            # that the launcher pins. These must be the same store path: a
            # second Rabbit runtime beside the services is the fault that
            # `programs.enable` plus a plain desktop package would give.
            agentPackages = builtins.filter (p: (p.pname or "") == "rabbit-agent") cfg.home.packages;

            # The backend of the service, as the unit or the agent runs it.
            backendScript =
              let
                argv =
                  if pkgs.stdenv.hostPlatform.isDarwin then
                    cfg.launchd.agents.rabbit-backend.config.ProgramArguments
                  else
                    [ cfg.systemd.user.services.rabbit-backend.Service.ExecStart ];
                first = lib.head (lib.flatten argv);
                # writeShellScript gives a store path. Read the real text, so
                # the check tests the script and not the option that made it.
                path = lib.head (lib.splitString " " first);
              in
              builtins.readFile path;

            failures =
              lib.optional (
                lib.length desktopPackages != 1
              ) "programs.desktop.enable must install exactly one rabbit-desktop package, got ${toString (lib.length desktopPackages)}"
              ++ lib.optional (
                setValue "RABBIT_HOME" != "/home/rabbit-check/.rabbit-work"
              ) "the launcher must carry RABBIT_HOME: a GUI launcher reads no shell profile, so home.sessionVariables never reaches it (got: ${toString (setValue "RABBIT_HOME")})"
              ++ lib.optional (
                setValue "RABBIT_MANAGED" != "home-manager"
              ) "the launcher must report RABBIT_MANAGED=home-manager while the services own the configuration (got: ${toString (setValue "RABBIT_MANAGED")})"
              ++ lib.optional (
                lib.length agentPackages == 1
                && setValue "RABBIT_DESKTOP_RABBIT" != "${lib.head agentPackages}/bin/rabbit"
              ) "the launcher must pin the agent package that programs.enable installs, and not a second runtime: ${toString (setValue "RABBIT_DESKTOP_RABBIT")}"

              # ── The application reaches the backend of the service ──────
              ++ lib.optional (
                setValue "RABBIT_DESKTOP_REMOTE_URL" != "http://127.0.0.1:9231"
              ) "the launcher must name the backend of the service, or the application starts a second one (got: ${toString (setValue "RABBIT_DESKTOP_REMOTE_URL")})"
              ++ lib.optional (
                !lib.hasInfix "RABBIT_DESKTOP_REMOTE_TOKEN" wrapper
              ) "the launcher must give a token with the URL: the desktop resolver throws when the URL is set alone"
              ++ lib.optional (
                !lib.hasInfix "RABBIT_DASHBOARD_SESSION_TOKEN" backendScript
              ) "the backend must read the session token, or it makes a new one that the application cannot know"

              # ── The token never enters the Nix store ────────────────────
              # Each side must read the file at start time. A --set flag or
              # an Environment= value writes the literal into a store path
              # that all users can read.
              ++ lib.optional (
                !lib.hasInfix tokenFile wrapper || !lib.hasInfix "--run" wrapper
              ) "the launcher must read the token from ${tokenFile} at start time, with --run"
              ++ lib.optional (
                !lib.hasInfix tokenFile backendScript
              ) "the backend must read the token from ${tokenFile} at start time"
              ++ lib.optional (
                setValue "RABBIT_DESKTOP_REMOTE_TOKEN" != null
              ) "the token must never be a --set value: makeWrapper writes it into the world-readable Nix store";
          in
          pkgs.runCommand "rabbit-home-manager-desktop" { } (
            if failures != [ ] then
              throw "Home Manager desktop check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: the desktop launcher shares RABBIT_HOME, the runtime and the backend of the service"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── The desktop application without the services ─────────────────
        # A person can want the application on a machine that runs no daemon.
        # Then nothing writes config.yaml or the .managed marker, so the
        # launcher must not claim a managed install: the CLI would refuse an
        # edit that nothing else owns. It must also not name a backend, since
        # there is none.
        home-manager-desktop-standalone =
          let
            enabled = evalHomeSplit {
              programs = {
                enable = true;
                desktop.enable = true;
              };
            };
            cfg = enabled.config;

            desktopPackages = builtins.filter (p: (p.pname or "") == "rabbit-desktop") cfg.home.packages;
            wrapper = (lib.head desktopPackages).installPhase;

            failures =
              lib.optional (
                lib.length desktopPackages != 1
              ) "programs.desktop.enable must install the application with no services enabled"
              ++ lib.optional (
                !lib.hasInfix "--set RABBIT_HOME" wrapper
              ) "the launcher must carry RABBIT_HOME even with no services"
              ++ lib.optional (
                lib.hasInfix "RABBIT_MANAGED" wrapper
              ) "the launcher must not claim a managed install when no activation writes one"
              ++ lib.optional (
                lib.hasInfix "RABBIT_DESKTOP_REMOTE_URL" wrapper
              ) "the launcher must not name a backend when the services run none"
              ++ lib.optional (
                cfg.systemd.user.services ? rabbit-backend || cfg.launchd.agents ? rabbit-backend
              ) "programs.enable alone must start no service";
          in
          pkgs.runCommand "rabbit-home-manager-desktop-standalone" { } (
            if failures != [ ] then
              throw "Home Manager standalone desktop check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: the application runs with no services, and claims nothing that no activation wrote"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── installPackage names its replacement ─────────────────────────
        # The option was removed by the programs./services. split. It
        # defaulted to true, so a person who never named it still got the
        # command line. A silent removal thus leaves them with no `rabbit`
        # and no message. The module must refuse the configuration and name
        # the replacement.
        home-manager-install-package-removed =
          let
            common = import ./moduleCommon.nix { inherit lib; };

            # `builtins.length` is enough to force the assertion, because
            # Home Manager wraps the whole `config` in its assertion check.
            # `lib.deepSeq` would walk each package of the closure instead,
            # and overflow the stack before it reached an answer.
            refuses =
              value:
              !(builtins.tryEval (
                builtins.length
                  (evalHomeSplit {
                    services = {
                      enable = true;
                      installPackage = value;
                    };
                  }).config.home.packages
              )).success;

            # The check calls the same function the module calls, so it reads
            # the real message. Matching the source text of the module instead
            # would pass while the message was wrong.
            messageFor = common.installPackageRemovedMessage;

            cases = [
              {
                value = true;
                expect = "programs.rabbit-agent.enable = true;";
              }
              {
                value = false;
                expect = "programs.rabbit-agent.enable = false;";
              }
            ];

            failures =
              lib.concatMap (
                case:
                lib.optional (
                  !refuses case.value
                ) "installPackage = ${lib.boolToString case.value} must be refused"
                ++ lib.optional (
                  !lib.hasInfix case.expect (messageFor case.value)
                ) "the message for installPackage = ${lib.boolToString case.value} must name `${case.expect}`"
                ++ lib.optional (
                  !lib.hasInfix "installPackage was removed" (messageFor case.value)
                ) "the message must say that the option was removed"
              ) cases
              ++ lib.optional (
                # A configuration that never names the option must still work.
                # An assertion that fires on the default value would break each
                # existing user at once.
                refuses null
              ) "a configuration that never names installPackage must evaluate";
          in
          pkgs.runCommand "rabbit-home-manager-install-package-removed" { } (
            if failures != [ ] then
              throw "installPackage removal check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: installPackage is refused with guidance, and its absence evaluates"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── The two modules keep the same options ────────────────────────
        # The modules share one option set, in nix/moduleCommon.nix. Thus a
        # NixOS example works on Home Manager without a change. This check
        # asserts that relation and not the current list of names. An option
        # that goes into the shared set must appear on both modules. An
        # option for one module must be in that module's exclusion list.
        module-option-parity =
          let
            nixosNames = moduleOptionNames (evalNixosModule { });
            homeNames = moduleOptionNames (evalHomeModule { });

            sharedFromNixos = lib.subtractLists nixosOnlyOptions nixosNames;
            sharedFromHome = lib.subtractLists homeOnlyOptions homeNames;

            missingInHome = lib.subtractLists homeNames sharedFromNixos;
            missingInNixos = lib.subtractLists nixosNames sharedFromHome;

            # These two values check the exclusion lists. An entry for an
            # option that does not exist makes the check weaker, and gives no
            # message.
            staleNixosOnly = lib.subtractLists nixosNames nixosOnlyOptions;
            staleHomeOnly = lib.subtractLists homeNames homeOnlyOptions;

            failures =
              lib.optional (
                missingInHome != [ ]
              ) "shared options missing from the Home Manager module: ${toString missingInHome} (add to nix/moduleCommon.nix, or list under nixosOnlyOptions if system-scoped)"
              ++ lib.optional (
                missingInNixos != [ ]
              ) "shared options missing from the NixOS module: ${toString missingInNixos} (add to nix/moduleCommon.nix, or list under homeOnlyOptions if user-scoped)"
              ++ lib.optional (
                staleNixosOnly != [ ]
              ) "nixosOnlyOptions names options the NixOS module no longer defines: ${toString staleNixosOnly}"
              ++ lib.optional (
                staleHomeOnly != [ ]
              ) "homeOnlyOptions names options the Home Manager module no longer defines: ${toString staleHomeOnly}";
          in
          pkgs.runCommand "rabbit-module-option-parity" { } (
            if failures != [ ] then
              throw "Module option parity failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: ${toString (lib.length sharedFromNixos)} shared options present on both modules"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );
      } // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
        # ── The NixOS module ─────────────────────────────────────────────
        # This check runs on Linux only. The evaluation of a NixOS module
        # needs a Linux hostPlatform.
        nixos-module =
          let
            cfg = (evalNixosModule {
              enable = true;
              backend.mode = "dashboard";
              settings.model.default = "test/model";
              environmentFiles = [ "/run/secrets/rabbit-env" ];
              rabbitHomeFiles."SOUL.md" = "test soul";
            }).config;

            units = lib.filterAttrs (n: _: lib.hasPrefix "rabbit" n) cfg.systemd.services;
            names = lib.attrNames units;
            execOf = name: units.${name}.serviceConfig.ExecStart;
            activation = cfg.system.activationScripts."rabbit-agent-setup".text;

            failures =
              lib.optional (names != [
                "rabbit-agent"
                "rabbit-backend"
              ]) "expected rabbit-agent + rabbit-backend units, got: ${toString names}"
              ++ lib.optional (
                !lib.hasInfix "bin/rabbit gateway" (execOf "rabbit-agent")
              ) "gateway unit does not run `rabbit gateway`: ${execOf "rabbit-agent"}"
              ++ lib.optional (
                !lib.hasInfix "bin/rabbit dashboard" (execOf "rabbit-backend")
              ) "backend unit does not run `rabbit dashboard`: ${execOf "rabbit-backend"}"
              ++ lib.optional (
                units.rabbit-agent.environment.RABBIT_HOME != units.rabbit-backend.environment.RABBIT_HOME
              ) "gateway and backend must share one RABBIT_HOME"
              ++ lib.optional (
                !lib.hasInfix "/var/lib/rabbit/.rabbit/SOUL.md" activation
              ) "rabbitHomeFiles must install into RABBIT_HOME";

            # You cannot use container mode and the backend together. The
            # module says so with an assertion. Without the assertion it
            # makes a unit that never starts.
            containerConflict = builtins.tryEval (
              lib.deepSeq
                (evalNixosModule {
                  enable = true;
                  container.enable = true;
                  backend.mode = "serve";
                }).config.system.build.toplevel.drvPath
                true
            );
          in
          pkgs.runCommand "rabbit-nixos-module" { } (
            if failures != [ ] then
              throw "NixOS module check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else if containerConflict.success then
              throw "NixOS module check failed:\n  - an assertion must reject backend.mode with container.enable"
            else
              ''
                echo "PASS: nixos module evaluates (${toString (lib.length names)} units)"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── How the backend waits for its bind target ────────────────────
        # The backend binds to `host` immediately by default. A unit that
        # starts at boot can lose the race against the daemon that supplies
        # the address, such as tailscaled. `backend.waitFor` puts a poll in
        # front of the bind. This check proves three properties: the default
        # keeps the direct command line, each wait mode makes a launcher that
        # polls and then execs rabbit, and the assertions reject a
        # configuration that cannot work.
        backend-bind-wait =
          let
            execOf =
              settings:
              (evalNixosModule ({ enable = true; } // settings)).config.systemd.services.rabbit-backend.serviceConfig.ExecStart;

            direct = execOf { backend.mode = "serve"; };

            hostnameWait = execOf {
              backend = {
                mode = "serve";
                host = "host.example.ts.net";
                waitFor = "hostname";
              };
            };

            interfaceWait = execOf {
              backend = {
                mode = "dashboard";
                waitFor = "interface";
                interfaceName = "tailscale0";
                waitTimeout = 30;
              };
            };

            # The launcher is a store path. Read it to see what it runs.
            hostnameScript = builtins.readFile hostnameWait;
            interfaceScript = builtins.readFile interfaceWait;

            evalFails =
              settings:
              !(builtins.tryEval (
                lib.deepSeq
                  (evalNixosModule ({ enable = true; } // settings)).config.system.build.toplevel.drvPath
                  true
              )).success;

            failures =
              # The default must not change.
              lib.optional (!lib.hasInfix "bin/rabbit serve --host 127.0.0.1" direct)
                "without waitFor the backend must exec rabbit directly, got: ${direct}"
              ++ lib.optional (lib.hasInfix "rabbit-backend-launch" direct)
                "without waitFor the backend must not use the launcher"

              # The hostname mode polls the resolver, then binds the name.
              ++ lib.optional (!lib.hasInfix "rabbit-backend-launch" hostnameWait)
                "waitFor = hostname must run the launcher, got: ${hostnameWait}"
              ++ lib.optional (!lib.hasInfix "getent hosts" hostnameScript)
                "the hostname launcher must poll with getent"
              ++ lib.optional (!lib.hasInfix "host.example.ts.net" hostnameScript)
                "the hostname launcher must poll for backend.host"
              ++ lib.optional (!lib.hasInfix "exec " hostnameScript)
                "the launcher must exec rabbit, so that it keeps the MainPID"
              ++ lib.optional (!lib.hasInfix ''--host "$_target"'' hostnameScript)
                "the launcher must bind the address that the poll resolved"

              # The interface mode reads an address off the interface.
              ++ lib.optional (!lib.hasInfix "tailscale0" interfaceScript)
                "the interface launcher must poll backend.interfaceName"
              ++ lib.optional (!lib.hasInfix "_timeout=30" interfaceScript)
                "the launcher must use backend.waitTimeout"
              ++ lib.optional (!lib.hasInfix "bin/rabbit dashboard" interfaceScript)
                "the launcher must keep backend.mode"

              # The assertions reject what cannot work.
              ++
                lib.optional
                  (!evalFails {
                    backend = {
                      mode = "serve";
                      waitFor = "interface";
                    };
                  })
                  "an assertion must reject waitFor = interface without interfaceName"
              ++
                lib.optional
                  (!evalFails {
                    backend = {
                      mode = "serve";
                      interfaceName = "tailscale0";
                    };
                  })
                  "an assertion must reject interfaceName without waitFor = interface";
          in
          pkgs.runCommand "rabbit-backend-bind-wait" { } (
            if failures != [ ] then
              throw "backend bind wait check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: backend bind wait (default, hostname, interface)"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── The user manager that restart-safe cron workers need ─────────
        # The cron scheduler launches every job in a transient `systemd-run
        # --user --scope`, so that a gateway restart cannot kill a running job,
        # and it fails the fire closed when the scope cannot be created. A
        # system service has no user manager — and so no /run/user/<uid>/bus —
        # unless the uid lingers. That makes `linger` load-bearing for cron on
        # this module, not cosmetic: without it every job dies before an agent
        # starts. This check proves three properties: the uid the gateway runs
        # as lingers, a lingering gateway does not start before that uid's bus
        # exists, and a gateway whose uid does not linger waits for nothing.
        cron-worker-user-scope =
          let
            configOf = settings: (evalNixosModule ({ enable = true; } // settings)).config;

            managed = configOf { };
            gateway = managed.systemd.services.rabbit-agent;
            gatewayUser = gateway.serviceConfig.User;

            # The operator declares the user themselves. Nothing here knows
            # whether they lingered it, so nothing may assume a bus.
            unmanaged = (configOf { createUser = false; }).systemd.services.rabbit-agent;

            failures =
              lib.optional (!((managed.users.users.${gatewayUser}.linger or false) == true))
                "the uid the gateway runs as (${gatewayUser}) must linger: `systemd-run --user --scope` has no user manager to ask without it, and cron dispatch fails closed"
              ++ lib.optional (!lib.elem "linger-users.service" gateway.after)
                "a lingering gateway must be ordered after linger-users.service, the unit that runs `loginctl enable-linger`"
              ++ lib.optional (!lib.hasInfix "/run/user" gateway.preStart)
                "a lingering gateway must wait for its user bus before ExecStart: the bus environment is resolved once at startup, so a bus that appears later is one this process never sees"
              ++ lib.optional (lib.hasInfix "/run/user" unmanaged.preStart)
                "a gateway whose uid is not known to linger must not block on a user bus that may never arrive";
          in
          pkgs.runCommand "rabbit-cron-worker-user-scope" { } (
            if failures != [ ] then
              throw "cron worker user scope check failed:\n${lib.concatMapStringsSep "\n" (f: "  - ${f}") failures}"
            else
              ''
                echo "PASS: the gateway uid lingers and the unit waits for its user bus"
                mkdir -p $out
                echo "ok" > $out/result
              ''
          );

        # ── How .env is built ────────────────────────────────────────────
        # This check runs the real script that both modules use to build
        # $RABBIT_HOME/.env. The important property is that a second run
        # gives the same result. Activation runs at each rebuild. If the
        # script added the secrets to the file that exists, the file would
        # grow at each rebuild. The script writes the file again from the
        # base in the Nix store, which prevents that fault. This check proves
        # it.
        env-file-assembly =
          let
            envScript = (import ./moduleCommon.nix { inherit lib; }).mkEnvScript {
              inherit pkgs;
              environment = {
                RABBIT_PUBLIC = "visible";
              };
            };
          in
          pkgs.runCommand "rabbit-env-file-assembly" { } ''
            set -e
            workdir=$(mktemp -d)
            printf 'SECRET_TOKEN=s3cret\n' > "$workdir/secret-a"
            printf 'OTHER_TOKEN=t0ken\n' > "$workdir/secret-b"

            echo "=== First activation ==="
            ${envScript} "$workdir/.env" 0600 "$workdir/secret-a" "$workdir/secret-b"
            first=$(cat "$workdir/.env")

            grep -qx 'RABBIT_PUBLIC=visible' "$workdir/.env" || \
              (echo "FAIL: non-secret environment missing"; cat "$workdir/.env"; exit 1)
            grep -qx 'SECRET_TOKEN=s3cret' "$workdir/.env" || \
              (echo "FAIL: secret from environmentFile missing"; cat "$workdir/.env"; exit 1)
            grep -qx 'OTHER_TOKEN=t0ken' "$workdir/.env" || \
              (echo "FAIL: second environmentFile missing"; cat "$workdir/.env"; exit 1)
            echo "PASS: .env contains the declared environment and every secret"

            test "$(stat -c %a "$workdir/.env")" = "600" || \
              (echo "FAIL: .env mode is $(stat -c %a "$workdir/.env"), want 600"; exit 1)
            echo "PASS: .env installed with the requested mode"

            echo "=== Re-activation is idempotent ==="
            ${envScript} "$workdir/.env" 0600 "$workdir/secret-a" "$workdir/secret-b"
            second=$(cat "$workdir/.env")
            test "$first" = "$second" || \
              (echo "FAIL: second run changed .env"; diff <(echo "$first") <(echo "$second") || true; exit 1)

            COUNT=$(grep -c '^SECRET_TOKEN=' "$workdir/.env")
            test "$COUNT" -eq 1 || \
              (echo "FAIL: secret appears $COUNT times after two activations"; exit 1)
            echo "PASS: secrets are not accumulated across activations"

            echo "=== A removed environmentFile disappears ==="
            ${envScript} "$workdir/.env" 0600 "$workdir/secret-a"
            if grep -q '^OTHER_TOKEN=' "$workdir/.env"; then
              echo "FAIL: dropped environmentFile still present in .env"; exit 1
            fi
            echo "PASS: .env tracks the declared environmentFiles"

            mkdir -p $out
            echo "ok" > $out/result
          '';

        # ── The command lines of the services ────────────────────────────
        # The modules build these command lines. This check runs each one
        # through the real parser of the CLI. A subcommand or a flag with a
        # new name then fails here, and not as a service that restarts again
        # and again after a rebuild.
        #
        # The method: add one sentinel flag that the CLI does not know, and
        # parse without --help. argparse refuses unknown arguments before it
        # calls the command, so no process starts and no port is bound. The
        # error names each argument that argparse did not accept. If the
        # error names only the sentinel, the parser accepts each other flag.
        #
        # `--help` cannot do this job. It returns before argparse reads the
        # remainder of the command line.
        service-argv =
          let
            common = import ./moduleCommon.nix { inherit lib; };
            cfgFor = mode: {
              package = rabbit-agent;
              extraPythonPackages = [ ];
              extraDependencyGroups = [ ];
              extraArgs = [ ];
              backend = {
                inherit mode;
                host = "127.0.0.1";
                port = 9119;
                extraArgs = [ ];
                waitFor = null;
                interfaceName = null;
                waitTimeout = 120;
                # No token here: this case asserts the plain argv, which the
                # module builds only when nothing must run before the
                # backend. A token needs the launcher script instead.
                sessionTokenFile = null;
              };
            };
            sentinel = "--rabbit-nix-argv-probe";
            probe = argv: lib.escapeShellArgs (argv ++ [ sentinel ]);
          in
          pkgs.runCommand "rabbit-service-argv" { } ''
            set -e
            export HOME=$(mktemp -d)

            check() {
              local label="$1"
              shift
              local output
              output=$("$@" 2>&1) && {
                echo "FAIL: $label — the sentinel flag was accepted, so this probe proves nothing"
                exit 1
              }
              case "$output" in
                *"unrecognized arguments: ${sentinel}")
                  echo "PASS: $label — every flag but the sentinel is recognized" ;;
                *"unrecognized arguments"*)
                  echo "FAIL: $label — the CLI also rejected flags the module passes:"
                  echo "$output" | tail -3
                  exit 1 ;;
                *)
                  echo "FAIL: $label — argv rejected before flag parsing (bad subcommand?):"
                  echo "$output" | tail -3
                  exit 1 ;;
              esac
            }

            check "gateway"   ${probe (common.gatewayArgv (cfgFor "none"))}
            check "serve"     ${probe (common.backendArgv { inherit pkgs; cfg = cfgFor "serve"; })}
            check "dashboard" ${probe (common.backendArgv { inherit pkgs; cfg = cfgFor "dashboard"; })}

            mkdir -p $out
            echo "ok" > $out/result
          '';

        # Verify binaries exist and are executable
        package-contents = pkgs.runCommand "rabbit-package-contents" { } ''
          set -e
          echo "=== Checking binaries ==="
          test -x ${rabbit-agent}/bin/rabbit || (echo "FAIL: rabbit binary missing"; exit 1)
          test -x ${rabbit-agent}/bin/rabbit-agent || (echo "FAIL: rabbit-agent binary missing"; exit 1)
          echo "PASS: All binaries present"

          echo "=== Checking version ==="
          ${rabbit-agent}/bin/rabbit --version 2>&1 | grep -qi "rabbit" || (echo "FAIL: version check"; exit 1)
          echo "PASS: Version check"

          echo "=== All checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Exercise every declared command and the environment delivered by
        # makeWrapper, plus the shared assembler's store-reference contract.
        entry-points-sync = pkgs.runCommand "rabbit-entry-points-sync" { } ''
          ${rabbit-agent.python}/bin/python3 ${./tests/agent-references.py} \
            ${rabbit-agent} ${../pyproject.toml} ${rabbit-agent.agentInputsFile}
          mkdir -p $out
        '';

        # A pre-existing CLI install must not override the package's backend.
        desktop-backend = pkgs.runCommand "rabbit-desktop-backend" {
          nativeBuildInputs = [ rabbit-agent.python pkgs.cage ];
        } ''
          python3 ${./tests/desktop-backend.py} \
            ${self'.packages.desktop}/bin/rabbit-desktop ${rabbit-agent}/bin/rabbit
          mkdir -p $out
        '';

        # Verify CLI subcommands are accessible
        cli-commands = pkgs.runCommand "rabbit-cli-commands" { } ''
          set -e
          export HOME=$(mktemp -d)

          echo "=== Checking rabbit --help ==="
          ${rabbit-agent}/bin/rabbit --help 2>&1 | grep -q "gateway" || (echo "FAIL: gateway subcommand missing"; exit 1)
          ${rabbit-agent}/bin/rabbit --help 2>&1 | grep -q "config" || (echo "FAIL: config subcommand missing"; exit 1)
          echo "PASS: All subcommands accessible"

          echo "=== All CLI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled skills are present in the package
        bundled-skills = pkgs.runCommand "rabbit-bundled-skills" { } ''
          set -e
          echo "=== Checking bundled skills ==="
          test -d ${rabbit-agent}/share/rabbit-agent/skills || (echo "FAIL: skills directory missing"; exit 1)
          echo "PASS: skills directory exists"

          # -L: skills/ is a symlink to the filtered source store path
          SKILL_COUNT=$(find -L ${rabbit-agent}/share/rabbit-agent/skills -name "SKILL.md" | wc -l)
          test "$SKILL_COUNT" -gt 0 || (echo "FAIL: no SKILL.md files found in skills directory"; exit 1)
          echo "PASS: $SKILL_COUNT bundled skills found"

          grep -q "RABBIT_BUNDLED_SKILLS" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_BUNDLED_SKILLS not in wrapper"; exit 1)
          echo "PASS: RABBIT_BUNDLED_SKILLS set in wrapper"

          # Optional skills ship via the wrapper too (pythonSrc excludes
          # them from the wheel, so the env var is the only path in nix).
          test -d ${rabbit-agent}/share/rabbit-agent/optional-skills || \
            (echo "FAIL: optional-skills directory missing"; exit 1)
          OPT_COUNT=$(find -L ${rabbit-agent}/share/rabbit-agent/optional-skills -name "SKILL.md" | wc -l)
          test "$OPT_COUNT" -gt 0 || (echo "FAIL: no SKILL.md files in optional-skills"; exit 1)
          grep -q "RABBIT_OPTIONAL_SKILLS" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_OPTIONAL_SKILLS not in wrapper"; exit 1)
          echo "PASS: $OPT_COUNT optional skills found, RABBIT_OPTIONAL_SKILLS set in wrapper"

          echo "=== All bundled skills checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled plugins (platforms, memory, context_engine) are present
        bundled-plugins = pkgs.runCommand "rabbit-bundled-plugins" { } ''
          set -e
          echo "=== Checking bundled plugins ==="
          test -d ${rabbit-agent}/share/rabbit-agent/plugins || (echo "FAIL: plugins directory missing"; exit 1)
          echo "PASS: plugins directory exists"

          test -f ${rabbit-agent}/share/rabbit-agent/plugins/platforms/irc/plugin.yaml || \
            (echo "FAIL: irc plugin manifest missing"; exit 1)
          echo "PASS: irc plugin manifest present"

          grep -q "RABBIT_BUNDLED_PLUGINS" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_BUNDLED_PLUGINS not in wrapper"; exit 1)
          echo "PASS: RABBIT_BUNDLED_PLUGINS set in wrapper"

          echo "=== All bundled plugins checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled i18n locale catalogs are present and resolvable.
        # Regression for #23943 / #27632 / #35374 — sealed Nix venvs dropped
        # locales/, surfacing raw i18n keys like gateway.reset.header_default.
        bundled-locales = pkgs.runCommand "rabbit-bundled-locales" { } ''
          set -e
          echo "=== Checking bundled locales ==="
          test -d ${rabbit-agent}/share/rabbit-agent/locales || (echo "FAIL: locales directory missing"; exit 1)
          echo "PASS: locales directory exists"

          # -L: locales/ is a symlink to the source store path
          LOC_COUNT=$(find -L ${rabbit-agent}/share/rabbit-agent/locales -name "*.yaml" | wc -l)
          test "$LOC_COUNT" -ge 16 || (echo "FAIL: expected >=16 catalogs, found $LOC_COUNT"; exit 1)
          echo "PASS: $LOC_COUNT locale catalogs found"

          test -f ${rabbit-agent}/share/rabbit-agent/locales/en.yaml || (echo "FAIL: en.yaml missing"; exit 1)
          echo "PASS: en.yaml present"

          grep -q "RABBIT_BUNDLED_LOCALES" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_BUNDLED_LOCALES not in wrapper"; exit 1)
          echo "PASS: RABBIT_BUNDLED_LOCALES set in wrapper"

          # locales/ is a bare data dir (no __init__.py), shipped via a
          # symlink + RABBIT_BUNDLED_LOCALES (not via wheel data-files).
          # Verify the wrapper override resolves real strings.
          export HOME=$(mktemp -d)
          RENDERED=$(cd "$HOME" && RABBIT_BUNDLED_LOCALES=${rabbit-agent}/share/rabbit-agent/locales \
            ${rabbitVenv}/bin/python3 -c "from agent import i18n; print(i18n.t('gateway.reset.header_default', lang='en'))")
          echo "rendered: $RENDERED"
          test "$RENDERED" != "gateway.reset.header_default" || (echo "FAIL: i18n returned the raw key with RABBIT_BUNDLED_LOCALES set"; exit 1)
          echo "PASS: i18n renders a human string via the wrapper override"

          echo "=== All bundled locales checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled optional-mcps catalog is present and resolvable.
        # optional-mcps/ is a bare data dir shipped via symlink +
        # RABBIT_OPTIONAL_MCPS (not via wheel data-files).
        bundled-mcps = pkgs.runCommand "rabbit-bundled-mcps" { } ''
          set -e
          echo "=== Checking bundled optional-mcps ==="
          test -d ${rabbit-agent}/share/rabbit-agent/optional-mcps || (echo "FAIL: optional-mcps directory missing"; exit 1)
          echo "PASS: optional-mcps directory exists"

          MANIFEST_COUNT=$(find -L ${rabbit-agent}/share/rabbit-agent/optional-mcps -name "manifest.yaml" | wc -l)
          test "$MANIFEST_COUNT" -gt 0 || (echo "FAIL: no manifest.yaml files found"; exit 1)
          echo "PASS: $MANIFEST_COUNT catalog manifests found"

          grep -q "RABBIT_OPTIONAL_MCPS" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_OPTIONAL_MCPS not in wrapper"; exit 1)
          echo "PASS: RABBIT_OPTIONAL_MCPS set in wrapper"

          export HOME=$(mktemp -d)
          CATALOG=$(cd "$HOME" && ${rabbit-agent}/bin/rabbit mcp catalog 2>/dev/null || true)
          echo "catalog output: $CATALOG"
          test -n "$CATALOG" || (echo "FAIL: rabbit mcp catalog returned empty"; exit 1)
          echo "PASS: mcp catalog resolves entries"

          echo "=== All bundled optional-mcps checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled TUI is present and compiled
        bundled-tui = pkgs.runCommand "rabbit-bundled-tui" { } ''
          set -e
          echo "=== Checking bundled TUI ==="
          test -d ${rabbit-agent}/ui-tui || (echo "FAIL: ui-tui directory missing"; exit 1)
          echo "PASS: ui-tui directory exists"

          test -f ${rabbit-agent}/ui-tui/dist/entry.js || (echo "FAIL: compiled entry.js missing"; exit 1)
          echo "PASS: compiled entry.js present"

          # self-contained bundle; no runtime node_modules expected

          grep -q "RABBIT_TUI_DIR" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_TUI_DIR not in wrapper"; exit 1)
          echo "PASS: RABBIT_TUI_DIR set in wrapper"

          echo "=== All bundled TUI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify RABBIT_NODE is set in wrapper and points to Node 26+
        # (Rabbit pins its toolchain to Node 26 everywhere)
        rabbit-node = pkgs.runCommand "rabbit-node-version" { } ''
          set -e
          echo "=== Checking RABBIT_NODE in wrapper ==="
          grep -q "RABBIT_NODE" ${rabbit-agent}/bin/rabbit || \
            (echo "FAIL: RABBIT_NODE not set in wrapper"; exit 1)
          echo "PASS: RABBIT_NODE present in wrapper"

          RABBIT_NODE=$(sed -n "s/^export RABBIT_NODE='\(.*\)'/\1/p" ${rabbit-agent}/bin/rabbit)
          test -x "$RABBIT_NODE" || (echo "FAIL: RABBIT_NODE=$RABBIT_NODE not executable"; exit 1)
          echo "PASS: RABBIT_NODE executable at $RABBIT_NODE"

          NODE_MAJOR=$("$RABBIT_NODE" --version | sed 's/^v//' | cut -d. -f1)
          test "$NODE_MAJOR" -ge 26 || \
            (echo "FAIL: Node v$NODE_MAJOR < 26, Rabbit requires Node 26"; exit 1)
          echo "PASS: Node v$NODE_MAJOR >= 26"

          echo "=== All RABBIT_NODE checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify RABBIT_MANAGED guard works on all mutation commands
        managed-guard = pkgs.runCommand "rabbit-managed-guard" { } ''
          set -e
          export HOME=$(mktemp -d)

          check_blocked() {
            local label="$1"
            shift
            OUTPUT=$(RABBIT_MANAGED=true "$@" 2>&1 || true)
            # Case-insensitive: the message names the managing system as the
            # identifier it is keyed by, and the display form is not the
            # property under test here.
            echo "$OUTPUT" | grep -qi "managed by nixos" || (echo "FAIL: $label not guarded"; echo "$OUTPUT"; exit 1)
            echo "PASS: $label blocked in managed mode"
          }

          echo "=== Checking RABBIT_MANAGED guards ==="
          check_blocked "config set" ${rabbit-agent}/bin/rabbit config set model foo
          check_blocked "config edit" ${rabbit-agent}/bin/rabbit config edit

          echo "=== All guard checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraPythonPackages PYTHONPATH injection
        extra-python-packages = let
          # Built with the lock-derived interpreter, so this check fails
          # loudly if the package set and the lock drift apart.
          testPkg = pythonLock.interpreter.pkgs.pyfiglet;
          rabbitWithExtra = rabbit-agent.override {
            extraPythonPackages = [ testPkg ];
          };
        in pkgs.runCommand "rabbit-extra-python-packages" { } ''
          set -e
          echo "=== Checking extraPythonPackages PYTHONPATH injection ==="

          grep -q "PYTHONPATH" ${rabbitWithExtra}/bin/rabbit || \
            (echo "FAIL: PYTHONPATH not in wrapper"; exit 1)
          echo "PASS: PYTHONPATH present in wrapper"

          grep -q "${testPkg}" ${rabbitWithExtra}/bin/rabbit || \
            (echo "FAIL: test package path not in PYTHONPATH"; exit 1)
          echo "PASS: test package path found in wrapper"

          echo "=== Checking base package has no PYTHONPATH ==="
          if grep -q "PYTHONPATH" ${rabbit-agent}/bin/rabbit; then
            echo "FAIL: base package should not have PYTHONPATH"; exit 1
          fi
          echo "PASS: base package clean"

          echo "=== All extraPythonPackages checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Exercise the actual uv2nix environment, not only the selector.
        python-lock-derived = pkgs.runCommand "rabbit-python-lock-derived" { } ''
          set -e
          echo "=== Checking Nix Python derives from pm/lock.json ==="
          family=${pythonLock.family}
          echo "locked family: $family"
          if [ "$family" != "$(${rabbitVenv}/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" ]; then
            echo "FAIL: selected interpreter major.minor does not match pm/lock.json"; exit 1
          fi
          echo "PASS: interpreter matches lock"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Selection must reject a package set without the locked family.
        python-lock-no-fallback = let
          inherit (pythonLock) selectPython;
          lockedFamily = pythonLock.family;
          # Fake package sets as data: matching family -> selected; absent
          # family -> throw (never silently pick another interpreter).
          matching = { "python${builtins.replaceStrings [ "." ] [ "" ] lockedFamily}" = "fake-python-matching"; };
          missing = { };
          selected = selectPython lockedFamily matching;
          threw = !(builtins.tryEval (selectPython lockedFamily missing)).success;
        in pkgs.runCommand "rabbit-python-lock-no-fallback" { } ''
          set -e
          echo "=== Checking python selector has no silent fallback ==="
          if [ "${toString (selected == "fake-python-matching")}" != "1" ] || [ "${toString threw}" != "1" ]; then
            echo "FAIL: selector behavior wrong (selected=${toString selected} threw=${toString threw})"; exit 1
          fi
          echo "PASS: selector picks locked family, throws on missing family"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraDependencyGroups passes through to python.nix
        extra-dependency-groups = let
          rabbitWithGroups = rabbit-agent.override {
            extraDependencyGroups = [ "exa" ];
          };
        in pkgs.runCommand "rabbit-extra-dependency-groups" { } ''
          set -e
          echo "=== Checking extraDependencyGroups override evaluates ==="

          # Eval-only: verify the override produces valid derivation paths
          # without building the full venv (which is expensive and redundant
          # since the mechanism is just list concatenation into python.nix).
          echo "derivation: ${rabbitWithGroups}"
          echo "venv: ${rabbitWithGroups.rabbitVenv}"
          echo "PASS: extraDependencyGroups override evaluates cleanly"

          echo "=== All extraDependencyGroups checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Regression guard: messaging deps live outside [all], so the
        # #messaging variant must actually ship discord.py — otherwise
        # `nix profile install .#messaging` regresses to the broken default.
        messaging-variant = pkgs.runCommand "rabbit-messaging-variant" { } ''
          set -e
          echo "=== Checking discord.py importable from messaging variant ==="
          ${self'.packages.messaging.rabbitVenv}/bin/python3 -c \
            "import discord; print(discord.__version__)"
          echo "PASS: discord.py importable from messaging variant venv"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # A fresh declarative install must be current to the Rabbit it ships
        # with. Rabbit cannot stamp config.yaml in managed mode, so an
        # unstamped file reads as version 0 and trips the support-floor
        # warning at every boot.
        generated-config-version =
          let
            common = import ./moduleCommon.nix { inherit lib; };
            configFiles = common.mkConfigFiles {
              inherit pkgs;
              cfg = {
                package = rabbit-agent;
                extraPythonPackages = [ ];
                extraDependencyGroups = [ ];
                configFile = null;
                settings.model.default = "test/nix-model";
              };
              workingDirectory = "/var/lib/rabbit/workspace";
            };
          in
          pkgs.runCommand "rabbit-generated-config-version" { } ''
            set -e
            export HOME=$(mktemp -d) RABBIT_HOME=$(mktemp -d)
            ${configMergeScript} ${configFiles.generated} "$RABBIT_HOME/config.yaml"
            ${rabbitVenv}/bin/python3 -c '
            from rabbit_cli.config import check_config_version
            current, latest = check_config_version(raise_on_parse_error=True)
            assert current == latest, f"generated config.yaml reads as v{current}, package is v{latest}"
            print(f"PASS: generated config.yaml is at v{current}")
            '
            mkdir -p $out
            echo "ok" > $out/result
          '';

        # ── Config merge + round-trip test ────────────────────────────────
        # Tests the merge script (Nix activation behavior) across 7
        # scenarios, then verifies Python's load_config() reads correctly.
        config-roundtrip = let
          # Nix settings used across scenarios
          nixSettings = pkgs.writeText "nix-settings.json" (builtins.toJSON {
            model = "test/nix-model";
            toolsets = ["nix-toolset"];
            terminal = { backend = "docker"; timeout = 999; };
            mcp_servers = {
              nix-server = { command = "echo"; args = ["nix"]; };
            };
          });

          # Pre-built YAML fixtures for each scenario
          fixtureB = pkgs.writeText "fixture-b.yaml" ''
            model: "old-model"
            mcp_servers:
              old-server:
                url: "http://old"
          '';
          fixtureC = pkgs.writeText "fixture-c.yaml" ''
            skills:
              disabled:
                - skill-a
                - skill-b
            session_reset:
              mode: idle
              idle_minutes: 30
            streaming:
              enabled: true
            fallback_model:
              provider: openrouter
              model: test-fallback
          '';
          fixtureD = pkgs.writeText "fixture-d.yaml" ''
            model: "user-model"
            skills:
              disabled:
                - skill-x
            streaming:
              enabled: true
              transport: edit
          '';
          fixtureE = pkgs.writeText "fixture-e.yaml" ''
            mcp_servers:
              user-server:
                url: "http://user-mcp"
              nix-server:
                command: "old-cmd"
                args: ["old"]
          '';
          fixtureF = pkgs.writeText "fixture-f.yaml" ''
            terminal:
              cwd: "/user/path"
              custom_key: "preserved"
              env_passthrough:
                - USER_VAR
          '';

        in pkgs.runCommand "rabbit-config-roundtrip" {
          nativeBuildInputs = [ pkgs.jq ];
        } ''
          set -e
          export HOME=$(mktemp -d)
          ERRORS=""

          fail() { ERRORS="$ERRORS\nFAIL: $1"; }

          # Helper: run merge then load with Python, output merged JSON
          merge_and_load() {
            local rabbit_home="$1"
            export RABBIT_HOME="$rabbit_home"
            ${configMergeScript} ${nixSettings} "$rabbit_home/config.yaml"
            ${rabbitVenv}/bin/python3 -c '
import json, sys
from rabbit_cli.config import load_config
json.dump(load_config(), sys.stdout, default=str)
'
          }

          # ═══════════════════════════════════════════════════════════════
          # Scenario A: Fresh install — no existing config.yaml
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario A: Fresh install ==="
          A_HOME=$(mktemp -d)
          A_CONFIG=$(merge_and_load "$A_HOME")

          echo "$A_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "A: model not set from Nix"
          echo "$A_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "A: MCP nix-server missing"
          echo "PASS: Scenario A"

          # ═══════════════════════════════════════════════════════════════
          # Scenario B: Nix keys override existing values
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario B: Nix overrides ==="
          B_HOME=$(mktemp -d)
          install -m 0644 ${fixtureB} "$B_HOME/config.yaml"
          B_CONFIG=$(merge_and_load "$B_HOME")

          echo "$B_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "B: Nix model did not override"
          echo "PASS: Scenario B"

          # ═══════════════════════════════════════════════════════════════
          # Scenario C: User-only keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario C: User keys preserved ==="
          C_HOME=$(mktemp -d)
          install -m 0644 ${fixtureC} "$C_HOME/config.yaml"
          C_CONFIG=$(merge_and_load "$C_HOME")

          echo "$C_CONFIG" | jq -e '.skills.disabled == ["skill-a", "skill-b"]' > /dev/null \
            || fail "C: skills.disabled not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.mode == "idle"' > /dev/null \
            || fail "C: session_reset.mode not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.idle_minutes == 30' > /dev/null \
            || fail "C: session_reset.idle_minutes not preserved"
          echo "$C_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "C: streaming.enabled not preserved"
          echo "$C_CONFIG" | jq -e '.fallback_model.provider == "openrouter"' > /dev/null \
            || fail "C: fallback_model not preserved"
          echo "PASS: Scenario C"

          # ═══════════════════════════════════════════════════════════════
          # Scenario D: Mixed — Nix wins for its keys, user keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario D: Mixed merge ==="
          D_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$D_HOME/config.yaml"
          D_CONFIG=$(merge_and_load "$D_HOME")

          echo "$D_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "D: Nix model did not override user model"
          echo "$D_CONFIG" | jq -e '.skills.disabled == ["skill-x"]' > /dev/null \
            || fail "D: user skills not preserved"
          echo "$D_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "D: user streaming not preserved"
          echo "PASS: Scenario D"

          # ═══════════════════════════════════════════════════════════════
          # Scenario E: MCP additive merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario E: MCP additive merge ==="
          E_HOME=$(mktemp -d)
          install -m 0644 ${fixtureE} "$E_HOME/config.yaml"
          E_CONFIG=$(merge_and_load "$E_HOME")

          echo "$E_CONFIG" | jq -e '.mcp_servers."user-server".url == "http://user-mcp"' > /dev/null \
            || fail "E: user MCP server not preserved"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "E: Nix MCP server did not override same-name user server"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".args == ["nix"]' > /dev/null \
            || fail "E: Nix MCP server args wrong"
          echo "PASS: Scenario E"

          # ═══════════════════════════════════════════════════════════════
          # Scenario F: Nested deep merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario F: Nested deep merge ==="
          F_HOME=$(mktemp -d)
          install -m 0644 ${fixtureF} "$F_HOME/config.yaml"
          F_CONFIG=$(merge_and_load "$F_HOME")

          echo "$F_CONFIG" | jq -e '.terminal.backend == "docker"' > /dev/null \
            || fail "F: Nix terminal.backend did not override"
          echo "$F_CONFIG" | jq -e '.terminal.timeout == 999' > /dev/null \
            || fail "F: Nix terminal.timeout did not override"
          echo "$F_CONFIG" | jq -e '.terminal.custom_key == "preserved"' > /dev/null \
            || fail "F: terminal.custom_key not preserved"
          echo "$F_CONFIG" | jq -e '.terminal.cwd == "/user/path"' > /dev/null \
            || fail "F: user terminal.cwd not preserved when Nix does not set it"
          echo "$F_CONFIG" | jq -e '.terminal.env_passthrough == ["USER_VAR"]' > /dev/null \
            || fail "F: user terminal.env_passthrough not preserved"
          echo "PASS: Scenario F"

          # ═══════════════════════════════════════════════════════════════
          # Scenario G: Idempotency — merging twice yields the same result
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario G: Idempotency ==="
          G_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$G_HOME/config.yaml"
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          FIRST=$(cat "$G_HOME/config.yaml")
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          SECOND=$(cat "$G_HOME/config.yaml")

          if [ "$FIRST" != "$SECOND" ]; then
            fail "G: second merge produced different output"
            echo "--- first ---"
            echo "$FIRST"
            echo "--- second ---"
            echo "$SECOND"
          fi
          echo "PASS: Scenario G"

          # ═══════════════════════════════════════════════════════════════
          # Report
          # ═══════════════════════════════════════════════════════════════
          if [ -n "$ERRORS" ]; then
            echo ""
            echo "FAILURES:"
            echo -e "$ERRORS"
            exit 1
          fi

          echo ""
          echo "=== All 7 merge scenarios passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';
      };
    };
}
