# nix/vmTest.nix — Opt-in NixOS VM integration test
#
# Covers the NixOS module's activation, tmpfiles rules, packaged CLI wrapper,
# and credential-less systemd gateway startup. It is a package rather than a
# flake check because booting a VM and its large package closure is expensive.
# Run explicitly with: nix build .#nixos-vm-test -L
{ inputs, ... }: {
  perSystem = { pkgs, lib, ... }: {
    packages = lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "x86_64-linux") {
      nixos-vm-test = pkgs.testers.runNixOSTest {
        name = "hermes-agent-nixos-module";

        nodes.machine = { pkgs, ... }: {
          imports = [ inputs.self.nixosModules.default ];

          services.hermes-agent = {
            enable = true;
            addToSystemPackages = true;
            settings.model = {
              provider = "openrouter";
              name = "openai/gpt-4o-mini";
            };
            environmentFiles = [
              "${pkgs.writeText "hermes-vm-test-env" ''
                OPENROUTER_API_KEY=dummy-vm-test-key
              ''}"
            ];
          };

          virtualisation = {
            memorySize = 3072;
            diskSize = 8192;
          };
        };

        testScript = ''
          machine.start()
          machine.wait_for_unit("multi-user.target")

          machine.succeed(
              "test \"$(stat -c '%a %U %G' /var/lib/hermes)\" = '2770 hermes hermes'"
          )
          machine.succeed(
              "for path in .hermes .hermes/cron .hermes/sessions .hermes/logs "
              ".hermes/memories .hermes/plugins; do "
              "test \"$(stat -c '%a %U %G' /var/lib/hermes/$path)\" "
              "= '2770 hermes hermes'; done"
          )

          machine.succeed(
              "runuser -u hermes -- env HOME=/var/lib/hermes "
              "HERMES_HOME=/var/lib/hermes/.hermes hermes version 2>&1 "
              "| grep -E 'Hermes Agent v[0-9]+'"
          )
          machine.succeed("hermes --help 2>&1 | grep -q gateway")

          machine.succeed("test -f /var/lib/hermes/.hermes/.managed")

          machine.succeed("systemctl cat hermes-agent.service >/dev/null")
          # With no messaging platform credentials configured, the gateway
          # intentionally stays active to run cron jobs. Assert both the live
          # unit and its credential-less startup log instead of relying on a
          # bare wait_for_unit(), which could pass during a restart loop.
          machine.wait_until_succeeds(
              "systemctl is-active --quiet hermes-agent.service"
          )
          machine.wait_until_succeeds(
              "journalctl -u hermes-agent.service --no-pager "
              "| grep -F 'No messaging platforms enabled.'"
          )
          machine.succeed(
              "test \"$(systemctl show -P ActiveState hermes-agent.service)\" = active"
          )
        '';
      };
    };
  };
}
