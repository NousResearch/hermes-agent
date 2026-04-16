# NixOS VM test: verify env.d/ activation behavior end-to-end.
{ pkgs, nixosModule }:

pkgs.testers.nixosTest {
  name = "hermes-env-d";

  nodes.machine = { config, pkgs, lib, ... }: {
    imports = [ nixosModule ];

    # Simulate sops-nix-like secrets: plain files with correct ownership.
    system.activationScripts."test-secrets" = lib.stringAfter [ "users" ] ''
      mkdir -p /run/test-secrets
      echo "API_KEY=sk-test-12345" > /run/test-secrets/hermes-api
      echo "DISCORD_TOKEN=tok-discord-abc" > /run/test-secrets/hermes-discord
      chown hermes:hermes /run/test-secrets/hermes-api /run/test-secrets/hermes-discord
      chmod 0400 /run/test-secrets/hermes-api /run/test-secrets/hermes-discord
    '';

    services.hermes-agent = {
      enable = true;
      environment = {
        MODEL = "test/nix-model";
        OPENAI_BASE_URL = "https://test.example.com/v1";
      };
      environmentFiles = [
        "/run/test-secrets/hermes-api"
        "/run/test-secrets/hermes-discord"
      ];
      settings.model = "test/nix-model";
    };

    # Don't start the service — no real API keys.
    systemd.services.hermes-agent.wantedBy = lib.mkForce [ ];
  };

  testScript = ''
    machine.wait_for_unit("multi-user.target")

    with subtest("env.d directory exists with correct permissions"):
        machine.succeed("test -d /var/lib/hermes/.hermes/env.d")
        result = machine.succeed("stat -c '%U:%G' /var/lib/hermes/.hermes/env.d").strip()
        assert result == "hermes:hermes", f"ownership: {result}"
        perms = machine.succeed("stat -c '%a' /var/lib/hermes/.hermes/env.d").strip()
        assert perms == "2770", f"permissions: {perms}"

    with subtest("nix-environment.env written correctly"):
        machine.succeed("test -f /var/lib/hermes/.hermes/env.d/nix-environment.env")
        content = machine.succeed("cat /var/lib/hermes/.hermes/env.d/nix-environment.env")
        assert "MODEL=test/nix-model" in content, f"MODEL not found in: {content}"
        assert "OPENAI_BASE_URL=https://test.example.com/v1" in content
        perms = machine.succeed("stat -c '%a' /var/lib/hermes/.hermes/env.d/nix-environment.env").strip()
        assert perms == "640", f"permissions: {perms}"

    with subtest("environmentFiles are symlinked not copied"):
        machine.succeed("test -L /var/lib/hermes/.hermes/env.d/nix-0.env")
        target = machine.succeed("readlink /var/lib/hermes/.hermes/env.d/nix-0.env").strip()
        assert target == "/run/test-secrets/hermes-api", f"nix-0 -> {target}"
        machine.succeed("test -L /var/lib/hermes/.hermes/env.d/nix-1.env")
        target = machine.succeed("readlink /var/lib/hermes/.hermes/env.d/nix-1.env").strip()
        assert target == "/run/test-secrets/hermes-discord", f"nix-1 -> {target}"

    with subtest("hermes user can read through symlinks"):
        content = machine.succeed("sudo -u hermes cat /var/lib/hermes/.hermes/env.d/nix-0.env")
        assert "API_KEY=sk-test-12345" in content
        content = machine.succeed("sudo -u hermes cat /var/lib/hermes/.hermes/env.d/nix-1.env")
        assert "DISCORD_TOKEN=tok-discord-abc" in content

    with subtest("old .env not written by activation"):
        machine.fail("test -f /var/lib/hermes/.hermes/.env")

    with subtest("no extra nix-N.env symlinks"):
        machine.fail("test -e /var/lib/hermes/.hermes/env.d/nix-2.env")

    with subtest("env.d files are glob-discoverable"):
        result = machine.succeed(
            "sudo -u hermes ls /var/lib/hermes/.hermes/env.d/*.env | wc -l"
        ).strip()
        assert result == "3", f"expected 3 env files, got {result}"
  '';
}
