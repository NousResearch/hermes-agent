# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }:
{
  perSystem =
    {
      pkgs,
      lib,
      inputs',
      ...
    }:
    let
      dirtyRevision = inputs.self.dirtyRev or null;
      # inputs.self.sourceInfo is a string (store path) in current Nix, not a
      # record. The metadata fields are available as top-level attributes on
      # inputs.self directly. A flake URL's requested ref is intentionally not
      # exposed as self.ref after Nix resolves it to an immutable source, so
      # branch is normally null for remote flakes. Preserve null rather than
      # inventing a sentinel that could be a real branch name.
      rev = inputs.self.rev or (if dirtyRevision != null then builtins.substring 0 40 dirtyRevision else null);
      revCount = inputs.self.revCount or null;
      rawRef = inputs.self.ref or null;
      branch = if rawRef != null then builtins.replaceStrings ["refs/heads/"] [""] rawRef else null;
      dirty = dirtyRevision != null;
      lastModified = inputs.self.lastModified or null;
      minimal = pkgs.callPackage ./hermes-agent.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
        npm-lockfile-fix = inputs'.npm-lockfile-fix.packages.default;
        inherit rev revCount branch dirty lastModified;
      };

      # All platform-portable optional integrations pre-built.
      full = minimal.override {
        extraDependencyGroups = [
          "anthropic"
          "azure-identity"
          "bedrock"
          "daytona"
          "dingtalk"
          "edge-tts"
          "exa"
          "fal"
          "feishu"
          "firecrawl"
          "hindsight"
          "honcho"
          "messaging"
          "modal"
          "parallel-web"
          "tts-premium"
          "voice"
        ]
        # matrix is Linux-only (oqs/liboqs lacks aarch64-darwin wheels).
        ++ lib.optionals pkgs.stdenv.isLinux [ "matrix" ];
      };
    in
    {
      packages = {
        default = full;

        inherit minimal;

        # Ships discord.py + python-telegram-bot + slack-sdk so a plain
        # `nix profile install .#messaging` connects to Discord/Telegram/Slack
        # on first run — lazy-install can't write to the read-only /nix/store.
        messaging = minimal.override {
          extraDependencyGroups = [ "messaging" ];
        };

        tui = full.hermesTui;
        web = full.hermesWeb;
        desktop = full.hermesDesktop;

        update-npm-lockfile = full.hermesNpmLib.updateNpmLockfile;
      };
    };
}
