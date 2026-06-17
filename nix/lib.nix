# nix/lib.nix — Shared helpers for nix stuff
#
# All npm packages in this repo are workspace members sharing a single
# root package-lock.json.  mkNpmPassthru provides the shared src, npmDeps,
# npmRoot, and npmDepsFetcherVersion so individual .nix files don't
# duplicate them.  One hash to rule them all.
#
# mkNpmPassthru returns packageJsonPath (e.g. "ui-tui/package.json")
# instead of a per-package devShellHook.  The root devshell hook
# (mkNpmDevShellHook) collects all package.json paths, stamps them,
# and if any changed, runs a single `npm i --package-lock-only` from
# root to update the lockfile, then `npm ci` if the lockfile changed.
{
  pkgs,
  npm-lockfile-fix,
  nodejs,
}:
let
  # The workspace root — where the single package-lock.json lives.
  src = ../.;

  # Single npm deps fetch from the workspace root lockfile.
  # All workspace packages share this derivation.
  npmDepsHash = "sha256-m9cjbjzi4SaFCjODfdrawS5e+1ag+MpRn528/upSNqo=";

  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    fetcherVersion = 2;
    hash = npmDepsHash;
  };
in
{
  # Returns a buildNpmPackage-compatible attrs set that provides:
  #   src, npmDeps, npmRoot, npmDepsFetcherVersion
  #   patchPhase             — ensures root lockfile has exactly one trailing newline
  #   nativeBuildInputs      — [ updateLockfileScript ] (list, prepend with ++ for more)
  #   passthru.packageJsonPath — relative path to this workspace's package.json
  #   nodejs                 — fixed nodejs version for all packages we use in the repo
  #
  # NOTE: npmConfigHook runs `diff` between the source lockfile and the
  # npm-deps cache lockfile. fetchNpmDeps preserves whatever trailing
  # newlines the lockfile has. The patchPhase normalizes to exactly one
  # trailing newline so both sides always match.
  #
  # Usage:
  #   npm = hermesNpmLib.mkNpmPassthru { folder = "ui-tui"; attr = "tui"; pname = "hermes-tui"; };
  #   pkgs.buildNpmPackage (npm // {
  #     sourceRoot = "ui-tui";
  #     buildPhase = '' ... '';
  #     installPhase = '' ... '';
  #   })
  mkNpmPassthru =
    {
      folder, # repo-relative folder with package.json, e.g. "ui-tui"
      attr, # flake package attr, e.g. "tui"
      ...
    }:
    let
      # No sourceRoot — the workspace root (with the single package-lock.json)
      # is auto-detected as sourceRoot by nix.  npmRoot stays at "."
      # so npmConfigHook finds the lockfile there.
    in
    {
      inherit src npmDeps nodejs;
      npmRoot = ".";
      npmDepsFetcherVersion = 2;

      ELECTRON_SKIP_BINARY_DOWNLOAD = 1;

      patchPhase = ''
        runHook prePatch
        # Normalize trailing newlines on the root lockfile so source and
        # npm-deps always match, regardless of what fetchNpmDeps preserves.
        sed -i -z 's/\\n*$/\\n/' package-lock.json

        # Make npmConfigHook's byte-for-byte diff newline-agnostic by
        # replacing its hardcoded /nix/store/.../diff with a wrapper that
        # normalizes trailing newlines on both sides before comparing.
        mkdir -p "$TMPDIR/bin"
        cat > "$TMPDIR/bin/diff" << DIFFWRAP
        #!/bin/sh
        f1=\\$(mktemp) && sed -z 's/\\n*$/\\n/' "\\$1" > "\\$f1"
        f2=\\$(mktemp) && sed -z 's/\\n*$/\\n/' "\\$2" > "\\$f2"
        ${pkgs.diffutils}/bin/diff "\\$f1" "\\$f2" && rc=0 || rc=\\$?
        rm -f "\\$f1" "\\$f2"
        exit \\$rc
        DIFFWRAP
        chmod +x "$TMPDIR/bin/diff"
        export PATH="$TMPDIR/bin:$PATH"

        runHook postPatch
      '';

      nativeBuildInputs = [
        (pkgs.writeShellScriptBin "update_${attr}_lockfile" ''
          set -euox pipefail

          REPO_ROOT=$(git rev-parse --show-toplevel)

          # All workspace packages share the root lockfile.
          cd "$REPO_ROOT"
          rm -rf node_modules/
          ${pkgs.lib.getExe' nodejs "npm"} cache clean --force
          CI=true ${pkgs.lib.getExe' nodejs "npm"} install --workspaces
          ${pkgs.lib.getExe npm-lockfile-fix} ./package-lock.json

          # Hash lives in lib.nix — just rebuild to verify.
          nix build .#${attr}
          echo "Lockfile updated and build verified for .#${attr}"
        '')
      ];

      passthru = {
        packageJsonPath = "${folder}/package.json";
      };
    };

  # Single devshell hook for all npm workspace packages.
  #
  # Takes a list of package.json relative paths (from mkNpmPassthru .passthru.packageJsonPath),
  # stamps all of them, and if any changed:
  #   1. Runs `npm i --package-lock-only` from root to update the lockfile
  #   2. If the lockfile changed, runs `npm ci` + fix-lockfiles
  #
  # fixLockfilesExe: absolute path to the fix-lockfiles binary
  # (from pkgs.lib.getExe self'.packages.fix-lockfiles in devShell.nix).
  mkNpmDevShellHook =
    packageJsonPaths: fixLockfilesExe:
    pkgs.writeShellScript "npm-dev-hook" ''
      REPO_ROOT=$(git rev-parse --show-toplevel)

      # Stamp all workspace package.jsons into one file.
      STAMP_DIR=".nix-stamps"
      STAMP="$STAMP_DIR/npm-package-jsons"
      STAMP_VALUE=$(
        ${pkgs.coreutils}/bin/sha256sum ${
          pkgs.lib.concatMapStringsSep " " (p: "\"$REPO_ROOT/${p}\"") packageJsonPaths
        } 2>/dev/null | ${pkgs.coreutils}/bin/sort | ${pkgs.coreutils}/bin/sha256sum | awk '{print $1}'
      )

      PKG_CHANGED=false
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        PKG_CHANGED=true
        echo "npm: package.json changed, updating lockfile..."
        ( cd "$REPO_ROOT" && ${pkgs.lib.getExe' nodejs "npm"} i --package-lock-only --silent --no-fund --no-audit 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$STAMP_VALUE" > "$STAMP"
      fi

      # Check if lockfile changed (either from the npm i above or from an
      # external edit).  Runs npm ci if so.
      LOCK_STAMP="$STAMP_DIR/root-lockfile"
      LOCK_STAMP_VALUE=$(sha256sum "$REPO_ROOT/package-lock.json" 2>/dev/null | awk '{print $1}')
      if [ ! -f "$LOCK_STAMP" ] || [ "$(cat "$LOCK_STAMP")" != "$LOCK_STAMP_VALUE" ]; then
        echo "npm: package-lock.json changed, running npm ci..."
        ( cd "$REPO_ROOT" && CI=true ${pkgs.lib.getExe' nodejs "npm"} ci --silent --no-fund --no-audit 2>/dev/null )
        mkdir -p "$STAMP_DIR"
        echo "$LOCK_STAMP_VALUE" > "$LOCK_STAMP"
      fi
    '';

  # Build `fix-lockfiles` bin that checks/updates the single npmDepsHash
  #   fix-lockfiles --check   # exit 1 if any hash is stale
  #   fix-lockfiles --apply   # rewrite stale hashes in place
  #   fix-lockfiles           # alias of --apply
  # Writes machine-readable fields (stale, changed, report) to $GITHUB_OUTPUT
  # when set, so CI workflows can post a sticky PR comment directly.
  mkFixLockfiles =
    {
      attr, # flake package attr for fallback verification build, e.g. "tui"
    }:
    pkgs.writeShellScriptBin "fix-lockfiles" ''
      set -uox pipefail
      MODE="''${1:---apply}"
      case "$MODE" in
        --check|--apply) ;;
        -h|--help)
          echo "usage: fix-lockfiles [--check|--apply]"
          exit 0 ;;
        *)
          echo "usage: fix-lockfiles [--check|--apply]" >&2
          exit 2 ;;
      esac

      REPO_ROOT="$(git rev-parse --show-toplevel)"
      cd "$REPO_ROOT"

      # When running in GH Actions, emit Markdown links in the report pointing
      # at the offending line of the nix file (and the lockfile) at the exact
      # commit that was checked. LINK_SHA should be set by the workflow to the
      # PR head SHA; falls back to GITHUB_SHA (which on pull_request is the
      # test-merge commit, still browseable).
      LINK_SERVER="''${GITHUB_SERVER_URL:-https://github.com}"
      LINK_REPO="''${GITHUB_REPOSITORY:-}"
      LINK_SHA="''${LINK_SHA:-''${GITHUB_SHA:-}}"

      STALE=0
      FIXED=0
      REPORT=""

      # All workspace packages share the root package-lock.json, so
      # we only need to check the hash once.
      LOCK_FILE="package-lock.json"
      LIB_FILE="nix/lib.nix"
      NEW_HASH=$(${pkgs.lib.getExe pkgs.prefetch-npm-deps} "$LOCK_FILE" 2>/dev/null)
      if [ -z "$NEW_HASH" ]; then
        echo "prefetch-npm-deps failed, falling back to nix build" >&2
        OUTPUT=$(nix build ".#${attr}.npmDeps" --no-link --print-build-logs 2>&1)
        STATUS=$?
        if [ "$STATUS" -eq 0 ]; then
          echo "ok (via nix build)"
          exit 0
        fi
        NEW_HASH=$(echo "$OUTPUT" | awk '/got:/ {print $2; exit}')
        if [ -z "$NEW_HASH" ]; then
          if echo "$OUTPUT" | grep -qE "throttled|HTTP error 418|substituter .* is disabled|some outputs of .* are not valid"; then
            echo "skipped (transient cache failure — see primary nix build for real status)" >&2
            echo "$OUTPUT" | tail -8 >&2
            exit 0
          fi
          echo "build failed with no hash mismatch:" >&2
          echo "$OUTPUT" | tail -40 >&2
          exit 1
        fi
      fi

      OLD_HASH=$(grep -oE 'npmDepsHash = "sha256-[^"]+"' "$LIB_FILE" | head -1 \
        | sed -E 's/npmDepsHash = "(.*)"/\1/')

      # prefetch-npm-deps says the hash already matches — but it only hashes the
      # lockfile *contents* and can disagree with fetchNpmDeps + npmConfigHook,
      # which validate the full source lockfile against the realized deps cache.
      # Trusting prefetch alone produced false "ok" results while the actual
      # build was broken (e.g. lockfile engines/os/cpu fields the pinned nixpkgs
      # strips from the deps cache, tripping npmConfigHook). So when prefetch
      # claims the hash is current, confirm with a real consumer build before
      # believing it.
      if [ "$NEW_HASH" = "$OLD_HASH" ]; then
        if VERIFY_OUT=$(nix build ".#${attr}" --no-link --print-build-logs 2>&1); then
          echo "ok"
          if [ -n "''${GITHUB_OUTPUT:-}" ]; then
            { echo "stale=false"; echo "changed=false"; } >> "$GITHUB_OUTPUT"
          fi
          exit 0
        fi
        # Build failed despite a matching hash. A fixed-output 'got:' means
        # prefetch genuinely disagreed with fetchNpmDeps — adopt the real hash
        # and fall through to the stale-handling path below.
        CORRECT_HASH=$(echo "$VERIFY_OUT" | awk '/got:/ {print $2; exit}')
        if [ -n "$CORRECT_HASH" ]; then
          echo "prefetch-npm-deps reported current ($OLD_HASH) but fetchNpmDeps wants $CORRECT_HASH" >&2
          NEW_HASH="$CORRECT_HASH"
        elif echo "$VERIFY_OUT" | grep -qE "throttled|HTTP error 418|substituter .* is disabled|some outputs of .* are not valid"; then
          echo "skipped (transient cache failure — see primary nix build for real status)" >&2
          echo "$VERIFY_OUT" | tail -8 >&2
          exit 0
        else
          # Not a stale-hash problem — surface it honestly instead of "ok".
          echo "::error::nix build .#${attr} failed and it is NOT a stale npmDepsHash (no 'got:' hash in output)." >&2
          echo "The committed lockfile may be incompatible with the pinned nixpkgs" >&2
          echo "(e.g. engines/os/cpu fields that prefetch-npm-deps strips from the" >&2
          echo "deps cache, tripping npmConfigHook). fix-lockfiles cannot repair this." >&2
          echo "$VERIFY_OUT" | tail -40 >&2
          if [ -n "''${GITHUB_OUTPUT:-}" ]; then
            { echo "stale=false"; echo "changed=false"; } >> "$GITHUB_OUTPUT"
          fi
          exit 1
        fi
      fi

      HASH_LINE=$(grep -n 'npmDepsHash = "sha256-' "$LIB_FILE" | head -1 | cut -d: -f1)
      echo "stale: $LIB_FILE:$HASH_LINE $OLD_HASH -> $NEW_HASH"
      STALE=1

      if [ -n "$LINK_REPO" ] && [ -n "$LINK_SHA" ]; then
        LIB_URL="$LINK_SERVER/$LINK_REPO/blob/$LINK_SHA/$LIB_FILE#L$HASH_LINE"
        LOCK_URL="$LINK_SERVER/$LINK_REPO/blob/$LINK_SHA/$LOCK_FILE"
        REPORT="- [\`$LIB_FILE:$HASH_LINE\`]($LIB_URL): \`$OLD_HASH\` → \`$NEW_HASH\` — lockfile: [\`$LOCK_FILE\`]($LOCK_URL)"$'\\n'
      else
        REPORT="- \`$LIB_FILE:$HASH_LINE\`: \`$OLD_HASH\` → \`$NEW_HASH\`"$'\\n'
      fi

      if [ "$MODE" = "--apply" ]; then
        sed -i -E "s|npmDepsHash = \"sha256-[^\"]+\";|npmDepsHash = \"$NEW_HASH\";|" "$LIB_FILE"
        if ! nix build ".#${attr}.npmDeps" --no-link --print-build-logs 2>/dev/null; then
          # prefetch-npm-deps may disagree with fetchNpmDeps (it hashes
          # the lockfile contents, not the full source tree).  Extract the
          # correct hash from the nix build error and retry.
          RETRY_OUTPUT=$(nix build ".#${attr}.npmDeps" --no-link --print-build-logs 2>&1)
          CORRECT_HASH=$(echo "$RETRY_OUTPUT" | awk '/got:/ {print $2; exit}')
          if [ -n "$CORRECT_HASH" ]; then
            echo "prefetch-npm-deps gave $NEW_HASH but nix wants $CORRECT_HASH — retrying" >&2
            sed -i -E "s|npmDepsHash = \"sha256-[^\"]+\";|npmDepsHash = \"$CORRECT_HASH\";|" "$LIB_FILE"
            if ! nix build ".#${attr}.npmDeps" --no-link --print-build-logs; then
              echo "verification build failed after hash retry" >&2
              exit 1
            fi
            NEW_HASH="$CORRECT_HASH"
          else
            echo "verification build failed after hash update" >&2
            exit 1
          fi
        fi
        FIXED=1
        echo "fixed"
      fi

      if [ -n "''${GITHUB_OUTPUT:-}" ]; then
        {
          [ "$STALE" -eq 1 ] && echo "stale=true" || echo "stale=false"
          [ "$FIXED" -eq 1 ] && echo "changed=true" || echo "changed=false"
          if [ -n "$REPORT" ]; then
            echo "report<<REPORT_EOF"
            printf "%s" "$REPORT"
            echo "REPORT_EOF"
          fi
        } >> "$GITHUB_OUTPUT"
      fi

      if [ "$STALE" -eq 1 ] && [ "$MODE" = "--check" ]; then
        echo
        echo "Stale lockfile hash detected. Run:"
        echo "  nix run .#fix-lockfiles"
        exit 1
      fi

      exit 0
    '';
}
