# nix/lib.nix — Shared helpers for nix stuff
{
  pkgs,
  npm-lockfile-fix,
  nodejs,
}:
{
  # Returns a buildNpmPackage-compatible attrs set that provides:
  #   patchPhase             — ensures lockfile has exactly one trailing newline
  #   nativeBuildInputs      — [ updateLockfileScript ] (list, prepend with ++ for more)
  #   passthru.devShellHook  — stamp-checked npm install + hash auto-update
  #   passthru.npmLockfile   — metadata for mkFixLockfiles
  #   nodejs                 — fixed nodejs version for all packages we use in the repo
  #
  # NOTE: npmConfigHook runs `diff` between the source lockfile and the
  # npm-deps cache lockfile. fetchNpmDeps preserves whatever trailing
  # newlines the lockfile has. The patchPhase normalizes to exactly one
  # trailing newline so both sides always match.
  #
  # Usage:
  #   npm = hermesNpmLib.mkNpmPassthru { folder = "ui-tui"; attr = "tui"; pname = "hermes-tui"; };
  #   pkgs.buildNpmPackage (npm // { ... } # or:
  #   pkgs.buildNpmPackage ({ ... } // npm)
  mkNpmPassthru =
    {
      folder, # repo-relative folder with package.json, e.g. "ui-tui"
      attr, # flake package attr, e.g. "tui"
      pname, # e.g. "hermes-tui"
      nixFile ? "nix/${attr}.nix", # defaults to nix/<attr>.nix
    }:
    {
      inherit nodejs;
      patchPhase = ''
        runHook prePatch
        # Normalize trailing newlines so source and npm-deps always match,
        # regardless of what fetchNpmDeps preserves.
        sed -i -z 's/\n*$/\n/' package-lock.json

        # Make npmConfigHook's byte-for-byte diff newline-agnostic by
        # replacing its hardcoded /nix/store/.../diff with a wrapper that
        # normalizes trailing newlines on both sides before comparing.
        mkdir -p "$TMPDIR/bin"
        cat > "$TMPDIR/bin/diff" << DIFFWRAP
        #!/bin/sh
        f1=\$(mktemp) && sed -z 's/\n*$/\n/' "\$1" > "\$f1"
        f2=\$(mktemp) && sed -z 's/\n*$/\n/' "\$2" > "\$f2"
        ${pkgs.diffutils}/bin/diff "\$f1" "\$f2" && rc=0 || rc=\$?
        rm -f "\$f1" "\$f2"
        exit \$rc
        DIFFWRAP
        chmod +x "$TMPDIR/bin/diff"
        export PATH="$TMPDIR/bin:$PATH"

        runHook postPatch
      '';

      nativeBuildInputs = [
        (pkgs.writeShellScriptBin "update_${attr}_lockfile" ''
          set -euox pipefail

          REPO_ROOT=$(git rev-parse --show-toplevel)

          cd "$REPO_ROOT/${folder}"
          rm -rf node_modules/
          ${pkgs.lib.getExe' nodejs "npm"} cache clean --force
          CI=true ${pkgs.lib.getExe' nodejs "npm"} install
          ${pkgs.lib.getExe npm-lockfile-fix} ./package-lock.json

          NIX_FILE="$REPO_ROOT/${nixFile}"
          sed -i "s/hash = \"[^\"]*\";/hash = \"\";/" $NIX_FILE
          NIX_OUTPUT=$(nix build .#${attr} 2>&1 || true)
          NEW_HASH=$(echo "$NIX_OUTPUT" | grep 'got:' | awk '{print $2}')
          echo got new hash $NEW_HASH
          sed -i "s|hash = \"[^\"]*\";|hash = \"$NEW_HASH\";|" $NIX_FILE
          nix build .#${attr}
          echo "Updated npm hash in $NIX_FILE to $NEW_HASH"
        '')
      ];

      passthru = {
        devShellHook = pkgs.writeShellScript "npm-dev-hook-${pname}" ''
          REPO_ROOT=$(git rev-parse --show-toplevel)

          _hermes_npm_stamp() {
            sha256sum "${folder}/package.json" "${folder}/package-lock.json" \
              2>/dev/null | sha256sum | awk '{print $1}'
          }
          STAMP=".nix-stamps/${pname}"
          STAMP_VALUE="$(_hermes_npm_stamp)"
          if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
            echo "${pname}: installing npm dependencies..."
            ( cd ${folder} && CI=true ${pkgs.lib.getExe' nodejs "npm"} install --silent --no-fund --no-audit 2>/dev/null )

            # Auto-update the nix hash so it stays in sync with the lockfile
            echo "${pname}: prefetching npm deps..."
            NIX_FILE="$REPO_ROOT/${nixFile}"
            if NEW_HASH=$(${pkgs.lib.getExe pkgs.prefetch-npm-deps} "${folder}/package-lock.json" 2>/dev/null); then
              sed -i "s|hash = \"sha256-[A-Za-z0-9+/=]+\"|hash = \"$NEW_HASH\";|" "$NIX_FILE"
              echo "${pname}: updated hash to $NEW_HASH"
            else
              echo "${pname}: warning: prefetch failed, run 'nix run .#fix-lockfiles' manually" >&2
            fi

            mkdir -p .nix-stamps
            _hermes_npm_stamp > "$STAMP"
          fi
          unset -f _hermes_npm_stamp
        '';

        npmLockfile = {
          inherit attr folder nixFile;
        };
      };
    };

  # Aggregate `fix-lockfiles` bin from a list of packages carrying
  #   passthru.npmLockfile = { attr; folder; nixFile; };
  # Invocations:
  #   fix-lockfiles --check   # exit 1 if any hash is stale
  #   fix-lockfiles --apply   # rewrite stale hashes in place
  #   fix-lockfiles           # alias of --apply
  # Writes machine-readable fields (stale, changed, report) to $GITHUB_OUTPUT
  # when set, so CI workflows can post a sticky PR comment directly.
  mkFixLockfiles =
    {
      packages, # list of packages with passthru.npmLockfile
    }:
    let
      entries = map (p: p.passthru.npmLockfile) packages;
      entryArgs = pkgs.lib.concatMapStringsSep " " (e: "\"${e.attr}:${e.folder}:${e.nixFile}\"") entries;
    in
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

      ENTRIES=(${entryArgs})

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

      for entry in "''${ENTRIES[@]}"; do
        IFS=":" read -r ATTR FOLDER NIX_FILE <<< "$entry"
        echo "==> .#$ATTR ($FOLDER -> $NIX_FILE)"
        OUTPUT=$(nix build ".#$ATTR.npmDeps" --no-link --print-build-logs 2>&1)
        STATUS=$?
        if [ "$STATUS" -eq 0 ]; then
          echo "    ok"
          continue
        fi

        NEW_HASH=$(echo "$OUTPUT" | awk '/got:/ {print $2; exit}')
        if [ -z "$NEW_HASH" ]; then
          # Magic-Nix-Cache occasionally returns HTTP 418 / cache-throttled
          # mid-run; nix then prints "outputs … not valid, so checking is
          # not possible" without a `got:` line.  That's an infrastructure
          # blip, not a stale lockfile — warn + skip rather than failing
          # the lint.  A real hash mismatch would still surface in the
          # primary `.#$ATTR` build, which is a separate CI job.
          if echo "$OUTPUT" | grep -qE "throttled|HTTP error 418|substituter .* is disabled|some outputs of .* are not valid"; then
            echo "    skipped (transient cache failure — see primary nix build for real status)" >&2
            echo "$OUTPUT" | tail -8 >&2
            continue
          fi
          echo "    build failed with no hash mismatch:" >&2
          echo "$OUTPUT" | tail -40 >&2
          exit 1
        fi

        HASH_LINE=$(grep -n 'hash = "sha256-' "$NIX_FILE" | head -1 | cut -d: -f1)
        OLD_HASH=$(grep -oE 'hash = "sha256-[^"]+"' "$NIX_FILE" | head -1 \
          | sed -E 's/hash = "(.*)"/\1/')
        LOCK_FILE="$FOLDER/package-lock.json"
        echo "    stale: $NIX_FILE:$HASH_LINE $OLD_HASH -> $NEW_HASH"
        STALE=1

        if [ -n "$LINK_REPO" ] && [ -n "$LINK_SHA" ]; then
          NIX_URL="$LINK_SERVER/$LINK_REPO/blob/$LINK_SHA/$NIX_FILE#L$HASH_LINE"
          LOCK_URL="$LINK_SERVER/$LINK_REPO/blob/$LINK_SHA/$LOCK_FILE"
          REPORT+="- [\`$NIX_FILE:$HASH_LINE\`]($NIX_URL) (\`.#$ATTR\`): \`$OLD_HASH\` → \`$NEW_HASH\` — lockfile: [\`$LOCK_FILE\`]($LOCK_URL)"$'\n'
        else
          REPORT+="- \`$NIX_FILE:$HASH_LINE\` (\`.#$ATTR\`): \`$OLD_HASH\` → \`$NEW_HASH\`"$'\n'
        fi

        if [ "$MODE" = "--apply" ]; then
          sed -i "s|hash = \"sha256-[^\"]*\";|hash = \"$NEW_HASH\";|" "$NIX_FILE"
          if ! nix build ".#$ATTR.npmDeps" --no-link --print-build-logs; then
            echo "    verification build failed after hash update" >&2
            exit 1
          fi
          FIXED=1
          echo "    fixed"
        fi
      done

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
        echo "Stale lockfile hashes detected. Run:"
        echo "  nix run .#fix-lockfiles"
        exit 1
      fi

      exit 0
    '';
}
