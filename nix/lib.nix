# nix/lib.nix — Shared helpers for nix stuff
{ pkgs, npm-lockfile-fix }:
{
  # Shell script that refreshes node_modules, fixes the lockfile, and
  # rewrites the `hash = "sha256-..."` line in the given nix file so
  # fetchNpmDeps picks up the new package-lock.json.
  mkUpdateLockfileScript =
    {
      name, # script binary name, e.g. "update_tui_lockfile"
      folder, # repo-relative folder with package.json, e.g. "ui-tui"
      nixFile, # repo-relative nix file with the hash line, e.g. "nix/tui.nix"
      attr, # flake package attr to build to cause the failure, e.g. "tui"
    }:
    pkgs.writeShellScriptBin name ''
      set -euox pipefail

      REPO_ROOT=$(git rev-parse --show-toplevel)

      cd "$REPO_ROOT/${folder}"
      rm -rf node_modules/
      npm cache clean --force
      CI=true npm install
      ${pkgs.lib.getExe npm-lockfile-fix} ./package-lock.json

      NIX_FILE="$REPO_ROOT/${nixFile}"
      sed -i "s/hash = \"[^\"]*\";/hash = \"\";/" $NIX_FILE
      NIX_OUTPUT=$(nix build .#${attr} 2>&1 || true)
      NEW_HASH=$(echo "$NIX_OUTPUT" | grep 'got:' | awk '{print $2}')
      echo got new hash $NEW_HASH
      sed -i "s|hash = \"[^\"]*\";|hash = \"$NEW_HASH\";|" $NIX_FILE
      nix build .#${attr}
      echo "Updated npm hash in $NIX_FILE to $NEW_HASH"
    '';

  # devShell bootstrap snippet: runs `npm install` in the target folder when
  # package.json or package-lock.json has changed since the last install.
  # Hashing happens in bash (not nix eval), and the post-install stamp is
  # recomputed so a lockfile that npm rewrites during install still matches.
  mkNpmDevShellHook =
    {
      name, # project-unique stampfile name, e.g. "hermes-tui"
      folder, # repo-relative folder with package.json + package-lock.json
    }:
    ''
      _hermes_npm_stamp() {
        sha256sum "${folder}/package.json" "${folder}/package-lock.json" \
          2>/dev/null | sha256sum | awk '{print $1}'
      }
      STAMP=".nix-stamps/${name}"
      STAMP_VALUE="$(_hermes_npm_stamp)"
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        echo "${name}: installing npm dependencies..."
        ( cd ${folder} && CI=true npm install --silent --no-fund --no-audit 2>/dev/null )
        mkdir -p .nix-stamps
        _hermes_npm_stamp > "$STAMP"
      fi
      unset -f _hermes_npm_stamp
    '';

  # Aggregate `fix-lockfiles` bin from a list of packages carrying
  #   passthru.npmLockfile = { attr; folder; nixFile; };
  # Invocations:
  #   fix-lockfiles --check   # exit 1 if any hash is stale
  #   fix-lockfiles --apply   # rewrite stale hashes in place
  # Writes machine-readable fields (stale, changed, report) to $GITHUB_OUTPUT
  # when set, so CI workflows can post a sticky PR comment directly.
  mkFixLockfiles =
    {
      packages, # list of packages with passthru.npmLockfile
    }:
    let
      entries = map (p: p.passthru.npmLockfile) packages;
      entryArgs = pkgs.lib.concatMapStringsSep " " (
        e: "\"${e.attr}:${e.folder}:${e.nixFile}\""
      ) entries;
    in
    pkgs.writeShellScriptBin "fix-lockfiles" ''
      set -uo pipefail
      MODE="''${1:---check}"
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
          echo "    build failed with no hash mismatch:" >&2
          echo "$OUTPUT" | tail -40 >&2
          exit 1
        fi

        OLD_HASH=$(grep -oE 'hash = "sha256-[^"]+"' "$NIX_FILE" | head -1 \
          | sed -E 's/hash = "(.*)"/\1/')
        echo "    stale: $OLD_HASH -> $NEW_HASH"
        STALE=1
        REPORT+="- \`$NIX_FILE\` (\`.#$ATTR\`): \`$OLD_HASH\` -> \`$NEW_HASH\`"$'\n'

        if [ "$MODE" = "--apply" ]; then
          sed -i "s|hash = \"sha256-[^\"]*\";|hash = \"$NEW_HASH\";|" "$NIX_FILE"
          nix build ".#$ATTR.npmDeps" --no-link --print-build-logs
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
        echo "  nix run .#fix-lockfiles -- --apply"
        exit 1
      fi

      exit 0
    '';
}
