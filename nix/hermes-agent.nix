# nix/hermes-agent.nix — Overridable Hermes Agent package
#
# callPackage auto-wires nixpkgs args; flake inputs are passed explicitly.
# Users override via: pkgs.hermes-agent.override { extraPythonPackages = [...]; }
{
  lib,
  stdenv,
  makeWrapper,
  callPackage,
  python312,
  nodejs_22,
  ripgrep,
  git,
  openssh,
  ffmpeg,
  tirith,
  # Flake inputs — passed explicitly by packages.nix and overlays.nix
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  npm-lockfile-fix,
  # Overridable parameters
  extraPythonPackages ? [ ],
}:
let
  hermesVenv = callPackage ./python.nix {
    inherit uv2nix pyproject-nix pyproject-build-systems;
  };

  hermesNpmLib = callPackage ./lib.nix {
    inherit npm-lockfile-fix;
  };

  hermesTui = callPackage ./tui.nix {
    inherit hermesNpmLib;
  };

  hermesWeb = callPackage ./web.nix {
    inherit hermesNpmLib;
  };

  bundledSkills = lib.cleanSourceWith {
    src = ../skills;
    filter = path: _type: !(lib.hasInfix "/index-cache/" path);
  };

  runtimeDeps = [
    nodejs_22
    ripgrep
    git
    openssh
    ffmpeg
    tirith
  ];

  runtimePath = lib.makeBinPath runtimeDeps;

  sitePackagesPath = python312.sitePackages;

  # Walk propagatedBuildInputs to include transitive Python deps in PYTHONPATH.
  # Without this, a plugin listing e.g. requests as a dep would fail at runtime
  # if requests isn't already in the sealed uv2nix venv.
  allExtraPythonPackages = python312.pkgs.requiredPythonModules extraPythonPackages;

  pythonPath = lib.makeSearchPath sitePackagesPath allExtraPythonPackages;

  pyprojectHash = builtins.hashString "sha256" (builtins.readFile ../pyproject.toml);
  uvLockHash =
    if builtins.pathExists ../uv.lock then
      builtins.hashString "sha256" (builtins.readFile ../uv.lock)
    else
      "none";
in
stdenv.mkDerivation {
  pname = "hermes-agent";
  version = (builtins.fromTOML (builtins.readFile ../pyproject.toml)).project.version;

  dontUnpack = true;
  dontBuild = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/hermes-agent $out/bin
    cp -r ${bundledSkills} $out/share/hermes-agent/skills
    cp -r ${hermesWeb} $out/share/hermes-agent/web_dist

    mkdir -p $out/ui-tui
    cp -r ${hermesTui}/lib/hermes-tui/* $out/ui-tui/

    ${lib.concatMapStringsSep "\n"
      (name: ''
        makeWrapper ${hermesVenv}/bin/${name} $out/bin/${name} \
          --suffix PATH : "${runtimePath}" \
          --set HERMES_BUNDLED_SKILLS $out/share/hermes-agent/skills \
          --set HERMES_WEB_DIST $out/share/hermes-agent/web_dist \
          --set HERMES_TUI_DIR $out/ui-tui \
          --set HERMES_PYTHON ${hermesVenv}/bin/python3 \
          --set HERMES_NODE ${nodejs_22}/bin/node \
          ${lib.optionalString (extraPythonPackages != [ ]) ''--suffix PYTHONPATH : "${pythonPath}"''}
      '')
      [
        "hermes"
        "hermes-agent"
        "hermes-acp"
      ]
    }

    ${lib.optionalString (extraPythonPackages != [ ]) ''
      echo "=== Checking for plugin/core package collisions ==="
      ${hermesVenv}/bin/python3 -c "
import pathlib, sys, re

def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()

# Collect core venv package names
core = set()
venv_sp = pathlib.Path('${hermesVenv}/${sitePackagesPath}')
for di in venv_sp.glob('*.dist-info'):
    meta = di / 'METADATA'
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line.startswith('Name:'):
                core.add(canonical(line.split(':', 1)[1].strip()))
                break

# Check each extra package for collisions
extras_dirs = [${lib.concatMapStringsSep ", " (p: "'${toString p}'") allExtraPythonPackages}]
for edir in extras_dirs:
    sp = pathlib.Path(edir) / '${sitePackagesPath}'
    if not sp.exists():
        continue
    for di in sp.glob('*.dist-info'):
        meta = di / 'METADATA'
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            if line.startswith('Name:'):
                pkg = canonical(line.split(':', 1)[1].strip())
                if pkg in core:
                    print(f'ERROR: plugin package \"{pkg}\" collides with a package in hermes sealed venv', file=sys.stderr)
                    print(f'  from: {di}', file=sys.stderr)
                    print(f'  Remove this dependency from extraPythonPackages.', file=sys.stderr)
                    sys.exit(1)
                break

print('No collisions found.')
      "
      echo "=== No collisions ==="
    ''}

    runHook postInstall
  '';

  passthru = {
    inherit hermesTui hermesWeb hermesNpmLib hermesVenv;

    devShellHook = ''
      STAMP=".nix-stamps/hermes-agent"
      STAMP_VALUE="${pyprojectHash}:${uvLockHash}"
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        echo "hermes-agent: installing Python dependencies..."
        uv venv .venv --python ${python312}/bin/python3 2>/dev/null || true
        source .venv/bin/activate
        uv pip install -e ".[all]"
        [ -d mini-swe-agent ] && uv pip install -e ./mini-swe-agent 2>/dev/null || true
        [ -d tinker-atropos ] && uv pip install -e ./tinker-atropos 2>/dev/null || true
        mkdir -p .nix-stamps
        echo "$STAMP_VALUE" > "$STAMP"
      else
        source .venv/bin/activate
        export HERMES_PYTHON=${hermesVenv}/bin/python3
      fi
    '';
  };

  meta = with lib; {
    description = "AI agent with advanced tool-calling capabilities";
    homepage = "https://github.com/NousResearch/hermes-agent";
    mainProgram = "hermes";
    license = licenses.mit;
    platforms = platforms.unix;
  };
}
