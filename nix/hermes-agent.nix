{
  lib,
  python311,
  callPackage,
  nodejs_20,
  ripgrep,
  git,
  openssh,
  ffmpeg,
  tirith,
  stdenv,
  makeWrapper,

  # inputs
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:

let
  hermesVenv = callPackage ./python.nix {
    inherit uv2nix pyproject-nix pyproject-build-systems;
    python = python311;
  };

  # Import bundled skills, excluding runtime caches
  bundledSkills = lib.cleanSourceWith {
    src = ../skills;
    filter = path: _type: !(lib.hasInfix "/index-cache/" path);
  };

  runtimeDeps = [
    nodejs_20
    ripgrep
    git
    openssh
    ffmpeg
    tirith
  ];

  runtimePath = lib.makeBinPath runtimeDeps;
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

    ${lib.concatMapStringsSep "\n"
      (name: ''
        makeWrapper ${hermesVenv}/bin/${name} $out/bin/${name} \
          --suffix PATH : "${runtimePath}" \
          --set HERMES_BUNDLED_SKILLS $out/share/hermes-agent/skills
      '')
      [
        "hermes"
        "hermes-agent"
        "hermes-acp"
      ]
    }

    runHook postInstall
  '';

  meta = {
    description = "AI agent with advanced tool-calling capabilities";
    homepage = "https://github.com/NousResearch/hermes-agent";
    mainProgram = "hermes";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
