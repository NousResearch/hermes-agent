{
  stdenv,
  makeWrapper,
  fetchurl,
  nodejs_24,
}:
stdenv.mkDerivation rec {
  pname = "npm";
  version = "12.0.2";

  src = fetchurl {
    url = "https://registry.npmjs.org/npm/-/npm-${version}.tgz";
    hash = "sha256-XbuGxx0HoZV/LpBzQJLdali9zZ68LY1ByhxuaiHTZOE=";
  };

  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;

  installPhase = ''
    mkdir -p $out/lib/npm12
    cp -r . $out/lib/npm12/
    mkdir -p $out/bin

    makeWrapper ${nodejs_24}/bin/node $out/bin/npm \
      --add-flags "$out/lib/npm12/bin/npm-cli.js"
    makeWrapper ${nodejs_24}/bin/node $out/bin/npx \
      --add-flags "$out/lib/npm12/bin/npx-cli.js"
  '';
}
