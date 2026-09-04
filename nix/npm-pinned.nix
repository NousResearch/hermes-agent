{
  stdenv,
  makeWrapper,
  fetchurl,
  nodejs_26,
}:
let
  # pm/lock.json is the one pin table. Every artifact carries its resolved
  # url + sha256, so nix consumes it as pure data — no version or url
  # knowledge lives on the nix side.
  lock = builtins.fromJSON (builtins.readFile ../pm/lock.json);
  pin = lock.packages.npm;
in
stdenv.mkDerivation {
  pname = "npm";
  version = pin.version;

  src = fetchurl {
    url = pin.artifacts.any.url;
    sha256 = pin.artifacts.any.sha256;
  };

  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;

  installPhase = ''
    mkdir -p $out/lib/npm
    cp -r . $out/lib/npm/
    mkdir -p $out/bin

    makeWrapper ${nodejs_26}/bin/node $out/bin/npm \
      --add-flags "$out/lib/npm/bin/npm-cli.js"
    makeWrapper ${nodejs_26}/bin/node $out/bin/npx \
      --add-flags "$out/lib/npm/bin/npx-cli.js"
  '';
}
