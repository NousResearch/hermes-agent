{
  callPackage,
  nodejs_24,
  symlinkJoin,
}:
let
  npm12 = callPackage ./npm-12-0-2.nix { };
in
symlinkJoin {
  name = "nodejs-24-npm-12";
  paths = [
    npm12
    nodejs_24
  ];
  inherit (nodejs_24) meta passthru;
}
