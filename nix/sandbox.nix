{
  # Electron needs its native runtime libraries on LD_LIBRARY_PATH when the
  # sandboxed command launches the desktop app (`sandbox hermes desktop`,
  # `sandbox npm run dev`); nothing else in the sandbox is nix-specific.
  alsa-lib,
  at-spi2-atk,
  atk,
  cairo,
  cups,
  dbus,
  expat,
  fontconfig,
  freetype,
  glib,
  gtk3,
  libdrm,
  libgbm,
  libxkbcommon,
  mesa,
  nspr,
  nss,
  pango,
  systemd,
  libX11,
  libXcomposite,
  libXdamage,
  libXext,
  libXfixes,
  libXrandr,
  libXrender,
  libXtst,
  libxcb,

  writeShellApplication,
  lib,
  stdenv,
}:
let
  electronRuntime = [
    alsa-lib
    at-spi2-atk
    atk
    cairo
    cups
    dbus
    expat
    fontconfig
    freetype
    glib
    gtk3
    libdrm
    libgbm
    libxkbcommon
    mesa
    nspr
    nss
    pango
    systemd
    libX11
    libXcomposite
    libXdamage
    libXext
    libXfixes
    libXrandr
    libXrender
    libXtst
    libxcb
  ];
in
writeShellApplication {
  name = "sandbox";
  # darwin's Electron is self-contained, and these libraries are Linux-only.
  text = lib.optionalString stdenv.hostPlatform.isLinux ''
    export LD_LIBRARY_PATH=${lib.makeLibraryPath electronRuntime}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  '' + ''
    exec ${../scripts/dev-sandbox.sh} "$@"
  '';
}
