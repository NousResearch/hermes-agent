{
  # electron deps
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

  # sandbox deps
  bash,
  bubblewrap,
  cacert,
  coreutils,
  curl,
  gawk,
  git,
  glibc,
  gnumake,
  gnugrep,
  gnused,
  gzip,
  nodejs_22,
  openssl,
  python3,
  slirp4netns,
  stdenv,
  gnutar,
  util-linux,

  # etc
  writeShellApplication,
  lib,
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
  runtimeInputs = [
    bash
    bubblewrap
    cacert
    coreutils
    curl
    gawk
    git
    glibc.bin
    gnumake
    gnugrep
    gnused
    gzip
    nodejs_22
    openssl
    python3
    slirp4netns
    stdenv.cc
    gnutar
    util-linux
  ]
  ++ electronRuntime;
  text = ''
    export DEV_SANDBOX_REAL_CA_CERT=${cacert}/etc/ssl/certs/ca-bundle.crt
    export DEV_SANDBOX_DYNAMIC_LINKER=${stdenv.cc.bintools.dynamicLinker}
    export DEV_SANDBOX_NODE_DIR=${nodejs_22}
    export DEV_SANDBOX_ELECTRON_LD_LIBRARY_PATH=${lib.makeLibraryPath electronRuntime}
    # The script is imported into the store as a single file, so its own
    # directory has no scripts/sandbox/ beside it. Point it at the assets
    # (fake-internet proxy, ssh shim) explicitly.
    export DEV_SANDBOX_ASSETS=${../scripts/sandbox}
    exec ${../scripts/dev-sandbox.sh} "$@"
  '';
}
