# Derived build image: the digest-pinned termux-docker base (same pin as
# pm/lock.json's termux-docker row -- build_builder_image.sh injects it via
# --build-arg) with the wheelhouse build toolchain pre-installed.
#
# The provisioning block below mirrors the in-container half of
# termux_build.sh EXACTLY (official-mirror pin, package list): the runtime
# probe there (`command -v clang ...`) makes the baked toolchain a no-op,
# so this image is a drop-in replacement that skips the ~10-minute
# per-run toolchain apt on arm runners.
#
# The termux image has no /bin/sh (termux's shell lives at $PREFIX/bin/sh),
# so every RUN goes through the termux bash explicitly.
ARG BASE
FROM ${BASE}

# The termux rootfs is owned by uid 1000 (the `system` user) and the
# termux-patched apt REFUSES to run as uid 0. The base image's ENTRYPOINT
# normally drops root to system via su -- but docker build bypasses the
# entrypoint, so we must become the owning user ourselves.
USER 1000:1000
SHELL ["/data/data/com.termux/files/usr/bin/bash", "-c"]
ENV PREFIX=/data/data/com.termux/files/usr
ENV PATH="${PREFIX}/bin:/usr/bin:/bin"
ENV DEBIAN_FRONTEND=noninteractive

RUN printf '%s\n' "deb https://packages.termux.dev/apt/termux-main stable main" \
      > "${PREFIX}/etc/apt/sources.list" \
 && rm -f "${PREFIX}/etc/apt/sources.list.d/"*.list \
 && (apt update || apt update) \
 && apt install -y \
      clang rust make git patchelf binutils pkg-config protobuf cmake ninja \
      autoconf automake libtool \
      libandroid-posix-semaphore libandroid-support libbz2 libffi \
      libjpeg-turbo libpng freetype libtiff libwebp openjpeg littlecms \
      libyaml openssl readline zlib liblzma libsqlite ncurses

# No /bin/sh link here: uid 1000 cannot write /, and the runtime wheelhouse
# phase links it itself (termux_build.sh) inside its --tmpfs /bin, which
# docker mounts owned by the image USER.
