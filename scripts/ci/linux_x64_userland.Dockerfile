FROM ubuntu:22.04 AS floor
# A test input for the offline smoke, not a desktop dependency or release asset.
ADD --checksum=sha256:270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d https://huggingface.co/ggml-org/models-moved/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf /stories260K.gguf
RUN chmod 0444 /stories260K.gguf
ARG UID_BUILDER
ARG GID_BUILDER
ENV DEBIAN_FRONTEND=noninteractive BOOTSTRAP_PYTHON=python3.11 ALSA_PACKAGE=libasound2
COPY install.sh /install.sh
RUN bash -eo pipefail /install.sh && apt-get install -yqq --no-install-recommends strace xvfb xauth fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /work /build-home && chown "$UID_BUILDER:$GID_BUILDER" /work /build-home

FROM floor AS builder
# The pilk sdist uses Clang; it is not a dependency of the packaged runtime.
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends clang && rm -rf /var/lib/apt/lists/*

FROM ubuntu:24.04 AS gpu
ARG UID_BUILDER
ARG GID_BUILDER
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends ca-certificates python3 libstdc++6 libssl3t64 git xz-utils strace && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /work /build-home && chown "$UID_BUILDER:$GID_BUILDER" /work /build-home
